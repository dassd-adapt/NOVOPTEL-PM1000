"""
feature_extraction.py
=====================
All feature computation functions for the NOVOPTEL PM1000 Stokes polarimeter
classification pipeline.

S1, S2, S3 in CSV files are already normalised (values in [-1, 1]).
S0 is total intensity and is NOT used to re-normalise S1/S2/S3.
"""

import os
import re

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import linregress

# NumPy ≥2.0 renamed trapz -> trapezoid; support both versions
_trapz = getattr(np, "trapezoid", None) or np.trapz

# ---------------------------------------------------------------------------
# File name parser
# ---------------------------------------------------------------------------

_FNAME_RE = re.compile(
    r"pm1000_sop_2kHz_1min_"
    r"(?P<source>[^_]+)"               # source token (no underscores; may have hyphens)
    r"_(?P<event>NE|FS|VB|MB|TAP)"
    r"_(?P<wavelength>\d+)"
    r"_(?P<date>\d+)"
    r"_(?P<run>\d+)\.csv$",
    re.IGNORECASE,
)


def parse_filename(fname):
    """Parse source and event label from filename.

    Parameters
    ----------
    fname : str
        Filename (basename only) such as
        ``pm1000_sop_2kHz_1min_SP-AGIL_FS_1550_310326_1.csv``.

    Returns
    -------
    dict with keys: source, event, wavelength, date, run
    or None if the filename does not match the expected pattern.
    """
    basename = os.path.basename(fname)
    m = _FNAME_RE.search(basename)
    if m is None:
        return None
    return m.groupdict()


# ---------------------------------------------------------------------------
# CSV loader with optional time windowing
# ---------------------------------------------------------------------------

def load_csv(filepath, t_start=0, t_end=60):
    """Load a PM1000 CSV file and window it to [t_start, t_end] seconds.

    Parameters
    ----------
    filepath : str
        Path to CSV file.
    t_start, t_end : float
        Time window in seconds.

    Returns
    -------
    df : pd.DataFrame
        Windowed dataframe with columns Time_s, S0, S1, S2, S3.
    fs : float
        Estimated sampling frequency (Hz) = 1 / median(diff(Time_s)).
    """
    df = pd.read_csv(filepath)
    df = df[(df["Time_s"] >= t_start) & (df["Time_s"] <= t_end)].reset_index(drop=True)
    diffs = np.diff(df["Time_s"].values)
    fs = 1.0 / np.median(diffs) if len(diffs) > 0 else np.nan
    return df, fs


# ---------------------------------------------------------------------------
# DOP computation
# ---------------------------------------------------------------------------

def compute_dop(df):
    """Compute Degree of Polarisation at each sample.

    DOP(t) = sqrt(S1^2 + S2^2 + S3^2)   [S1..S3 already normalised]

    Returns
    -------
    dop : np.ndarray, shape (N,)
    """
    s1 = df["S1"].values.astype(float)
    s2 = df["S2"].values.astype(float)
    s3 = df["S3"].values.astype(float)
    return np.sqrt(s1 ** 2 + s2 ** 2 + s3 ** 2)


# ---------------------------------------------------------------------------
# Gated unit Stokes
# ---------------------------------------------------------------------------

def compute_shat(df, dop, dop_thresh=0.2):
    """Compute unit SOP direction, gated by DOP threshold.

    shat[i] = [S1, S2, S3][i] / DOP[i]  if DOP[i] > dop_thresh, else NaN row.

    Returns
    -------
    shat : np.ndarray, shape (N, 3)
        NaN rows where DOP <= dop_thresh.
    """
    s1 = df["S1"].values.astype(float)
    s2 = df["S2"].values.astype(float)
    s3 = df["S3"].values.astype(float)
    stokes = np.column_stack([s1, s2, s3])

    shat = np.full_like(stokes, np.nan)
    valid = dop > dop_thresh
    shat[valid] = stokes[valid] / dop[valid, np.newaxis]
    return shat


# ---------------------------------------------------------------------------
# Step angles (gated)
# ---------------------------------------------------------------------------

def compute_step_angles(shat):
    """Compute step-to-step angle on the Poincaré sphere.

    step_deg[n] = arccos(shat[n] . shat[n-1]) in degrees.
    NaN if either sample is a NaN row.

    Returns
    -------
    step_deg : np.ndarray, shape (N-1,)
    """
    n = shat.shape[0]
    step_deg = np.full(n - 1, np.nan)
    for i in range(n - 1):
        a = shat[i]
        b = shat[i + 1]
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            continue
        dot = np.clip(np.dot(a, b), -1.0, 1.0)
        step_deg[i] = np.degrees(np.arccos(dot))
    return step_deg


# ---------------------------------------------------------------------------
# Theta to reference (gated)
# ---------------------------------------------------------------------------

def compute_theta_ref(shat, sref):
    """Compute angle between each SOP sample and the reference direction.

    theta_ref[i] = arccos(shat[i] . sref) in degrees.
    NaN where shat row is NaN.

    Parameters
    ----------
    shat : np.ndarray, shape (N, 3)
    sref : np.ndarray, shape (3,)

    Returns
    -------
    theta_ref : np.ndarray, shape (N,)
    """
    sref_unit = sref / (np.linalg.norm(sref) + 1e-12)
    dots = shat @ sref_unit  # (N,); NaN rows give NaN dot products
    dots = np.clip(dots, -1.0, 1.0)
    return np.degrees(np.arccos(dots))


# ---------------------------------------------------------------------------
# Helper: robust MAD
# ---------------------------------------------------------------------------

def _nanmad(x):
    """Median Absolute Deviation ignoring NaNs."""
    med = np.nanmedian(x)
    return np.nanmedian(np.abs(x - med))


# ---------------------------------------------------------------------------
# Baseline computation (per source, from NE files)
# ---------------------------------------------------------------------------

def compute_source_baseline(ne_filepaths, dop_thresh=0.2, t_start=0, t_end=60):
    """Compute per-source reference SOP and step-angle noise floor from NE files.

    Parameters
    ----------
    ne_filepaths : list of str
        All NE CSV files for a single source.
    dop_thresh : float
    t_start, t_end : float

    Returns
    -------
    sref : np.ndarray, shape (3,)
        Normalised mean reference SOP.
    floor_deg : float
        Noise floor in degrees = nanmedian + 3 * nanMAD of all NE step angles.
    mad_val : float
        nanMAD of all NE step angles.
    """
    all_shat_rows = []
    all_step_deg = []

    for fp in ne_filepaths:
        df, _fs = load_csv(fp, t_start=t_start, t_end=t_end)
        dop = compute_dop(df)
        shat = compute_shat(df, dop, dop_thresh=dop_thresh)
        step = compute_step_angles(shat)

        valid_rows = shat[~np.any(np.isnan(shat), axis=1)]
        all_shat_rows.append(valid_rows)
        all_step_deg.append(step)

    if all_shat_rows:
        combined = np.vstack(all_shat_rows)
        mean_vec = np.nanmean(combined, axis=0)
        norm = np.linalg.norm(mean_vec)
        sref = mean_vec / norm if norm > 1e-12 else mean_vec
    else:
        sref = np.array([1.0, 0.0, 0.0])

    if all_step_deg:
        combined_steps = np.concatenate(all_step_deg)
    else:
        combined_steps = np.array([np.nan])

    mad_val = _nanmad(combined_steps)
    floor_deg = np.nanmedian(combined_steps) + 3.0 * mad_val

    return sref, float(floor_deg), float(mad_val)


# ---------------------------------------------------------------------------
# Main feature extraction function
# ---------------------------------------------------------------------------

def extract_features(filepath, sref, floor_deg, mad_val, fs,
                     dop_thresh=0.2, t_start=0, t_end=60):
    """Extract a dict of scalar features from one CSV file.

    Parameters
    ----------
    filepath : str
    sref : np.ndarray, shape (3,)
        Per-source reference unit SOP from NE files.
    floor_deg : float
        Baseline noise floor in degrees.
    mad_val : float
        Baseline MAD value in degrees.
    fs : float
        Sampling rate (Hz).
    dop_thresh : float
    t_start, t_end : float

    Returns
    -------
    feats : dict of scalar feature values.
    """
    df, _fs_est = load_csv(filepath, t_start=t_start, t_end=t_end)

    # Use provided fs (from baseline NE), fall back to per-file estimate
    if np.isnan(fs) or fs <= 0:
        fs = _fs_est

    dop = compute_dop(df)
    shat = compute_shat(df, dop, dop_thresh=dop_thresh)
    step_deg = compute_step_angles(shat)
    theta_ref = compute_theta_ref(shat, sref)

    feats = {}

    # ------------------------------------------------------------------
    # DOP features (always valid)
    # ------------------------------------------------------------------
    feats["mean_dop"] = float(np.mean(dop))
    feats["std_dop"] = float(np.std(dop))
    feats["median_dop"] = float(np.median(dop))
    feats["frac_dop_above"] = float(np.mean(dop > dop_thresh))

    # ------------------------------------------------------------------
    # SOP motion features (gated; NaN for fully depolarised sources)
    # ------------------------------------------------------------------
    feats["max_theta_ref"] = float(np.nanmax(theta_ref)) if not np.all(np.isnan(theta_ref)) else np.nan
    feats["mean_theta_ref"] = float(np.nanmean(theta_ref))
    feats["rms_theta_ref"] = float(np.sqrt(np.nanmean(theta_ref ** 2)))
    feats["std_theta_ref"] = float(np.nanstd(theta_ref))

    feats["max_step"] = float(np.nanmax(step_deg)) if not np.all(np.isnan(step_deg)) else np.nan
    feats["mean_step"] = float(np.nanmean(step_deg))
    feats["rms_step"] = float(np.sqrt(np.nanmean(step_deg ** 2)))
    feats["std_step"] = float(np.nanstd(step_deg))
    feats["cum_arc"] = float(np.nansum(step_deg))

    excess_step = np.where(step_deg > floor_deg, step_deg - floor_deg, 0.0)
    # NaN propagation: keep NaN where step_deg is NaN
    excess_step = np.where(np.isnan(step_deg), np.nan, excess_step)
    feats["excess_cum_arc"] = float(np.nansum(excess_step))

    non_nan_steps = step_deg[~np.isnan(step_deg)]
    feats["frac_above_floor"] = (float(np.mean(non_nan_steps > floor_deg))
                                 if len(non_nan_steps) > 0 else np.nan)

    # ------------------------------------------------------------------
    # Event-specific features
    # ------------------------------------------------------------------
    fs_threshold = floor_deg + 5.0 * mad_val
    feats["fs_jump_count"] = int(np.nansum(step_deg > fs_threshold))

    tap_excess = np.where(step_deg > floor_deg, step_deg - floor_deg, 0.0)
    tap_excess = np.where(np.isnan(step_deg), 0.0, tap_excess)
    feats["tap_spike_energy"] = float(np.nansum(tap_excess ** 2))

    # mb_drift_slope: linear fit theta_ref(t) ~ a*t + b on valid samples
    valid_mask = ~np.isnan(theta_ref)
    t_arr = df["Time_s"].values.astype(float)
    t_valid = t_arr[valid_mask]
    tr_valid = theta_ref[valid_mask]
    if len(t_valid) >= 20:
        slope, _intercept, _r, _p, _se = linregress(t_valid, tr_valid)
        feats["mb_drift_slope"] = float(slope)
    else:
        feats["mb_drift_slope"] = 0.0

    # ------------------------------------------------------------------
    # Vibration / frequency features
    # ------------------------------------------------------------------
    # Signal: x(t) = ||shat(t) - sref||  (distance from baseline mean SOP)
    valid_shat_mask = ~np.any(np.isnan(shat), axis=1)
    n_valid = int(np.sum(valid_shat_mask))

    if n_valid >= 256:
        e = shat[valid_shat_mask] - sref  # (M, 3)
        x = np.sqrt(np.sum(e ** 2, axis=1))  # (M,)
        x = x - np.mean(x)  # detrend mean

        nperseg = min(256, n_valid)
        f_psd, pxx = signal.welch(x, fs=fs, nperseg=nperseg)
        f_psd = f_psd.ravel()
        pxx = pxx.ravel()

        # Narrow bandpower 75-85 Hz
        idx_narrow = (f_psd >= 75) & (f_psd <= 85)
        feats["vb_bp_narrow"] = (float(_trapz(pxx[idx_narrow], f_psd[idx_narrow]))
                                 if idx_narrow.any() else np.nan)

        # Wide bandpower 10-200 Hz
        idx_wide = (f_psd >= 10) & (f_psd <= 200)
        feats["vb_bp_wide"] = (float(_trapz(pxx[idx_wide], f_psd[idx_wide]))
                               if idx_wide.any() else np.nan)

        # Peak frequency and prominence in 10-200 Hz
        if idx_wide.any():
            pxx_wide = pxx[idx_wide]
            f_wide = f_psd[idx_wide]
            peak_idx_arr, props = signal.find_peaks(pxx_wide,
                                                    prominence=0.0)
            if len(peak_idx_arr) > 0:
                best = np.argmax(pxx_wide[peak_idx_arr])
                feats["vb_peak_freq"] = float(f_wide[peak_idx_arr[best]])
                feats["vb_peak_prominence"] = float(props["prominences"][best])
            else:
                feats["vb_peak_freq"] = np.nan
                feats["vb_peak_prominence"] = np.nan
        else:
            feats["vb_peak_freq"] = np.nan
            feats["vb_peak_prominence"] = np.nan
    else:
        feats["vb_bp_narrow"] = np.nan
        feats["vb_bp_wide"] = np.nan
        feats["vb_peak_freq"] = np.nan
        feats["vb_peak_prominence"] = np.nan

    # ------------------------------------------------------------------
    # S0 intensity stability features
    # ------------------------------------------------------------------
    s0 = df["S0"].values.astype(float)
    feats["mean_s0"] = float(np.mean(s0))
    feats["std_s0"] = float(np.std(s0))
    feats["cv_s0"] = float(feats["std_s0"] / feats["mean_s0"]) if feats["mean_s0"] != 0 else 0.0

    return feats
