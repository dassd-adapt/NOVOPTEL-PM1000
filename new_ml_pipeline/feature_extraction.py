"""
feature_extraction.py  —  new_ml_pipeline
==========================================
1-second non-overlapping windowed feature extraction for the NOVOPTEL PM1000
SOP dataset.

Design principles
-----------------
* Every window computes ALL Time Domain (TD) AND Frequency Domain (FD) features
  unconditionally — no gating on source type or event type.
* ``is_modulated`` is added as a numeric *input feature* (0 or 1) so that
  classifiers can learn modulation-dependence; it is NOT used to skip FD features.
* Welch PSD uses ``nperseg=512`` for 1 Hz frequency resolution at 2 kHz sampling,
  giving well-resolved 80 Hz VB peaks.

Physical justification of key features
---------------------------------------
* VB (vibration)  : strong narrowband 80 Hz peak →
  ``vb_snr_80hz_db`` + ``spectral_entropy`` are the primary FD discriminators.
  With 1 s windows at 2 kHz, frequency resolution = 1 Hz — the 80 Hz peak is
  fully resolved.
* TAP (tapping)   : impulsive, high-kurtosis SOP bursts →
  ``kurtosis_step`` + ``burst_count`` are the primary TD discriminators.
* NE (no event)   : low step angles, flat PSD, DOP ≈ 1 for SP sources →
  ``step_mean`` + ``spectral_entropy`` separate NE from active events.
* FS (fibre squeeze) : moderate step angles, broadband PSD →
  ``bp_mid`` + ``step_rms`` are key.
* MB (macro-bend)    : slow monotonic SOP drift →
  ``range_theta_ref`` + ``cum_arc`` + ``theta_max`` are key.
* DPQAM16 / DPQPSK   : DOP < 1 (partially polarised) →
  ``var_dop``, ``frac_dop_low``, ``is_modulated`` capture this.

Usage
-----
From the repository root::

    python new_ml_pipeline/feature_extraction.py
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

# ---------------------------------------------------------------------------
# Configurable parameters
# ---------------------------------------------------------------------------

WINDOW_S: float = 1.0        # 1-second window
OVERLAP: float = 0.0         # non-overlapping — one feature vector per second
MIN_SAMPLES: int = 50        # safety floor (1 s @ 2 kHz ≈ 2000 samples)

MIN_FILE_SIZE_BYTES: int = 1_000_000   # 1 MB — skip truncated files

# DOP gating threshold (PM1000 accuracy ≈ ±1%; 0.2 is well above noise floor)
DOP_GATE: float = 0.2

# PSD band edges (Hz)
BAND_LOW  = (1.0,   20.0)
BAND_MID  = (20.0,  100.0)
BAND_HIGH = (100.0, 500.0)

# Modulated source names (DOP < 1 for partially polarised light)
MODULATED_SOURCES = {"DPQAM16-200G", "DPQPSK-200G"}

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

REPO_ROOT   = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset-1603"
OUTPUT_CSV  = REPO_ROOT / "new_ml_pipeline" / "features_1s_windows.csv"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _geodesic_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the geodesic angle (degrees) between unit-vector rows in *a* and *b*.

    Inputs are (N, 3) arrays.  The dot-product is clipped to [-1, 1] before
    calling arccos to avoid NaN from floating-point rounding.
    """
    dot = np.einsum("ij,ij->i", a, b)
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def _unit_normalise(s123: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Defensively normalise (N, 3) Stokes rows to unit vectors.

    Returns ``(unit_vecs, dop)`` where rows with DOP == 0 are set to NaN.
    DOP = ‖[S1, S2, S3]‖ — the true optical DOP because the PM1000 stores
    vectors already S0-normalised in Standard Normalisation mode.
    """
    dop = np.linalg.norm(s123, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = s123 / dop[:, None]
    unit[dop == 0] = np.nan
    return unit, dop


def _band_power(freqs: np.ndarray, psd: np.ndarray, f_lo: float, f_hi: float) -> float:
    """Integrate *psd* between *f_lo* and *f_hi* Hz using the trapezoid rule."""
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if mask.sum() < 2:
        return 0.0
    try:
        _trapz = np.trapezoid
    except AttributeError:
        _trapz = np.trapz  # type: ignore[attr-defined]
    return float(_trapz(psd[mask], freqs[mask]))


def _spectral_entropy(psd: np.ndarray) -> float:
    """Shannon entropy of the normalised PSD.

    Low entropy indicates a narrowband peak (e.g. VB at 80 Hz);
    high entropy indicates broadband noise (NE, FS).
    """
    p = psd / (psd.sum() + 1e-12)
    return float(-np.sum(p * np.log(p + 1e-12)))

# ---------------------------------------------------------------------------
# File discovery and parsing
# ---------------------------------------------------------------------------

def discover_files(dataset_dir: Path) -> list[dict]:
    """Return a list of file-info dicts for all valid CSV files in *dataset_dir*.

    Each dict has keys: ``path``, ``source``, ``event``, ``date``, ``rep``.
    Files smaller than :data:`MIN_FILE_SIZE_BYTES` are skipped with a warning.
    """
    pattern = "pm1000_sop_2kHz_1min_*_*_1550_*_*.csv"
    records = []

    for csv_path in sorted(dataset_dir.glob(pattern)):
        size = csv_path.stat().st_size
        if size < MIN_FILE_SIZE_BYTES:
            print(
                f"  [SKIP] {csv_path.name}  ({size / 1024:.1f} KB < "
                f"{MIN_FILE_SIZE_BYTES // 1024} KB)"
            )
            continue

        stem  = csv_path.stem
        parts = stem.split("_")
        # prefix: pm1000, sop, 2kHz, 1min  → 4 tokens
        prefix_len    = 4
        suffix_tokens = parts[-3:]   # ['1550', DATE, REP]
        if len(suffix_tokens) != 3 or suffix_tokens[0] != "1550":
            print(f"  [SKIP] Cannot parse filename: {csv_path.name}")
            continue

        date_str = suffix_tokens[1]
        rep_str  = suffix_tokens[2]

        middle_tokens = parts[prefix_len:-3]
        event_tags    = {"NE", "FS", "VB", "MB", "TAP"}
        event_idx     = None
        for i, tok in enumerate(middle_tokens):
            if tok in event_tags:
                event_idx = i
                break

        if event_idx is None:
            print(f"  [SKIP] Cannot identify event tag: {csv_path.name}")
            continue

        source = "_".join(middle_tokens[:event_idx])
        event  = middle_tokens[event_idx]

        records.append(
            {
                "path":   csv_path,
                "source": source,
                "event":  event,
                "date":   date_str,
                "rep":    rep_str,
            }
        )

    return records

# ---------------------------------------------------------------------------
# Per-file loading
# ---------------------------------------------------------------------------

def load_and_normalise(file_info: dict) -> pd.DataFrame | None:
    """Load a CSV and return a DataFrame with defensively unit-normalised S1, S2, S3.

    S0 is intentionally NOT used to re-normalise — the PM1000 already stores
    S1/S2/S3 in Standard Normalisation mode (unit Poincaré sphere).
    """
    try:
        df = pd.read_csv(file_info["path"])
    except Exception as exc:
        print(f"  [ERROR] Could not read {file_info['path'].name}: {exc}")
        return None

    required_cols = {"Time_s", "S0", "S1", "S2", "S3"}
    if not required_cols.issubset(df.columns):
        print(f"  [ERROR] Missing columns in {file_info['path'].name}")
        return None

    df = df[["Time_s", "S0", "S1", "S2", "S3"]].copy()
    df.sort_values("Time_s", inplace=True)
    df.reset_index(drop=True, inplace=True)

    s123  = df[["S1", "S2", "S3"]].values.astype(float)
    norms = np.linalg.norm(s123, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    s123_unit = s123 / norms

    df["S1"] = s123_unit[:, 0]
    df["S2"] = s123_unit[:, 1]
    df["S3"] = s123_unit[:, 2]

    return df

# ---------------------------------------------------------------------------
# Feature extraction for a single window  (ALL TD + ALL FD, unconditionally)
# ---------------------------------------------------------------------------

def extract_window_features(
    win: pd.DataFrame,
    s_ref: np.ndarray,
) -> dict:
    """Compute all 41 features for a single 1-second window *win*.

    ALL Time Domain AND Frequency Domain features are computed unconditionally
    for every window, regardless of source type or event type.

    Parameters
    ----------
    win:
        Slice of the file DataFrame for this window.
    s_ref:
        (3,) unit vector — mean reference SOP computed from the **entire file**.
    """
    from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew

    t    = win["Time_s"].values.astype(float)
    s123 = win[["S1", "S2", "S3"]].values.astype(float)

    # Estimate sampling frequency from timestamps
    dt = np.diff(t)
    dt = dt[dt > 0]
    fs = 1.0 / np.median(dt) if len(dt) > 0 else 1.0

    # Unit vectors and DOP
    unit, dop = _unit_normalise(s123)

    # ------------------------------------------------------------------ #
    #  TIME DOMAIN FEATURES                                               #
    # ------------------------------------------------------------------ #

    # --- DOP features (4) ---------------------------------------------
    dop_mean     = float(np.nanmean(dop))
    dop_std      = float(np.nanstd(dop))
    dop_min      = float(np.nanmin(dop))
    dop_max      = float(np.nanmax(dop))

    # --- DOP spread / modulation features (3) -------------------------
    # var_dop: variance — key discriminator for DP vs SP sources
    # iqr_dop: robust spread
    # frac_dop_low: fraction with DOP < 0.7 — DP sources (DOP < 1)
    var_dop      = float(np.nanvar(dop))
    iqr_dop      = float(np.nanpercentile(dop, 75) - np.nanpercentile(dop, 25))
    frac_dop_low = float(np.mean(dop < 0.7))

    # DOP gating for SOP-based features
    gate         = dop > DOP_GATE
    unit_gated   = unit.copy()
    unit_gated[~gate] = np.nan

    # --- Step-angle features (8 + 2 higher-order + 1 burst + 1 cum + 1 autocorr) -
    n = len(unit_gated)
    if n >= 2:
        a = unit_gated[:-1]
        b = unit_gated[1:]
        valid = ~(np.any(np.isnan(a), axis=1) | np.any(np.isnan(b), axis=1))
        non_nan_steps = _geodesic_angle(a[valid], b[valid]) if valid.sum() > 0 else np.array([])
    else:
        non_nan_steps = np.array([])

    if len(non_nan_steps) > 0:
        step_mean  = float(np.nanmean(non_nan_steps))
        step_std   = float(np.nanstd(non_nan_steps))
        step_max   = float(np.nanmax(non_nan_steps))
        step_rms   = float(np.sqrt(np.nanmean(non_nan_steps ** 2)))
        step_p95   = float(np.nanpercentile(non_nan_steps, 95))
        step_p99   = float(np.nanpercentile(non_nan_steps, 99))
        cum_arc    = float(np.nansum(non_nan_steps))
    else:
        step_mean = step_std = step_max = step_rms = step_p95 = step_p99 = cum_arc = 0.0

    if len(non_nan_steps) > 3:
        kurtosis_step = float(scipy_kurtosis(non_nan_steps))
        skew_step     = float(scipy_skew(non_nan_steps))
    else:
        kurtosis_step = skew_step = 0.0

    # burst_count: number of step-angle samples exceeding 3× the median
    # Captures impulsive TAP events
    if len(non_nan_steps) > 0:
        median_step = float(np.median(non_nan_steps))
        burst_count = int(np.sum(non_nan_steps > 3.0 * median_step))
    else:
        burst_count = 0

    # step_autocorr_lag1: 80 Hz VB is periodic → autocorrelated
    if len(non_nan_steps) >= 2:
        s_c = non_nan_steps - non_nan_steps.mean()
        denom = float(np.dot(s_c, s_c))
        if denom > 0:
            step_autocorr_lag1 = float(np.dot(s_c[:-1], s_c[1:]) / denom)
        else:
            step_autocorr_lag1 = 0.0
    else:
        step_autocorr_lag1 = 0.0

    # --- θ_ref features (6) -------------------------------------------
    valid_rows = ~np.any(np.isnan(unit_gated), axis=1)
    if valid_rows.sum() > 0 and not np.any(np.isnan(s_ref)):
        ref_tiled    = np.tile(s_ref, (valid_rows.sum(), 1))
        theta        = _geodesic_angle(unit_gated[valid_rows], ref_tiled)
        theta_mean   = float(np.nanmean(theta))
        theta_std    = float(np.nanstd(theta))
        theta_max    = float(np.nanmax(theta))
        theta_rms    = float(np.sqrt(np.nanmean(theta ** 2)))
        range_theta_ref = float(np.nanmax(theta) - np.nanmin(theta))
        p95_theta_ref   = float(np.nanpercentile(theta, 95))
    else:
        theta_mean = theta_std = theta_max = theta_rms = 0.0
        range_theta_ref = p95_theta_ref = 0.0

    # --- Raw Stokes trajectory variability (6) -------------------------
    s1_std   = float(np.nanstd(s123[:, 0]))
    s2_std   = float(np.nanstd(s123[:, 1]))
    s3_std   = float(np.nanstd(s123[:, 2]))
    s1_range = float(np.nanmax(s123[:, 0]) - np.nanmin(s123[:, 0]))
    s2_range = float(np.nanmax(s123[:, 1]) - np.nanmin(s123[:, 1]))
    s3_range = float(np.nanmax(s123[:, 2]) - np.nanmin(s123[:, 2]))

    # ------------------------------------------------------------------ #
    #  FREQUENCY DOMAIN FEATURES  (computed unconditionally for ALL      #
    #  sources and events — no is_modulated gating)                      #
    # ------------------------------------------------------------------ #
    # x = displacement of SOP from the per-file reference: ‖ŝ - ŝ_ref‖
    # With 1 s windows at 2 kHz, nperseg=512 → frequency resolution = 1 Hz.
    # The 80 Hz VB peak is well-resolved; vb_snr_80hz_db is source-normalised
    # so it generalises across SP and DP sources.

    if valid_rows.sum() >= 2 and not np.any(np.isnan(s_ref)):
        x       = np.linalg.norm(unit_gated[valid_rows] - s_ref, axis=1)
        nperseg = min(512, len(x))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f_psd, pxx = welch(x, fs=fs, nperseg=nperseg)

        if len(pxx) > 0 and pxx.sum() > 0:
            peak_idx       = int(np.argmax(pxx))
            psd_peak_freq  = float(f_psd[peak_idx])
            psd_peak_power = float(pxx[peak_idx])

            bp_low  = _band_power(f_psd, pxx, *BAND_LOW)
            bp_mid  = _band_power(f_psd, pxx, *BAND_MID)
            bp_high = _band_power(f_psd, pxx, *BAND_HIGH)

            bp_ratio_mid_low   = bp_mid / (bp_low + 1e-12)
            psd_peak_sharpness = psd_peak_power / (float(np.mean(pxx)) + 1e-12)
            spectral_entropy   = _spectral_entropy(pxx)

            # vb_snr_80hz_db: SNR of 75–85 Hz band vs local noise floor
            # (10–200 Hz excluding 75–85 Hz).
            # Source-normalised → generalises across SP and DP sources.
            vb_band_idx  = (f_psd >= 75) & (f_psd <= 85)
            noise_idx    = (f_psd >= 10) & (f_psd <= 200) & ~vb_band_idx
            vb_power     = float(np.mean(pxx[vb_band_idx])) if vb_band_idx.any() else 0.0
            noise_floor  = float(np.mean(pxx[noise_idx]))   if noise_idx.any()  else 1e-12
            if noise_floor > 0 and vb_power > 0:
                vb_snr_80hz_db = float(10.0 * np.log10(vb_power / noise_floor))
            else:
                vb_snr_80hz_db = 0.0
        else:
            (psd_peak_freq, psd_peak_power,
             bp_low, bp_mid, bp_high,
             bp_ratio_mid_low, psd_peak_sharpness,
             spectral_entropy, vb_snr_80hz_db) = (0.0,) * 9
    else:
        (psd_peak_freq, psd_peak_power,
         bp_low, bp_mid, bp_high,
         bp_ratio_mid_low, psd_peak_sharpness,
         spectral_entropy, vb_snr_80hz_db) = (0.0,) * 9

    # ------------------------------------------------------------------ #
    #  Assemble feature row                                               #
    # ------------------------------------------------------------------ #
    row = {
        # TD — DOP (4)
        "dop_mean":     dop_mean,
        "dop_std":      dop_std,
        "dop_min":      dop_min,
        "dop_max":      dop_max,
        # TD — DOP spread / modulation (3)
        "var_dop":      var_dop,
        "iqr_dop":      iqr_dop,
        "frac_dop_low": frac_dop_low,
        # TD — step-angle statistics (6)
        "step_mean":    step_mean,
        "step_std":     step_std,
        "step_max":     step_max,
        "step_rms":     step_rms,
        "step_p95":     step_p95,
        "step_p99":     step_p99,
        # TD — higher-order step statistics (2)
        "kurtosis_step":      kurtosis_step,
        "skew_step":          skew_step,
        # TD — burst / arc / autocorr (3)
        "burst_count":        burst_count,
        "cum_arc":            cum_arc,
        "step_autocorr_lag1": step_autocorr_lag1,
        # TD — θ_ref features (6)
        "theta_mean":      theta_mean,
        "theta_std":       theta_std,
        "theta_max":       theta_max,
        "theta_rms":       theta_rms,
        "range_theta_ref": range_theta_ref,
        "p95_theta_ref":   p95_theta_ref,
        # TD — raw Stokes trajectory variability (6)
        "s1_std":   s1_std,
        "s2_std":   s2_std,
        "s3_std":   s3_std,
        "s1_range": s1_range,
        "s2_range": s2_range,
        "s3_range": s3_range,
        # FD — PSD features (9)
        "psd_peak_freq":      psd_peak_freq,
        "psd_peak_power":     psd_peak_power,
        "bp_low":             bp_low,
        "bp_mid":             bp_mid,
        "bp_high":            bp_high,
        "bp_ratio_mid_low":   bp_ratio_mid_low,
        "psd_peak_sharpness": psd_peak_sharpness,
        "spectral_entropy":   spectral_entropy,
        "vb_snr_80hz_db":     vb_snr_80hz_db,
    }

    # Replace any remaining NaN / inf with 0
    return {k: (0.0 if (not math.isfinite(v)) else v) for k, v in row.items()}

# ---------------------------------------------------------------------------
# Per-file windowing
# ---------------------------------------------------------------------------

def process_file(file_info: dict) -> list[dict]:
    """Segment one file into 1-second windows and extract all features.

    ``is_modulated`` is set at the row level based on the source field.
    """
    df = load_and_normalise(file_info)
    if df is None:
        return []

    t              = df["Time_s"].values.astype(float)
    t_start_file   = t[0]
    t_end_file     = t[-1]
    duration       = t_end_file - t_start_file

    if duration <= 0:
        return []

    # Per-file reference SOP — DOP-gated mean over the **entire file**
    s123_all         = df[["S1", "S2", "S3"]].values.astype(float)
    unit_all, dop_all = _unit_normalise(s123_all)
    gate_all         = dop_all > DOP_GATE
    unit_all[~gate_all] = np.nan

    valid_all = ~np.any(np.isnan(unit_all), axis=1)
    if valid_all.sum() > 0:
        raw_mean  = np.nanmean(unit_all[valid_all], axis=0)
        ref_norm  = np.linalg.norm(raw_mean)
        s_ref     = raw_mean / ref_norm if ref_norm > 0 else np.full(3, np.nan)
    else:
        s_ref = np.full(3, np.nan)

    # is_modulated: input feature for classifiers (NOT a gate for FD)
    is_modulated = 1 if file_info["source"] in MODULATED_SOURCES else 0

    step_s       = WINDOW_S * (1.0 - OVERLAP)   # = WINDOW_S since OVERLAP=0
    rows         = []
    window_idx   = 0
    win_t_start  = t_start_file

    while win_t_start + WINDOW_S <= t_end_file + 1e-9:
        win_t_end = win_t_start + WINDOW_S
        mask      = (t >= win_t_start) & (t < win_t_end)
        n_pts     = mask.sum()

        if n_pts < MIN_SAMPLES:
            win_t_start += step_s
            window_idx  += 1
            continue

        win_df = df.loc[mask].reset_index(drop=True)
        feats  = extract_window_features(win_df, s_ref)

        row = {
            "source":      file_info["source"],
            "event":       file_info["event"],
            "date":        file_info["date"],
            "replicate":   file_info["rep"],
            "window_idx":  window_idx,
            "t_start_s":   round(win_t_start, 4),
            "t_end_s":     round(win_t_end,   4),
            "is_modulated": is_modulated,
        }
        row.update(feats)
        rows.append(row)

        win_t_start += step_s
        window_idx  += 1

    return rows

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full feature extraction pipeline and write features_1s_windows.csv."""
    print("=" * 60)
    print("NOVOPTEL PM1000 — 1-Second Window Feature Extraction")
    print("=" * 60)
    print(f"Dataset directory : {DATASET_DIR}")
    print(f"Output CSV        : {OUTPUT_CSV}")
    print(
        f"Window parameters : {WINDOW_S}s window, {int(OVERLAP * 100)}% overlap, "
        f"min {MIN_SAMPLES} samples"
    )
    print("Features          : ALL TD + ALL FD (unconditional)")
    print()

    print("Discovering CSV files …")
    file_list = discover_files(DATASET_DIR)
    print(f"  Found {len(file_list)} valid file(s)\n")

    if not file_list:
        print("No files to process.  Exiting.")
        return

    all_rows: list[dict] = []
    summary: dict[str, dict] = {}

    for fi in file_list:
        event = fi["event"]
        if event not in summary:
            summary[event] = {"files": 0, "windows": 0}
        summary[event]["files"]   += 1

        rows = process_file(fi)
        all_rows.extend(rows)
        summary[event]["windows"] += len(rows)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Column order — metadata + is_modulated + 34 TD + 9 FD = 41 features
    columns = [
        "source", "event", "date", "replicate", "window_idx",
        "t_start_s", "t_end_s",
        "is_modulated",
        # TD — DOP (4)
        "dop_mean", "dop_std", "dop_min", "dop_max",
        # TD — DOP spread (3)
        "var_dop", "iqr_dop", "frac_dop_low",
        # TD — step-angle (6)
        "step_mean", "step_std", "step_max", "step_rms", "step_p95", "step_p99",
        # TD — higher-order step (2)
        "kurtosis_step", "skew_step",
        # TD — burst / arc / autocorr (3)
        "burst_count", "cum_arc", "step_autocorr_lag1",
        # TD — θ_ref (6)
        "theta_mean", "theta_std", "theta_max", "theta_rms",
        "range_theta_ref", "p95_theta_ref",
        # TD — raw Stokes (6)
        "s1_std", "s2_std", "s3_std", "s1_range", "s2_range", "s3_range",
        # FD (9)
        "psd_peak_freq", "psd_peak_power",
        "bp_low", "bp_mid", "bp_high",
        "bp_ratio_mid_low", "psd_peak_sharpness",
        "spectral_entropy", "vb_snr_80hz_db",
    ]

    out_df = pd.DataFrame(all_rows, columns=columns)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print("=== Feature Extraction Summary ===")
    print(f"{'Event':<8} {'Files':>6} {'Windows':>8}")
    print("-" * 26)
    total_windows = 0
    total_files   = 0
    for event in sorted(summary):
        f = summary[event]["files"]
        w = summary[event]["windows"]
        print(f"{event:<8} {f:>6} {w:>8}")
        total_windows += w
        total_files   += f

    print("-" * 26)
    print(f"{'TOTAL':<8} {total_files:>6} {total_windows:>8}")
    print()
    print(f"Total windows  : {total_windows} across {total_files} files")
    print(f"Features/window: 41 (TD + FD, unconditional)")
    print(f"Output         : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
