"""
ml_pipeline_universal.py
========================
Universal modular machine learning pipeline for the NOVOPTEL PM1000 optical
fiber SOP dataset (dataset-1603).  Processes one source type at a time.

Supported sources: 10GE, DPQAM16, DPQPSK, SP-PURE, SP-AGIL
Event classes    : NE (no event), FS (fiber side), MB (modal birefringence),
                   TAP (fiber tap), VB (vibration/bend)

Usage
-----
From the repository root::

    python ml_pipeline_universal.py \\
        --data_dir dataset-1603 \\
        --source   10GE \\
        --output_dir outputs_10GE_enhanced

    python ml_pipeline_universal.py \\
        --data_dir dataset-1603 \\
        --source   DPQAM16 \\
        --output_dir outputs_DPQAM16_enhanced

Full list of supported --source values::

    10GE  DPQAM16  DPQPSK  SP-PURE  SP-AGIL
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.signal import welch
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Constants and defaults
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
TRAIN_FRACTION = 0.20          # first 20 % of rows per file → training; remaining 80 % → test
WINDOW_S = 1.0                 # 1-second windows (no overlap)
EXPECTED_FS = 2000             # expected sampling rate (Hz)
FFT_PAD = 2000                 # zero-pad length for FFT
N_FFT_BINS = 1001              # bins 0 – 1000 Hz
DOP_GATE = 0.2                 # DOP threshold for gating
VB_FREQ = 80.0                 # Hz — VB SNR reference frequency
BAND_LOW = (1.0, 20.0)
BAND_MID = (20.0, 100.0)
BAND_HIGH = (100.0, 500.0)
BURST_THRESH_FACTOR = 2.0      # step-angle burst threshold = mean * factor
MIN_SAMPLES_PER_WIN = 100
MIN_FILE_SIZE_BYTES = 1_000_000

# Source-name aliases: CLI name → filename token(s)
SOURCE_ALIASES: dict[str, list[str]] = {
    "10GE":     ["10GE"],
    "DPQAM16":  ["DPQAM16-200G", "DPQAM16"],
    "DPQPSK":   ["DPQPSK-200G", "DPQPSK"],
    "SP-PURE":  ["SP-PURE"],
    "SP-AGIL":  ["SP-AGIL"],
}

EVENT_ORDER = ["NE", "FS", "MB", "TAP", "VB"]
N_MODELS = 3  # RF, XGB, SVM


# ---------------------------------------------------------------------------
# Logging setup (call once after output_dir is known)
# ---------------------------------------------------------------------------

def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ml_pipeline")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Helper math utilities
# ---------------------------------------------------------------------------

def _safe_trapz(y: np.ndarray, x: np.ndarray) -> float:
    try:
        return float(np.trapezoid(y, x))
    except AttributeError:
        return float(np.trapz(y, x))  # type: ignore[attr-defined]


def _geodesic_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geodesic angle (degrees) between (N,3) unit-vector rows."""
    dot = np.einsum("ij,ij->i", a, b)
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def _unit_normalise(s123: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (unit_vecs, dop) for (N,3) Stokes array."""
    dop = np.linalg.norm(s123, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = s123 / np.where(dop[:, None] < 1e-9, 1.0, dop[:, None])
    unit[dop == 0] = np.nan
    return unit, dop


def _band_power(f: np.ndarray, p: np.ndarray, f_lo: float, f_hi: float) -> float:
    mask = (f >= f_lo) & (f <= f_hi)
    if mask.sum() < 2:
        return 0.0
    return _safe_trapz(p[mask], f[mask])


def _spectral_entropy(p: np.ndarray) -> float:
    total = p.sum()
    if total <= 0:
        return 0.0
    pn = p / total
    pn = pn[pn > 0]
    return float(-np.sum(pn * np.log2(pn)))


def _autocorr_lag1(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    x = x - np.nanmean(x)
    denom = np.nansum(x ** 2)
    if denom == 0:
        return 0.0
    return float(np.nansum(x[:-1] * x[1:]) / denom)


# ---------------------------------------------------------------------------
# Feature extraction (1,035 features per 1-second window)
# ---------------------------------------------------------------------------

def extract_features(
    win_df: pd.DataFrame,
    fs: float,
    s_ref: np.ndarray,
) -> dict:
    """Compute 32 time-domain + 1,001 FFT features from one window.

    Returns a dict with keys: scalar feature names + 'fft_bin_0' … 'fft_bin_1000'.
    """
    t = win_df["Time_s"].values.astype(float)
    s123 = win_df[["S1", "S2", "S3"]].values.astype(float)

    unit, dop = _unit_normalise(s123)
    gate = dop > DOP_GATE
    unit_g = unit.copy()
    unit_g[~gate] = np.nan

    valid = ~np.any(np.isnan(unit_g), axis=1)

    # ------ DOP features (4) ------------------------------------------------
    dop_mean = float(np.nanmean(dop))
    dop_std = float(np.nanstd(dop))
    dop_iqr = float(np.nanpercentile(dop, 75) - np.nanpercentile(dop, 25))
    frac_dop_low = float(np.mean(dop < DOP_GATE))

    # ------ Step-angle features (9) -----------------------------------------
    n = len(unit_g)
    if n >= 2:
        a = unit_g[:-1]
        b = unit_g[1:]
        v = ~(np.any(np.isnan(a), axis=1) | np.any(np.isnan(b), axis=1))
        if v.sum() > 1:
            angles = _geodesic_angle(a[v], b[v])
            step_mean = float(np.nanmean(angles))
            step_std = float(np.nanstd(angles))
            step_max = float(np.nanmax(angles))
            step_p99 = float(np.nanpercentile(angles, 99))
            step_kurtosis = float(scipy_kurtosis(angles, bias=False)
                                  if len(angles) >= 4 else 0.0)
            step_skewness = float(scipy_skew(angles, bias=False)
                                  if len(angles) >= 3 else 0.0)
            burst_thresh = step_mean * BURST_THRESH_FACTOR
            burst_count = int(np.sum(angles > burst_thresh))
            cum_arc = float(np.nansum(angles))
            autocorr_lag1 = _autocorr_lag1(angles)
        else:
            step_mean = step_std = step_max = step_p99 = 0.0
            step_kurtosis = step_skewness = 0.0
            burst_count = 0
            cum_arc = 0.0
            autocorr_lag1 = 0.0
    else:
        step_mean = step_std = step_max = step_p99 = 0.0
        step_kurtosis = step_skewness = 0.0
        burst_count = 0
        cum_arc = 0.0
        autocorr_lag1 = 0.0

    # ------ Theta-ref features (4) ------------------------------------------
    if valid.sum() > 0 and not np.any(np.isnan(s_ref)):
        ref_tile = np.tile(s_ref, (valid.sum(), 1))
        theta = _geodesic_angle(unit_g[valid], ref_tile)
        theta_mean = float(np.nanmean(theta))
        theta_std = float(np.nanstd(theta))
        theta_range = float(np.nanmax(theta) - np.nanmin(theta))
        theta_p95 = float(np.nanpercentile(theta, 95))
    else:
        theta_mean = theta_std = theta_range = theta_p95 = 0.0

    # ------ Stokes variance (6) ---------------------------------------------
    s1_std = float(np.nanstd(s123[:, 0]))
    s2_std = float(np.nanstd(s123[:, 1]))
    s3_std = float(np.nanstd(s123[:, 2]))
    s1_range = float(np.nanmax(s123[:, 0]) - np.nanmin(s123[:, 0]))
    s2_range = float(np.nanmax(s123[:, 1]) - np.nanmin(s123[:, 1]))
    s3_range = float(np.nanmax(s123[:, 2]) - np.nanmin(s123[:, 2]))

    # ------ Welch PSD features (9) ------------------------------------------
    if valid.sum() >= 4 and not np.any(np.isnan(s_ref)):
        x = np.linalg.norm(unit_g[valid] - s_ref, axis=1)
        nperseg = min(int(fs), len(x))
        nperseg = max(nperseg, 4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            freqs_w, psd_w = welch(x, fs=fs, nperseg=nperseg)
        pk = int(np.argmax(psd_w))
        peak_freq = float(freqs_w[pk])
        peak_power = float(psd_w[pk])
        bp_low = _band_power(freqs_w, psd_w, *BAND_LOW)
        bp_mid = _band_power(freqs_w, psd_w, *BAND_MID)
        bp_high = _band_power(freqs_w, psd_w, *BAND_HIGH)
        ratio_mid_low = bp_mid / (bp_low + 1e-12)
        ratio_high_mid = bp_high / (bp_mid + 1e-12)
        spec_entropy = _spectral_entropy(psd_w)
        # VB SNR: power near 80 Hz vs broadband noise floor
        vb_mask = (freqs_w >= VB_FREQ - 5) & (freqs_w <= VB_FREQ + 5)
        noise_mask = (freqs_w >= 10) & (freqs_w <= 900)
        vb_power = psd_w[vb_mask].mean() if vb_mask.sum() > 0 else 0.0
        noise_floor = psd_w[noise_mask].mean() if noise_mask.sum() > 0 else 1e-12
        vb_snr_80hz = float(10 * np.log10(vb_power / (noise_floor + 1e-12) + 1e-12))
    else:
        peak_freq = peak_power = 0.0
        bp_low = bp_mid = bp_high = 0.0
        ratio_mid_low = ratio_high_mid = spec_entropy = vb_snr_80hz = 0.0

    # ------ Raw FFT bins (1,001) --------------------------------------------
    # Compute FFT of the scalar DOP-gated displacement signal.
    # Zero-pad to FFT_PAD=2000 → frequency resolution = fs/FFT_PAD = 1 Hz.
    if valid.sum() >= 4 and not np.any(np.isnan(s_ref)):
        x_fft = np.linalg.norm(unit_g[valid] - s_ref, axis=1)
        x_pad = np.zeros(FFT_PAD)
        x_pad[: min(len(x_fft), FFT_PAD)] = x_fft[: min(len(x_fft), FFT_PAD)]
        spectrum = np.abs(np.fft.rfft(x_pad))  # length FFT_PAD//2 + 1 = 1001
        fft_bins = spectrum[: N_FFT_BINS].astype(float)
    else:
        fft_bins = np.zeros(N_FFT_BINS)

    # ------ Assemble row ---------------------------------------------------
    row: dict = {
        # DOP (4)
        "dop_mean": dop_mean,
        "dop_std": dop_std,
        "dop_iqr": dop_iqr,
        "frac_dop_low": frac_dop_low,
        # Step-angle (9)
        "step_mean": step_mean,
        "step_std": step_std,
        "step_max": step_max,
        "step_p99": step_p99,
        "step_kurtosis": step_kurtosis,
        "step_skewness": step_skewness,
        "burst_count": float(burst_count),
        "cum_arc": cum_arc,
        "autocorr_lag1": autocorr_lag1,
        # Theta-ref (4)
        "theta_mean": theta_mean,
        "theta_std": theta_std,
        "theta_range": theta_range,
        "theta_p95": theta_p95,
        # Stokes variance (6)
        "s1_std": s1_std,
        "s2_std": s2_std,
        "s3_std": s3_std,
        "s1_range": s1_range,
        "s2_range": s2_range,
        "s3_range": s3_range,
        # Welch PSD (9): peak_freq, peak_power, bp_low, bp_mid, bp_high,
        #                ratio_mid_low, ratio_high_mid, spec_entropy, vb_snr_80hz
        "peak_freq": peak_freq,
        "peak_power": peak_power,
        "bp_low": bp_low,
        "bp_mid": bp_mid,
        "bp_high": bp_high,
        "ratio_mid_low": ratio_mid_low,
        "ratio_high_mid": ratio_high_mid,
        "spec_entropy": spec_entropy,
        "vb_snr_80hz": vb_snr_80hz,
    }

    # Replace non-finite scalars with 0
    row = {k: (0.0 if (not math.isfinite(v)) else v) for k, v in row.items()}

    # FFT bins
    fft_bins = np.where(np.isfinite(fft_bins), fft_bins, 0.0)
    for i, v in enumerate(fft_bins):
        row[f"fft_bin_{i}"] = float(v)

    return row


SCALAR_FEATURE_NAMES = [
    "dop_mean", "dop_std", "dop_iqr", "frac_dop_low",
    "step_mean", "step_std", "step_max", "step_p99",
    "step_kurtosis", "step_skewness", "burst_count", "cum_arc", "autocorr_lag1",
    "theta_mean", "theta_std", "theta_range", "theta_p95",
    "s1_std", "s2_std", "s3_std", "s1_range", "s2_range", "s3_range",
    "peak_freq", "peak_power", "bp_low", "bp_mid", "bp_high",
    "ratio_mid_low", "ratio_high_mid", "spec_entropy", "vb_snr_80hz",
]
ALL_FEATURE_NAMES = SCALAR_FEATURE_NAMES + [f"fft_bin_{i}" for i in range(N_FFT_BINS)]


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_files(data_dir: Path, source_tokens: list[str]) -> list[dict]:
    """Return file-info dicts for all CSV files matching the given source tokens."""
    pattern = "pm1000_sop_2kHz_1min_*_*_1550_*_*.csv"
    event_tags = {"NE", "FS", "VB", "MB", "TAP"}
    records = []

    for csv_path in sorted(data_dir.glob(pattern)):
        if csv_path.stat().st_size < MIN_FILE_SIZE_BYTES:
            continue

        stem = csv_path.stem
        parts = stem.split("_")
        # prefix: pm1000, sop, 2kHz, 1min → 4 tokens
        middle = parts[4:-3]   # strip prefix and suffix (1550, date, rep)
        suffix = parts[-3:]
        if len(suffix) != 3 or suffix[0] != "1550":
            continue

        event_idx = None
        for i, tok in enumerate(middle):
            if tok in event_tags:
                event_idx = i
                break
        if event_idx is None:
            continue

        source_name = "_".join(middle[:event_idx])
        event = middle[event_idx]

        if source_name not in source_tokens:
            continue

        records.append({
            "path": csv_path,
            "source": source_name,
            "event": event,
            "date": suffix[1],
            "rep": suffix[2],
        })

    return records


# ---------------------------------------------------------------------------
# Data loading and windowing
# ---------------------------------------------------------------------------

def load_file(file_info: dict, logger: logging.Logger) -> pd.DataFrame | None:
    """Load CSV, unit-normalise S1/S2/S3, validate quality."""
    path = file_info["path"]
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.warning("Could not read %s: %s", path.name, exc)
        return None

    required = {"Time_s", "S0", "S1", "S2", "S3"}
    if not required.issubset(df.columns):
        logger.warning("Missing columns in %s", path.name)
        return None

    df = df[["Time_s", "S0", "S1", "S2", "S3"]].copy()
    df.sort_values("Time_s", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Detect NaN values
    nan_count = df[["S1", "S2", "S3"]].isna().sum().sum()
    if nan_count > 0:
        logger.warning("%s: %d NaN values in S1/S2/S3", path.name, nan_count)

    # Defensive unit normalisation
    s123 = df[["S1", "S2", "S3"]].values.astype(float)
    norms = np.linalg.norm(s123, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    s123_unit = s123 / norms
    df["S1"] = s123_unit[:, 0]
    df["S2"] = s123_unit[:, 1]
    df["S3"] = s123_unit[:, 2]

    # Verify unit sphere (log if deviation > 1e-4)
    unit_check = np.linalg.norm(df[["S1", "S2", "S3"]].values, axis=1)
    deviation = np.abs(unit_check - 1.0).max()
    if deviation > 1e-4:
        logger.warning("%s: max unit-sphere deviation = %.2e", path.name, deviation)

    return df


def detect_fs(df: pd.DataFrame) -> float:
    """Detect sampling rate from timestamp deltas (Hz)."""
    dt = np.diff(df["Time_s"].values)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return float(EXPECTED_FS)
    return float(1.0 / np.median(dt))


def compute_file_reference(df: pd.DataFrame) -> np.ndarray:
    """Compute mean reference SOP unit vector from entire file (DOP-gated)."""
    s123 = df[["S1", "S2", "S3"]].values.astype(float)
    unit, dop = _unit_normalise(s123)
    gate = dop > DOP_GATE
    unit[~gate] = np.nan
    valid = ~np.any(np.isnan(unit), axis=1)
    if valid.sum() > 0:
        raw_mean = np.nanmean(unit[valid], axis=0)
        n = np.linalg.norm(raw_mean)
        return raw_mean / n if n > 0 else np.zeros(3)
    return np.zeros(3)


def extract_windows_from_file(
    file_info: dict,
    df: pd.DataFrame,
    fs: float,
    s_ref: np.ndarray,
    logger: logging.Logger,
) -> list[dict]:
    """Segment file into 1-second non-overlapping windows and extract features."""
    t = df["Time_s"].values.astype(float)
    t_start = t[0]
    t_end = t[-1]

    rows = []
    win_start = t_start
    while win_start + WINDOW_S <= t_end + 1e-9:
        win_end = win_start + WINDOW_S
        mask = (t >= win_start) & (t < win_end)
        n_pts = mask.sum()
        if n_pts >= MIN_SAMPLES_PER_WIN:
            win_df = df.loc[mask].reset_index(drop=True)
            feats = extract_features(win_df, fs, s_ref)
            row = {
                "source": file_info["source"],
                "event": file_info["event"],
                "file": file_info["path"].name,
                "t_start": round(win_start, 4),
                "split": "",      # filled in later
            }
            row.update(feats)
            rows.append(row)
        win_start += WINDOW_S

    return rows


# ---------------------------------------------------------------------------
# Dataset building with time-based split
# ---------------------------------------------------------------------------

def build_dataset(
    files: list[dict],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build train/test DataFrames from all files for one source.

    Time-based split per file: first 20% of rows → train, last 80% → test.
    Returns (train_df, test_df) each with all feature columns and 'event' column.
    """
    train_rows: list[dict] = []
    test_rows: list[dict] = []

    for fi in files:
        logger.info("  Loading %s", fi["path"].name)
        df = load_file(fi, logger)
        if df is None:
            continue

        fs = detect_fs(df)
        logger.debug("    Detected fs = %.1f Hz  rows = %d", fs, len(df))
        if abs(fs - EXPECTED_FS) > 500:
            logger.warning("    Sampling rate %.1f Hz differs from expected %d Hz in %s",
                           fs, EXPECTED_FS, fi["path"].name)

        s_ref = compute_file_reference(df)

        # Time-based split
        n_total = len(df)
        n_train = math.ceil(TRAIN_FRACTION * n_total)
        df_train_raw = df.iloc[:n_train].reset_index(drop=True)
        df_test_raw = df.iloc[n_train:].reset_index(drop=True)

        for subset, df_sub, split_label in [
            (train_rows, df_train_raw, "train"),
            (test_rows, df_test_raw, "test"),
        ]:
            windows = extract_windows_from_file(fi, df_sub, fs, s_ref, logger)
            for w in windows:
                w["split"] = split_label
            subset.extend(windows)

    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)
    return train_df, test_df


# ---------------------------------------------------------------------------
# Model training helpers
# ---------------------------------------------------------------------------

def _label_encode(events: pd.Series) -> tuple[np.ndarray, list[str]]:
    """Encode event string labels to integer indices, returning (y, classes)."""
    present = sorted(events.unique())
    # Use EVENT_ORDER where possible to maintain consistent ordering
    order = [e for e in EVENT_ORDER if e in present]
    extra = [e for e in present if e not in order]
    classes = order + extra
    mapping = {c: i for i, c in enumerate(classes)}
    return events.map(mapping).values.astype(int), classes


def build_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    feat_cols = ALL_FEATURE_NAMES
    X = df[feat_cols].values.astype(float)
    y = df["event"].values
    return X, y


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
    logger: logging.Logger,
) -> tuple[dict, SimpleImputer, StandardScaler]:
    """Impute, scale, train RF / XGB / SVM, return (models, imputer, scaler)."""
    logger.info("Pre-processing: impute + scale …")
    imputer = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_imp)
    X_test_sc = scaler.transform(X_test_imp)

    n_classes = len(classes)
    models = {}

    # ---- Random Forest -------------------------------------------------
    logger.info("Training Random Forest …")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    rf.fit(X_train_sc, y_train)
    logger.info("  RF trained in %.1f s", time.time() - t0)
    models["Random Forest"] = {
        "clf": rf,
        "X_train": X_train_sc,
        "X_test": X_test_sc,
    }

    # ---- XGBoost -------------------------------------------------------
    logger.info("Training XGBoost …")
    t0 = time.time()
    # Encode labels as integers for XGBoost
    le_map = {c: i for i, c in enumerate(classes)}
    y_train_enc = np.array([le_map[c] for c in y_train])
    y_test_enc = np.array([le_map[c] for c in y_test])

    n_samples_per_class = np.bincount(y_train_enc, minlength=n_classes)
    n_majority = n_samples_per_class.max()
    spw = (n_majority / np.maximum(n_samples_per_class, 1)).mean()

    xgb = XGBClassifier(
        n_estimators=200,
        early_stopping_rounds=20,
        scale_pos_weight=spw,
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
        verbosity=0,
    )
    xgb.fit(
        X_train_sc, y_train_enc,
        eval_set=[(X_test_sc, y_test_enc)],
        verbose=False,
    )
    logger.info("  XGB trained in %.1f s", time.time() - t0)
    models["XGBoost"] = {
        "clf": xgb,
        "X_train": X_train_sc,
        "X_test": X_test_sc,
        "le_map": le_map,
        "y_train_enc": y_train_enc,
        "y_test_enc": y_test_enc,
    }

    # ---- SVM -----------------------------------------------------------
    logger.info("Training SVM …")
    t0 = time.time()
    svm = SVC(
        kernel="rbf",
        C=10,
        class_weight="balanced",
        probability=True,
        random_state=RANDOM_SEED,
    )
    svm.fit(X_train_sc, y_train)
    logger.info("  SVM trained in %.1f s", time.time() - t0)
    models["SVM"] = {
        "clf": svm,
        "X_train": X_train_sc,
        "X_test": X_test_sc,
    }

    return models, imputer, scaler


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_model(
    name: str,
    info: dict,
    y_test: np.ndarray,
    classes: list[str],
    logger: logging.Logger,
) -> dict:
    """Compute predictions, probabilities and metrics for one model."""
    clf = info["clf"]
    X_test = info["X_test"]

    if name == "XGBoost":
        y_pred_enc = clf.predict(X_test)
        le_map = info["le_map"]
        inv_map = {v: k for k, v in le_map.items()}
        y_pred = np.array([inv_map[i] for i in y_pred_enc])
        y_proba = clf.predict_proba(X_test)
    else:
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_mic = f1_score(y_test, y_pred, average="micro", zero_division=0)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)

    # ROC-AUC
    y_bin = label_binarize(y_test, classes=classes)
    n_cls = len(classes)
    if n_cls == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])
    try:
        auc_macro = roc_auc_score(y_bin, y_proba, average="macro",
                                  multi_class="ovr")
        auc_micro = roc_auc_score(y_bin, y_proba, average="micro",
                                  multi_class="ovr")
    except Exception:
        auc_macro = auc_micro = float("nan")

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    report = classification_report(y_test, y_pred, labels=classes,
                                   target_names=classes, zero_division=0,
                                   output_dict=True)

    logger.info(
        "  [%s] acc=%.4f  F1_macro=%.4f  AUC_macro=%.4f",
        name, acc, f1_mac, auc_macro,
    )

    return {
        "name": name,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": acc,
        "f1_macro": f1_mac,
        "f1_micro": f1_mic,
        "precision_macro": prec,
        "recall_macro": rec,
        "auc_macro": auc_macro,
        "auc_micro": auc_micro,
        "cm": cm,
        "report": report,
    }


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_models(
    models: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    classes: list[str],
    logger: logging.Logger,
) -> dict:
    """Run 5-fold stratified CV on train set; return {model_name: cv_scores}."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_results = {}
    for name, info in models.items():
        X = info["X_train"]
        if name == "XGBoost":
            # Clone XGBoost without early stopping for CV (no eval_set available)
            clf_cv = XGBClassifier(
                n_estimators=200,
                eval_metric="mlogloss",
                random_state=RANDOM_SEED,
                verbosity=0,
            )
            y = info["y_train_enc"]
        else:
            clf_cv = info["clf"]
            y = y_train
        logger.info("  CV: %s …", name)
        try:
            scores = cross_val_score(
                clf_cv, X, y, cv=cv, scoring="f1_macro", n_jobs=1
            )
        except Exception as exc:
            logger.warning("  CV failed for %s: %s", name, exc)
            scores = np.array([float("nan")] * 5)
        cv_results[name] = scores
        logger.info("    F1 = %.4f ± %.4f", scores.mean(), scores.std())
    return cv_results


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

PLOT_DPI = 150


def _save_fig(fig: plt.Figure, path: Path, logger: logging.Logger) -> None:
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved %s", path.name)


def plot_confusion_matrix(
    result: dict,
    classes: list[str],
    source: str,
    out_dir: Path,
    logger: logging.Logger,
) -> None:
    """Plot normalised confusion matrix for one model."""
    cm = result["cm"].astype(float)
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)

    name = result["name"]
    acc = result["accuracy"]
    safe_name = name.lower().replace(" ", "_")
    idx = {"Random Forest": "01", "XGBoost": "02", "SVM": "03"}.get(name, "0X")

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm_norm, vmin=0, vmax=1,
                   cmap=plt.cm.Reds, aspect="auto")  # type: ignore[attr-defined]
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title(f"{name} Confusion Matrix — {source} (Acc {acc:.2%})", fontsize=13)

    for i in range(len(classes)):
        for j in range(len(classes)):
            pct = cm_norm[i, j]
            cnt = int(cm[i, j])
            color = "white" if pct > 0.55 else "black"
            ax.text(j, i, f"{pct:.0%}\n({cnt})",
                    ha="center", va="center", fontsize=9, color=color)

    fig.tight_layout()
    _save_fig(fig, out_dir / f"{idx}_cm_{safe_name}.png", logger)


def plot_roc_curves(
    results: list[dict],
    classes: list[str],
    y_test: np.ndarray,
    source: str,
    out_dir: Path,
    logger: logging.Logger,
) -> None:
    """Multi-class ROC-AUC curves for all models."""
    y_bin = label_binarize(y_test, classes=classes)
    n_cls = len(classes)
    if n_cls == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])

    fig, ax = plt.subplots(figsize=(10, 8))
    linestyles = ["-", "--", ":"]
    colors_model = ["steelblue", "darkorange", "green"]
    light_colors = ["lightblue", "moccasin", "lightgreen"]

    for m_idx, res in enumerate(results):
        y_prob = res["y_proba"]
        ls = linestyles[m_idx % len(linestyles)]
        mc = colors_model[m_idx]
        mname = res["name"]

        # Macro average
        all_fpr = np.unique(np.concatenate([
            roc_curve(y_bin[:, i], y_prob[:, i])[0]
            for i in range(n_cls)
            if len(np.unique(y_bin[:, i])) > 1
        ]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_cls):
            if len(np.unique(y_bin[:, i])) > 1:
                fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
        valid_cls = sum(1 for i in range(n_cls)
                        if len(np.unique(y_bin[:, i])) > 1)
        mean_tpr /= max(valid_cls, 1)

        auc_mac = res["auc_macro"]
        ax.plot(all_fpr, mean_tpr, ls=ls, color=mc, lw=2,
                label=f"{mname} macro AUC={auc_mac:.3f}")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC-AUC Curves — {source}", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    _save_fig(fig, out_dir / "04_roc_auc_curves.png", logger)


def plot_learning_curves(
    results: list[dict],
    y_test: np.ndarray,
    source: str,
    out_dir: Path,
    logger: logging.Logger,
) -> None:
    """Plot per-window running F1 score as a proxy learning curve."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_model = ["steelblue", "darkorange", "green"]
    win_size = 10

    for m_idx, res in enumerate(results):
        y_pred = res["y_pred"]
        # Compute rolling window F1
        n = len(y_pred)
        running_f1 = []
        indices = []
        for end in range(win_size, n + 1):
            start = max(0, end - win_size)
            f1_v = f1_score(y_test[start:end], y_pred[start:end],
                            average="macro", zero_division=0)
            running_f1.append(f1_v)
            indices.append(end)

        if len(running_f1) > 0:
            ax.plot(indices, running_f1, color=colors_model[m_idx],
                    lw=1.5, label=res["name"], alpha=0.85)

    ax.set_xlabel("Test window index", fontsize=12)
    ax.set_ylabel("Rolling F1 score (macro)", fontsize=12)
    ax.set_title(f"Learning Curves (Rolling F1, window=10) — {source}", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3)
    _save_fig(fig, out_dir / "05_learning_curves.png", logger)


def plot_kmeans_clustering(
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
    source: str,
    out_dir: Path,
    logger: logging.Logger,
) -> None:
    """K-Means clustering visualised via 2D PCA projection."""
    from sklearn.metrics import silhouette_score

    n_clusters = min(5, len(classes))
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_2d = pca.fit_transform(X_test)

    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    km_labels = km.fit_predict(X_test)

    try:
        sil = silhouette_score(X_test, km_labels, sample_size=min(5000, len(X_test)),
                               random_state=RANDOM_SEED)
    except Exception:
        sil = float("nan")

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap_class = plt.cm.tab10  # type: ignore[attr-defined]
    markers = ["o", "s", "^", "D", "v"]

    for ci, cls in enumerate(classes):
        mask = y_test == cls
        if mask.sum() == 0:
            continue
        marker = markers[km_labels[mask][0] % len(markers)]
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=[cmap_class(ci / max(len(classes) - 1, 1))] * mask.sum(),
            marker=marker,
            label=cls, alpha=0.5, s=20,
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)", fontsize=11)
    ax.set_title(
        f"K-Means Clustering (k={n_clusters}) — {source}\n"
        f"Silhouette = {sil:.3f}",
        fontsize=12,
    )
    ax.legend(title="True class", fontsize=9)
    ax.grid(alpha=0.3)
    _save_fig(fig, out_dir / "06_kmeans_clustering.png", logger)


def plot_class_distribution(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    classes: list[str],
    source: str,
    out_dir: Path,
    logger: logging.Logger,
) -> None:
    """Stacked bar chart of class distribution in train vs test."""
    present_classes = [c for c in classes if c in train_df["event"].values
                       or c in test_df["event"].values]

    train_counts = {c: (train_df["event"] == c).sum() for c in present_classes}
    test_counts = {c: (test_df["event"] == c).sum() for c in present_classes}

    x = np.arange(len(present_classes))
    width = 0.35
    colors = plt.cm.Set2(np.linspace(0, 1, len(present_classes)))  # type: ignore[attr-defined]

    fig, ax = plt.subplots(figsize=(10, 6))
    train_vals = [train_counts[c] for c in present_classes]
    test_vals = [test_counts[c] for c in present_classes]

    bars_train = ax.bar(x - width / 2, train_vals, width, label="Train",
                        color="steelblue", alpha=0.8)
    bars_test = ax.bar(x + width / 2, test_vals, width, label="Test",
                       color="darkorange", alpha=0.8)

    total_train = max(sum(train_vals), 1)
    total_test = max(sum(test_vals), 1)
    for bar, val in zip(bars_train, train_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{val/total_train:.1%}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars_test, test_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{val/total_test:.1%}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(present_classes, fontsize=11)
    ax.set_ylabel("Number of windows", fontsize=12)
    ax.set_title(f"Class Distribution — {source}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    _save_fig(fig, out_dir / "07_class_distribution.png", logger)


def plot_model_comparison(
    results: list[dict],
    cv_results: dict,
    source: str,
    out_dir: Path,
    logger: logging.Logger,
) -> None:
    """Grouped bar chart comparing accuracy, precision, recall, F1 across models."""
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
    model_names = [r["name"] for r in results]

    x = np.arange(len(metrics))
    width = 0.25
    colors_model = ["steelblue", "darkorange", "green"]

    fig, ax = plt.subplots(figsize=(11, 6))
    for m_idx, res in enumerate(results):
        vals = [res[m] for m in metrics]
        # Error bar from CV std (only for F1)
        err = [0, 0, 0,
               cv_results.get(res["name"], np.zeros(5)).std()]
        offset = (m_idx - 1) * width
        bars = ax.bar(x + offset, vals, width, label=res["name"],
                      color=colors_model[m_idx], alpha=0.85, yerr=err,
                      capsize=4)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim([0, 1.12])
    ax.set_title(f"Model Comparison — {source}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    _save_fig(fig, out_dir / "08_model_comparison.png", logger)


def plot_feature_distributions(
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
    rf_importances: np.ndarray,
    source: str,
    out_dir: Path,
    logger: logging.Logger,
) -> None:
    """Box plots of the top 6 scalar features by RF importance."""
    scalar_idx = list(range(len(SCALAR_FEATURE_NAMES)))
    scalar_importances = rf_importances[scalar_idx]
    top6_local = np.argsort(scalar_importances)[::-1][:6]
    top6_names = [SCALAR_FEATURE_NAMES[i] for i in top6_local]
    top6_data = X_test[:, top6_local]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = axes.flatten()
    cmap = plt.cm.Set2  # type: ignore[attr-defined]
    colors = [cmap(i / max(len(classes) - 1, 1)) for i in range(len(classes))]

    for fi, (feat_name, ax) in enumerate(zip(top6_names, axes_flat)):
        data_by_class = [top6_data[y_test == cls, fi] for cls in classes]
        bp = ax.boxplot(data_by_class, patch_artist=True, notch=False,
                        showfliers=True, flierprops={"marker": ".", "ms": 2})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=9)
        ax.set_title(feat_name, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Top-6 Feature Distributions — {source}", fontsize=13, y=1.01)
    fig.tight_layout()
    _save_fig(fig, out_dir / "09_feature_distributions.png", logger)


# ---------------------------------------------------------------------------
# Output file helpers
# ---------------------------------------------------------------------------

def save_results_csv(
    test_df: pd.DataFrame,
    results: list[dict],
    classes: list[str],
    out_dir: Path,
    logger: logging.Logger,
) -> None:
    """Save per-window predictions + probabilities to results.csv."""
    out = pd.DataFrame()
    out["file"] = test_df["file"].values
    out["t_start"] = test_df["t_start"].values
    out["true_event"] = test_df["event"].values

    for res in results:
        name_safe = res["name"].lower().replace(" ", "_")
        out[f"pred_{name_safe}"] = res["y_pred"]
        for ci, cls in enumerate(classes):
            out[f"prob_{name_safe}_{cls}"] = res["y_proba"][:, ci]

    out.to_csv(out_dir / "results.csv", index=False)
    logger.info("  Saved results.csv (%d rows)", len(out))


def save_performance_txt(
    results: list[dict],
    cv_results: dict,
    classes: list[str],
    source: str,
    out_dir: Path,
    logger: logging.Logger,
) -> None:
    """Write human-readable model performance summary."""
    lines = [
        f"NOVOPTEL PM1000 — ML Pipeline Performance Summary",
        f"Source : {source}",
        f"Date   : {datetime.now(timezone.utc).isoformat()}",
        "=" * 60,
    ]
    for res in results:
        name = res["name"]
        lines += [
            f"\n{'─' * 40}",
            f"Model: {name}",
            f"  Accuracy        : {res['accuracy']:.4f}",
            f"  F1 macro        : {res['f1_macro']:.4f}",
            f"  F1 micro        : {res['f1_micro']:.4f}",
            f"  Precision macro : {res['precision_macro']:.4f}",
            f"  Recall macro    : {res['recall_macro']:.4f}",
            f"  AUC macro       : {res['auc_macro']:.4f}",
            f"  AUC micro       : {res['auc_micro']:.4f}",
        ]
        cv = cv_results.get(name, np.zeros(5))
        lines.append(
            f"  CV F1 (5-fold)  : {cv.mean():.4f} ± {cv.std():.4f}"
        )
        lines += ["", "  Per-class report:"]
        report = res["report"]
        for cls in classes:
            if cls in report:
                r = report[cls]
                lines.append(
                    f"    {cls:<8} P={r['precision']:.3f}  "
                    f"R={r['recall']:.3f}  F1={r['f1-score']:.3f}  "
                    f"N={r['support']}"
                )

    txt = "\n".join(lines)
    (out_dir / "model_performance.txt").write_text(txt)
    logger.info("  Saved model_performance.txt")


def save_feature_importance(
    rf_importances: np.ndarray,
    out_dir: Path,
    logger: logging.Logger,
    top_n: int = 50,
) -> None:
    """Save top-N features by RF importance."""
    order = np.argsort(rf_importances)[::-1][:top_n]
    df_fi = pd.DataFrame({
        "rank": range(1, len(order) + 1),
        "feature": [ALL_FEATURE_NAMES[i] for i in order],
        "importance": rf_importances[order],
    })
    df_fi.to_csv(out_dir / "feature_importance.csv", index=False)
    logger.info("  Saved feature_importance.csv (top %d)", top_n)


def save_config_yaml(
    args: argparse.Namespace,
    source: str,
    source_tokens: list[str],
    n_train: int,
    n_test: int,
    classes: list[str],
    out_dir: Path,
    logger: logging.Logger,
) -> None:
    """Save run configuration as YAML."""
    git_commit = "unknown"
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=str(Path(__file__).parent),
        ).decode().strip()
    except Exception:
        pass

    cfg = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "random_seed": RANDOM_SEED,
        "source": source,
        "source_tokens": source_tokens,
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "window_s": WINDOW_S,
        "train_fraction": TRAIN_FRACTION,
        "n_train_windows": n_train,
        "n_test_windows": n_test,
        "n_features": len(ALL_FEATURE_NAMES),
        "n_scalar_features": len(SCALAR_FEATURE_NAMES),
        "n_fft_bins": N_FFT_BINS,
        "classes": classes,
        "models": {
            "Random Forest": {"n_estimators": 100, "max_depth": 15,
                              "class_weight": "balanced"},
            "XGBoost": {"n_estimators": 200, "early_stopping_rounds": 20},
            "SVM": {"kernel": "rbf", "C": 10, "class_weight": "balanced"},
        },
    }
    with open(out_dir / "config.yaml", "w") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)
    logger.info("  Saved config.yaml")


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Universal ML pipeline for NOVOPTEL PM1000 SOP dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data_dir", type=Path, default=Path("dataset-1603"),
        help="Path to the dataset directory containing CSV files.",
    )
    p.add_argument(
        "--source", type=str, default="10GE",
        choices=list(SOURCE_ALIASES.keys()),
        help="Source type to process.",
    )
    p.add_argument(
        "--output_dir", type=Path, default=None,
        help="Output directory (default: outputs_{SOURCE}_enhanced).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    source = args.source
    source_tokens = SOURCE_ALIASES[source]
    out_dir = args.output_dir or Path(f"outputs_{source}_enhanced")
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(out_dir / "training.log")
    logger.info("=" * 60)
    logger.info("NOVOPTEL PM1000 — Universal ML Pipeline")
    logger.info("=" * 60)
    logger.info("Source     : %s  (tokens: %s)", source, source_tokens)
    logger.info("Data dir   : %s", args.data_dir)
    logger.info("Output dir : %s", out_dir)
    logger.info("Random seed: %d", RANDOM_SEED)

    np.random.seed(RANDOM_SEED)

    # ------------------------------------------------------------------ #
    # 1. File discovery
    # ------------------------------------------------------------------ #
    data_dir = args.data_dir
    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    files = discover_files(data_dir, source_tokens)
    if not files:
        logger.error(
            "No files found for source '%s' in %s.  "
            "Available sources: %s",
            source, data_dir,
            sorted({f['source'] for f in discover_files(data_dir, list(
                t for tokens in SOURCE_ALIASES.values() for t in tokens))}),
        )
        sys.exit(1)

    detected_events = sorted({fi["event"] for fi in files})
    logger.info("Files found: %d  |  Events: %s", len(files), detected_events)

    # ------------------------------------------------------------------ #
    # 2. Build train / test DataFrames
    # ------------------------------------------------------------------ #
    logger.info("Building dataset (feature extraction) …")
    t_build0 = time.time()
    train_df, test_df = build_dataset(files, logger)
    logger.info(
        "Dataset ready in %.1f s  |  train=%d windows  test=%d windows",
        time.time() - t_build0, len(train_df), len(test_df),
    )

    if len(train_df) == 0 or len(test_df) == 0:
        logger.error("Insufficient data after windowing.  Exiting.")
        sys.exit(1)

    # Determine event classes from data
    all_events = sorted(set(train_df["event"].tolist() + test_df["event"].tolist()))
    classes = [e for e in EVENT_ORDER if e in all_events]
    classes += [e for e in all_events if e not in classes]

    logger.info("Classes: %s", classes)

    X_train_raw, y_train = build_xy(train_df)
    X_test_raw, y_test = build_xy(test_df)

    # ------------------------------------------------------------------ #
    # 3. Train models
    # ------------------------------------------------------------------ #
    logger.info("Training models …")
    models, imputer, scaler = train_models(
        X_train_raw, y_train, X_test_raw, y_test, classes, logger
    )

    # ------------------------------------------------------------------ #
    # 4. Evaluate
    # ------------------------------------------------------------------ #
    logger.info("Evaluating models …")
    X_test_scaled = scaler.transform(imputer.transform(X_test_raw))
    results = []
    for name, info in models.items():
        res = evaluate_model(name, info, y_test, classes, logger)
        results.append(res)

    # Cross-validation
    logger.info("Running 5-fold CV on training set …")
    X_train_sc = scaler.transform(imputer.transform(X_train_raw))
    cv_results = cross_validate_models(models, X_train_sc, y_train, classes, logger)

    # ------------------------------------------------------------------ #
    # 5. Generate plots
    # ------------------------------------------------------------------ #
    logger.info("Generating plots …")

    # 01, 02, 03 — Confusion matrices
    for res in results:
        plot_confusion_matrix(res, classes, source, plots_dir, logger)

    # 04 — ROC-AUC curves
    plot_roc_curves(results, classes, y_test, source, plots_dir, logger)

    # 05 — Learning curves
    plot_learning_curves(results, y_test, source, plots_dir, logger)

    # 06 — K-Means clustering
    plot_kmeans_clustering(X_test_scaled, y_test, classes,
                           source, plots_dir, logger)

    # 07 — Class distribution
    plot_class_distribution(train_df, test_df, classes,
                            source, plots_dir, logger)

    # 08 — Model comparison
    plot_model_comparison(results, cv_results, source, plots_dir, logger)

    # 09 — Feature distributions (uses RF importances)
    rf_info = models["Random Forest"]
    rf_clf = rf_info["clf"]
    rf_importances = rf_clf.feature_importances_
    plot_feature_distributions(
        X_test_scaled, y_test, classes,
        rf_importances, source, plots_dir, logger,
    )

    # ------------------------------------------------------------------ #
    # 6. Save output files
    # ------------------------------------------------------------------ #
    logger.info("Saving output files …")
    save_results_csv(test_df, results, classes, out_dir, logger)
    save_performance_txt(results, cv_results, classes, source, out_dir, logger)
    save_feature_importance(rf_importances, out_dir, logger)
    save_config_yaml(
        args, source, source_tokens,
        len(train_df), len(test_df), classes, out_dir, logger,
    )

    # ------------------------------------------------------------------ #
    # 7. Final summary
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("Output directory: %s", out_dir)
    for res in results:
        logger.info(
            "  %-15s  acc=%.4f  F1=%.4f  AUC=%.4f",
            res["name"], res["accuracy"], res["f1_macro"], res["auc_macro"],
        )
    logger.info("Plots saved to: %s", plots_dir)


if __name__ == "__main__":
    main()
