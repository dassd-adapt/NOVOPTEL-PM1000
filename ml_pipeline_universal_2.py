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

EVENT_ORDER = ["NE", "FS",  "VB", "MB", "TAP"]
N_MODELS = 3  # XGBoost, HGB, GB


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


# ===== TIME-DOMAIN S-PARAMETER DEVIATION FEATURES =====

def extract_s_parameter_features(s123: np.ndarray) -> dict:
    """
    Extract time-domain features focusing on S-parameter deviation.

    S-parameters represent polarization state:
    - S1: Horizontal/Vertical difference
    - S2: Diagonal/Anti-diagonal difference
    - S3: Right/Left circular difference

    Changes in these indicate fiber perturbations.
    """
    features = {}

    for i, s_name in enumerate(['S1', 'S2', 'S3']):
        s_signal = s123[:, i]

        # Statistical measures (deviation from baseline)
        features[f'{s_name}_mean'] = float(np.nanmean(s_signal))
        features[f'{s_name}_std'] = float(np.nanstd(s_signal))
        features[f'{s_name}_var'] = float(np.nanvar(s_signal))
        features[f'{s_name}_range'] = float(np.nanmax(s_signal) - np.nanmin(s_signal))
        features[f'{s_name}_iqr'] = float(np.nanpercentile(s_signal, 75) - np.nanpercentile(s_signal, 25))

        # Distribution shape
        features[f'{s_name}_skewness'] = float(scipy_skew(s_signal))
        features[f'{s_name}_kurtosis'] = float(scipy_kurtosis(s_signal))

        # Rate of change (velocity)
        if len(s_signal) > 1:
            s_diff = np.diff(s_signal)
            features[f'{s_name}_diff_mean'] = float(np.nanmean(np.abs(s_diff)))
            features[f'{s_name}_diff_std'] = float(np.nanstd(s_diff))
            features[f'{s_name}_diff_max'] = float(np.nanmax(np.abs(s_diff)))
        else:
            features[f'{s_name}_diff_mean'] = 0.0
            features[f'{s_name}_diff_std'] = 0.0
            features[f'{s_name}_diff_max'] = 0.0

        # Anomaly detection (Z-score based outliers)
        z_scores = np.abs((s_signal - np.nanmean(s_signal)) / (np.nanstd(s_signal) + 1e-9))
        features[f'{s_name}_outlier_fraction'] = float(np.mean(z_scores > 3.0))  # 3-sigma
        features[f'{s_name}_extreme_fraction'] = float(np.mean(z_scores > 5.0))  # 5-sigma

    return features


def extract_80hz_features(freqs_w: np.ndarray, psd_w: np.ndarray) -> dict:
    """
    Extract frequency-domain features focusing ONLY on 80 Hz vibration.

    80 Hz is characteristic of:
    - Heavy machinery (excavators at ~4800 RPM)
    - Mechanical vibration signatures
    - Fiber bending/stress
    """
    features = {}

    # ===== CORE 80 HZ BAND ANALYSIS =====

    # Feature 1-3: 80 Hz Core Band
    vb_core_mask = (freqs_w >= 75) & (freqs_w <= 85)
    vb_core_power = psd_w[vb_core_mask].mean() if vb_core_mask.sum() > 0 else 0.0
    vb_core_sum = psd_w[vb_core_mask].sum() if vb_core_mask.sum() > 0 else 0.0
    vb_core_max = psd_w[vb_core_mask].max() if vb_core_mask.sum() > 0 else 0.0

    features['vb_core_power'] = float(vb_core_power)
    features['vb_core_sum'] = float(vb_core_sum)
    features['vb_core_max'] = float(vb_core_max)

    # Feature 4: 80 Hz SNR (signal-to-noise ratio)
    noise_mask = (freqs_w >= 10) & (freqs_w <= 900)
    noise_floor = psd_w[noise_mask].mean() if noise_mask.sum() > 0 else 1e-12
    vb_snr = float(10 * np.log10(vb_core_power / (noise_floor + 1e-12) + 1e-12))
    features['vb_snr_db'] = vb_snr

    # Feature 5: 80 Hz Dominance (fraction of total power)
    total_power = np.sum(psd_w) if np.sum(psd_w) > 0 else 1e-12
    features['vb_dominance_ratio'] = float(vb_core_power / total_power)

    # Feature 6: 80 Hz Peak Proximity
    pk = int(np.argmax(psd_w))
    peak_freq = float(freqs_w[pk])
    features['peak_freq'] = peak_freq
    features['vb_peak_proximity'] = float(1.0 / (1.0 + np.abs(peak_freq - 80.0)))

    # ===== SPECTRAL CONCENTRATION (SHARPNESS) =====

    # Feature 7: 80 Hz Sharpness (how concentrated energy is around 80 Hz)
    lower_band_mask = (freqs_w >= 50) & (freqs_w < 75)
    upper_band_mask = (freqs_w > 85) & (freqs_w <= 100)
    lower_power = psd_w[lower_band_mask].mean() if lower_band_mask.sum() > 0 else 1e-12
    upper_power = psd_w[upper_band_mask].mean() if upper_band_mask.sum() > 0 else 1e-12

    features['vb_sharpness'] = float((vb_core_power - lower_power) + (vb_core_power - upper_power))

    # Feature 8: 80 Hz Bandwidth (Q-factor)
    vb_band_30hz = (freqs_w >= 65) & (freqs_w <= 95)
    vb_bandwidth = float(vb_core_power / (psd_w[vb_band_30hz].mean() + 1e-12)) if vb_band_30hz.sum() > 0 else 0.0
    features['vb_bandwidth_qfactor'] = vb_bandwidth

    # ===== HARMONIC ANALYSIS =====

    # Feature 9: 2nd Harmonic (160 Hz) - Indicates mechanical resonance
    harmonic_2x_mask = (freqs_w >= 155) & (freqs_w <= 165)
    harmonic_2x_power = psd_w[harmonic_2x_mask].mean() if harmonic_2x_mask.sum() > 0 else 0.0
    features['vb_harmonic_2x_160hz'] = float(harmonic_2x_power)

    # Feature 10: 2x/80 Ratio - Detects harmonic structure
    features['vb_harmonic_2x_ratio'] = float(harmonic_2x_power / (vb_core_power + 1e-12))

    # Feature 11: 0.5 Harmonic (40 Hz) - Indicates modulation
    harmonic_05x_mask = (freqs_w >= 35) & (freqs_w <= 45)
    harmonic_05x_power = psd_w[harmonic_05x_mask].mean() if harmonic_05x_mask.sum() > 0 else 0.0
    features['vb_harmonic_05x_40hz'] = float(harmonic_05x_power)

    # Feature 12: 0.5x/80 Ratio
    features['vb_harmonic_05x_ratio'] = float(harmonic_05x_power / (vb_core_power + 1e-12))

    # ===== OVERALL SPECTRAL CHARACTERISTICS =====

    # Feature 13: Spectral Entropy (how "noisy" is the spectrum)
    features['spectral_entropy'] = _spectral_entropy(psd_w)

    # Feature 14: 80 Hz Prominence (power above surroundings)
    surrounding_mask = (freqs_w >= 50) & (freqs_w <= 100)
    surrounding_median = np.median(psd_w[surrounding_mask]) if surrounding_mask.sum() > 0 else 1e-12
    features['vb_prominence'] = float(vb_core_power - surrounding_median)

    return features


def compute_optimal_weights(y_train: np.ndarray, classes: list[str]) -> dict:
    from collections import Counter
    counts = Counter(y_train)
    total = len(y_train)

    weights = {}
    for i, cls in enumerate(classes):
        freq = counts.get(cls, 1) / total
        base_weight = 1.0 / freq if freq > 0 else 1.0

        # Tiered boost based on difficulty on SP-AGIL
        if cls == "VB":
            base_weight *= 1.5   # strong 80 Hz signal — less help needed
        elif cls == "MB":
            base_weight *= 2.5   # subtle slow drift — hardest class
        elif cls == "TAP":
            base_weight *= 2.5   # step change — hard to catch in 1s window
        elif cls in ["FS"]:
            base_weight *= 1.2

        weights[i] = base_weight

    max_weight = max(weights.values())
    return {k: v / max_weight for k, v in weights.items()}

# ============================================================
# REDESIGNED MB FEATURES
# Physical basis: MB = slow monotonic drift of SOP on Poincaré
# sphere. Observable even in 1-second windows as:
#   1. Non-zero linear slope across the window
#   2. Low autocorrelation of step-angles (smooth rotation)
#   3. Consistent rotation direction (signed cumulative arc)
# ============================================================

def extract_mb_features(s123: np.ndarray, fs: float) -> dict:
    """
    Modal Birefringence features redesigned for 1-second windows.
    
    Key insight: at 1-second window length, 0.1-5 Hz oscillations
    are NOT observable (need >10 seconds for one cycle).
    Instead, detect MB via INTRA-WINDOW drift characteristics:
    - Linear trend (slope) of each Stokes parameter
    - Smoothness of angular rotation (low step-angle variance)
    - Directional consistency of rotation
    """
    features = {}
    n = len(s123)

    # ── 1. LINEAR TREND SLOPE per Stokes parameter ──────────────
    # MB causes monotonic drift → non-zero slope
    # NE is stable → slope ≈ 0
    # TAP has a step → slope is high but non-monotonic
    t_norm = np.linspace(0, 1, n)  # normalised time 0→1
    for i, name in enumerate(['S1', 'S2', 'S3']):
        sig = s123[:, i]
        if n >= 3:
            slope, intercept = np.polyfit(t_norm, sig, 1)
            # Residual after removing linear trend
            residual = sig - (slope * t_norm + intercept)
            features[f'{name}_trend_slope'] = float(slope)
            features[f'{name}_trend_r2'] = float(
                1.0 - np.var(residual) / (np.var(sig) + 1e-12)
            )  # R² — how linear is the drift? MB=high, NE=low
        else:
            features[f'{name}_trend_slope'] = 0.0
            features[f'{name}_trend_r2'] = 0.0

    # ── 2. ROTATION SMOOTHNESS on Poincaré sphere ────────────────
    # MB = smooth continuous rotation → low step-angle std, high autocorr
    # NE = near-stationary → very small angles, no direction
    # FS = random jumps → high step-angle std
    unit, dop = _unit_normalise(s123)
    gate = dop > DOP_GATE
    unit[~gate] = np.nan
    valid_unit = unit[~np.any(np.isnan(unit), axis=1)]

    if len(valid_unit) >= 4:
        angles = _geodesic_angle(valid_unit[:-1], valid_unit[1:])
        features['mb_step_mean']    = float(np.nanmean(angles))
        features['mb_step_std']     = float(np.nanstd(angles))
        features['mb_step_cv']      = float(           # coefficient of variation
            np.nanstd(angles) / (np.nanmean(angles) + 1e-9)
        )  # MB=low CV (uniform steps), FS=high CV (random)
        features['mb_angle_autocorr'] = _autocorr_lag1(angles)
        features['mb_cum_arc']      = float(np.nansum(angles))

        # Direction consistency: project consecutive steps onto
        # mean rotation axis — MB rotates consistently, NE wanders
        if len(valid_unit) >= 3:
            # Cross products give rotation axis direction
            crosses = np.cross(valid_unit[:-1], valid_unit[1:])
            norms = np.linalg.norm(crosses, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            crosses_unit = crosses / norms
            # Mean direction vector
            mean_dir = np.nanmean(crosses_unit, axis=0)
            mean_dir_norm = np.linalg.norm(mean_dir)
            # Directional consistency: 1.0=always same direction, 0=random
            features['mb_rotation_consistency'] = float(mean_dir_norm)
        else:
            features['mb_rotation_consistency'] = 0.0
    else:
        features['mb_step_mean']           = 0.0
        features['mb_step_std']            = 0.0
        features['mb_step_cv']             = 0.0
        features['mb_angle_autocorr']      = 0.0
        features['mb_cum_arc']             = 0.0
        features['mb_rotation_consistency'] = 0.0

    # ── 3. STOKES TRAJECTORY LINEARITY ──────────────────────────
    # MB traces a great circle arc → trajectory is low-dimensional
    # Compute PCA on the 3D Stokes trajectory:
    # MB: first PC explains ~100% variance (linear arc)
    # NE: near-zero variance in all PCs
    # FS/VB: variance spread across multiple PCs
    if len(valid_unit) >= 4:
        try:
            from sklearn.decomposition import PCA as _PCA
            pca = _PCA(n_components=min(3, len(valid_unit)))
            pca.fit(valid_unit)
            evr = pca.explained_variance_ratio_
            features['mb_traj_pc1_ratio'] = float(evr[0])  # MB≈1.0
            features['mb_traj_pc2_ratio'] = float(evr[1]) if len(evr) > 1 else 0.0
            # Linearity index: high if trajectory is nearly 1D
            features['mb_traj_linearity'] = float(
                evr[0] - (evr[1] if len(evr) > 1 else 0.0)
            )
        except Exception:
            features['mb_traj_pc1_ratio']  = 0.0
            features['mb_traj_pc2_ratio']  = 0.0
            features['mb_traj_linearity']  = 0.0
    else:
        features['mb_traj_pc1_ratio']  = 0.0
        features['mb_traj_pc2_ratio']  = 0.0
        features['mb_traj_linearity']  = 0.0

    return features


# ============================================================
# REDESIGNED TAP FEATURES
# Physical basis: TAP = fiber tap coupler causes a PERMANENT
# step change in polarization state at one specific moment.
# 
# The key problem: 80% of TAP windows are POST-TAP STEADY STATE
# and look identical to NE. Features must compare ACROSS windows
# using a rolling reference, not within a single window.
#
# Solution: Pass the previous window's mean SOP as context.
# If not available, use intra-window step detection.
# ============================================================

def extract_tap_features(
    s123: np.ndarray,
    unit_g: np.ndarray,
    dop: np.ndarray,
    prev_window_mean: np.ndarray | None = None,  # NEW: cross-window context
) -> dict:
    """
    Fiber TAP features redesigned with cross-window context.
    
    prev_window_mean: unit vector of mean SOP from previous window.
    When provided, enables detection of post-tap steady-state windows
    by measuring displacement from previous state.
    """
    features = {}
    n = len(s123)

    # ── 1. CROSS-WINDOW DISPLACEMENT (primary TAP detector) ──────
    # TAP causes permanent SOP shift → current window mean is far
    # from previous window mean, even in post-tap steady state
    if prev_window_mean is not None and not np.any(np.isnan(prev_window_mean)):
        valid = ~np.any(np.isnan(unit_g), axis=1)
        if valid.sum() > 0:
            curr_mean_raw = np.nanmean(unit_g[valid], axis=0)
            curr_norm = np.linalg.norm(curr_mean_raw)
            curr_mean = curr_mean_raw / curr_norm if curr_norm > 0 else curr_mean_raw

            # Angular displacement from previous window
            dot = np.clip(np.dot(curr_mean, prev_window_mean), -1.0, 1.0)
            cross_window_angle = float(np.degrees(np.arccos(dot)))
            features['tap_cross_window_angle'] = cross_window_angle

            # Euclidean distance in Stokes space
            features['tap_cross_window_dist'] = float(
                np.linalg.norm(curr_mean - prev_window_mean)
            )
        else:
            features['tap_cross_window_angle'] = 0.0
            features['tap_cross_window_dist']  = 0.0
    else:
        features['tap_cross_window_angle'] = 0.0
        features['tap_cross_window_dist']  = 0.0

    # ── 2. INTRA-WINDOW STEP DETECTION ───────────────────────────
    # Split window into halves and compare — detects the tap moment
    # if it falls within this window
    if n < 10:
        features.update({
            'tap_half_angle':          0.0,
            'tap_half_dist':           0.0,
            'tap_step_sharpness':      0.0,
            'tap_max_jump_angle':      0.0,
            'tap_max_jump_position':   0.5,
            'tap_before_stability':    0.0,
            'tap_after_stability':     0.0,
            'tap_stability_ratio':     1.0,
            'tap_dop_drop':            0.0,
        })
        return features

    half = n // 2

    # Mean SOP of first vs second half
    valid_first  = ~np.any(np.isnan(unit_g[:half]),  axis=1)
    valid_second = ~np.any(np.isnan(unit_g[half:]),  axis=1)

    if valid_first.sum() > 0 and valid_second.sum() > 0:
        mean_first_raw  = np.nanmean(unit_g[:half][valid_first],  axis=0)
        mean_second_raw = np.nanmean(unit_g[half:][valid_second], axis=0)

        n1 = np.linalg.norm(mean_first_raw)
        n2 = np.linalg.norm(mean_second_raw)
        mean_first  = mean_first_raw  / n1 if n1 > 0 else mean_first_raw
        mean_second = mean_second_raw / n2 if n2 > 0 else mean_second_raw

        dot = np.clip(np.dot(mean_first, mean_second), -1.0, 1.0)
        features['tap_half_angle'] = float(np.degrees(np.arccos(dot)))
        features['tap_half_dist']  = float(np.linalg.norm(mean_first - mean_second))
    else:
        features['tap_half_angle'] = 0.0
        features['tap_half_dist']  = 0.0

    # Step sharpness: maximum single-step angle / mean step angle
    # TAP has one large jump; NE/MB have uniform small steps
    valid_all = ~np.any(np.isnan(unit_g), axis=1)
    valid_idx = np.where(valid_all)[0]

    if len(valid_idx) >= 3:
        angles = _geodesic_angle(unit_g[valid_idx[:-1]], unit_g[valid_idx[1:]])
        max_angle  = float(np.nanmax(angles))
        mean_angle = float(np.nanmean(angles))
        features['tap_step_sharpness']    = float(max_angle / (mean_angle + 1e-9))
        features['tap_max_jump_angle']    = max_angle
        features['tap_max_jump_position'] = float(
            np.argmax(angles) / max(len(angles) - 1, 1)
        )

        # Stability before and after the biggest jump
        jump_idx = int(np.argmax(angles))
        before_angles = angles[:jump_idx]   if jump_idx > 0             else np.array([0.0])
        after_angles  = angles[jump_idx+1:] if jump_idx < len(angles)-1 else np.array([0.0])

        features['tap_before_stability'] = float(np.nanstd(before_angles))
        features['tap_after_stability']  = float(np.nanstd(after_angles))
        features['tap_stability_ratio']  = float(
            np.nanstd(before_angles) / (np.nanstd(after_angles) + 1e-9)
        )
    else:
        features['tap_step_sharpness']    = 0.0
        features['tap_max_jump_angle']    = 0.0
        features['tap_max_jump_position'] = 0.5
        features['tap_before_stability']  = 0.0
        features['tap_after_stability']   = 0.0
        features['tap_stability_ratio']   = 1.0

    # DOP drop (coupling loss at tap point)
    split = max(2, n // 5)
    dop_before = float(np.nanmean(dop[:split]))
    dop_after  = float(np.nanmean(dop[-split:]))
    features['tap_dop_drop'] = float(dop_before - dop_after)

    return features
# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_features(
    win_df: pd.DataFrame,
    fs: float,
    s_ref: np.ndarray,
    feature_mode: str = "domain_specific",
) -> dict:
    s123 = win_df[["S1", "S2", "S3"]].values.astype(float)

    unit, dop = _unit_normalise(s123)
    gate = dop > DOP_GATE
    unit_g = unit.copy()
    unit_g[~gate] = np.nan

    valid = ~np.any(np.isnan(unit_g), axis=1)

    row = {}

    if feature_mode == "domain_specific":
        # ===== TIME-DOMAIN: S-PARAMETER DEVIATION =====
        s_param_features = extract_s_parameter_features(s123)
        row.update(s_param_features)

        # Additional DOP features
        row['dop_mean'] = float(np.nanmean(dop))
        row['dop_std'] = float(np.nanstd(dop))
        row['dop_iqr'] = float(np.nanpercentile(dop, 75) - np.nanpercentile(dop, 25))

        # ===== FREQUENCY-DOMAIN: 80 HZ VIBRATION =====
        if valid.sum() >= 4 and not np.any(np.isnan(s_ref)):
            x = np.linalg.norm(unit_g[valid] - s_ref, axis=1)
            nperseg = min(int(fs), len(x))
            nperseg = max(nperseg, 4)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                freqs_w, psd_w = welch(x, fs=fs, nperseg=nperseg)

            vb_features = extract_80hz_features(freqs_w, psd_w)
            row.update(vb_features)

            # FFT bins around 80 Hz (exactly 10)
            hz_per_bin = freqs_w[1] - freqs_w[0] if len(freqs_w) > 1 else 1.0
            bin_center_80 = np.argmin(np.abs(freqs_w - 80.0))
            bin_range = max(5, int(5.0 / hz_per_bin))

            bin_start = max(0, bin_center_80 - bin_range)
            bin_end = min(len(psd_w), bin_center_80 + bin_range + 1)

            fft_values = psd_w[bin_start:bin_end]
            if len(fft_values) < 10:
                fft_values = np.pad(fft_values, (0, 10 - len(fft_values)), mode='constant')
            else:
                fft_values = fft_values[:10]

            for i, v in enumerate(fft_values):
                row[f'fft_80hz_bin_{i}'] = float(v) if np.isfinite(v) else 0.0
        else:
            vb_features_default = extract_80hz_features(np.array([80.0]), np.array([0.0]))
            row.update(vb_features_default)
            for i in range(10):
                row[f'fft_80hz_bin_{i}'] = 0.0

        # ===== NEW: MB-SPECIFIC FEATURES (12 features) =====
        # Captures: low-freq band power per S1/S2/S3 (0.1-5 Hz),
        # LF/mid ratio, linear trend slope, rotation rate + autocorr
        mb_features = extract_mb_features(s123, fs)
        row.update(mb_features)

        # ===== NEW: TAP-SPECIFIC FEATURES (5 features) =====
        # Captures: step magnitude, step position, before/after variance
        # ratio, permanence, DOP drop
        tap_features = extract_tap_features(s123, unit_g, dop)
        row.update(tap_features)

        # Replace non-finite values with 0
        row = {k: (0.0 if not math.isfinite(v) else v) for k, v in row.items()}

    else:  # "all" — original 1,035-feature mode
        # ... (unchanged)
        pass

    return row

# ===== DOMAIN-SPECIFIC FEATURE NAMES =====

DOMAIN_SPECIFIC_SCALAR_FEATURES = [
    # S1/S2/S3 deviation (36)
    "S1_mean", "S1_std", "S1_var", "S1_range", "S1_iqr",
    "S1_skewness", "S1_kurtosis",
    "S1_diff_mean", "S1_diff_std", "S1_diff_max",
    "S1_outlier_fraction", "S1_extreme_fraction",
    "S2_mean", "S2_std", "S2_var", "S2_range", "S2_iqr",
    "S2_skewness", "S2_kurtosis",
    "S2_diff_mean", "S2_diff_std", "S2_diff_max",
    "S2_outlier_fraction", "S2_extreme_fraction",
    "S3_mean", "S3_std", "S3_var", "S3_range", "S3_iqr",
    "S3_skewness", "S3_kurtosis",
    "S3_diff_mean", "S3_diff_std", "S3_diff_max",
    "S3_outlier_fraction", "S3_extreme_fraction",
    # DOP (3)
    "dop_mean", "dop_std", "dop_iqr",
    # 80 Hz VB features (15)
    "vb_core_power", "vb_core_sum", "vb_core_max",
    "vb_snr_db", "vb_dominance_ratio",
    "peak_freq", "vb_peak_proximity",
    "vb_sharpness", "vb_bandwidth_qfactor",
    "vb_harmonic_2x_160hz", "vb_harmonic_2x_ratio",
    "vb_harmonic_05x_40hz", "vb_harmonic_05x_ratio",
    "spectral_entropy", "vb_prominence",
    # MB features — redesigned (15)
    "S1_trend_slope", "S1_trend_r2",
    "S2_trend_slope", "S2_trend_r2",
    "S3_trend_slope", "S3_trend_r2",
    "mb_step_mean", "mb_step_std", "mb_step_cv",
    "mb_angle_autocorr", "mb_cum_arc", "mb_rotation_consistency",
    "mb_traj_pc1_ratio", "mb_traj_pc2_ratio", "mb_traj_linearity",
    # TAP features — redesigned (11)
    "tap_cross_window_angle", "tap_cross_window_dist",
    "tap_half_angle", "tap_half_dist",
    "tap_step_sharpness", "tap_max_jump_angle", "tap_max_jump_position",
    "tap_before_stability", "tap_after_stability",
    "tap_stability_ratio", "tap_dop_drop",
]
# Total scalar: 36 + 3 + 15 + 15 + 11 = 80
# + 10 FFT bins = 90 features total
# FFT bins around 80 Hz (10 bins)
DOMAIN_SPECIFIC_FFT_FEATURES = [f"fft_80hz_bin_{i}" for i in range(10)]

# ── NEW: named sub-lists for MB and TAP (for documentation/reference) ──
# MB features: 3×2 trend + 6 rotation + 3 trajectory = 15 features
MB_FEATURE_NAMES = [
    # Linear trend per Stokes (6)
    "S1_trend_slope", "S1_trend_r2",
    "S2_trend_slope", "S2_trend_r2",
    "S3_trend_slope", "S3_trend_r2",
    # Rotation smoothness (6)
    "mb_step_mean", "mb_step_std", "mb_step_cv",
    "mb_angle_autocorr", "mb_cum_arc", "mb_rotation_consistency",
    # Trajectory PCA linearity (3)
    "mb_traj_pc1_ratio", "mb_traj_pc2_ratio", "mb_traj_linearity",
]

# TAP features: 2 cross-window + 9 intra-window = 11 features
TAP_FEATURE_NAMES = [
    # Cross-window displacement (2) — requires prev_window_mean
    "tap_cross_window_angle", "tap_cross_window_dist",
    # Intra-window step detection (9)
    "tap_half_angle", "tap_half_dist",
    "tap_step_sharpness", "tap_max_jump_angle", "tap_max_jump_position",
    "tap_before_stability", "tap_after_stability",
    "tap_stability_ratio", "tap_dop_drop",
]

DOMAIN_SPECIFIC_SCALAR_FEATURES = [
    # S1/S2/S3 deviation (36)
    "S1_mean", "S1_std", "S1_var", "S1_range", "S1_iqr",
    "S1_skewness", "S1_kurtosis",
    "S1_diff_mean", "S1_diff_std", "S1_diff_max",
    "S1_outlier_fraction", "S1_extreme_fraction",
    "S2_mean", "S2_std", "S2_var", "S2_range", "S2_iqr",
    "S2_skewness", "S2_kurtosis",
    "S2_diff_mean", "S2_diff_std", "S2_diff_max",
    "S2_outlier_fraction", "S2_extreme_fraction",
    "S3_mean", "S3_std", "S3_var", "S3_range", "S3_iqr",
    "S3_skewness", "S3_kurtosis",
    "S3_diff_mean", "S3_diff_std", "S3_diff_max",
    "S3_outlier_fraction", "S3_extreme_fraction",
    # DOP (3)
    "dop_mean", "dop_std", "dop_iqr",
    # 80 Hz VB features (15)
    "vb_core_power", "vb_core_sum", "vb_core_max",
    "vb_snr_db", "vb_dominance_ratio",
    "peak_freq", "vb_peak_proximity",
    "vb_sharpness", "vb_bandwidth_qfactor",
    "vb_harmonic_2x_160hz", "vb_harmonic_2x_ratio",
    "vb_harmonic_05x_40hz", "vb_harmonic_05x_ratio",
    "spectral_entropy", "vb_prominence",
    # MB features — redesigned (15)
    *MB_FEATURE_NAMES,
    # TAP features — redesigned (11)
    *TAP_FEATURE_NAMES,
]
# Total scalar: 36 + 3 + 15 + 15 + 11 = 80

DOMAIN_SPECIFIC_ALL_FEATURES = (
    DOMAIN_SPECIFIC_SCALAR_FEATURES + DOMAIN_SPECIFIC_FFT_FEATURES
)
# Total: 80 scalar + 10 FFT = 90 features

# ── "all" mode feature names (unchanged) ──
SCALAR_FEATURE_NAMES = [
    "dop_mean", "dop_std", "dop_iqr", "frac_dop_low",
    # ... rest unchanged ...
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
        middle = parts[4:-3]
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

    nan_count = df[["S1", "S2", "S3"]].isna().sum().sum()
    if nan_count > 0:
        logger.warning("%s: %d NaN values in S1/S2/S3", path.name, nan_count)

    s123 = df[["S1", "S2", "S3"]].values.astype(float)
    norms = np.linalg.norm(s123, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    s123_unit = s123 / norms
    df["S1"] = s123_unit[:, 0]
    df["S2"] = s123_unit[:, 1]
    df["S3"] = s123_unit[:, 2]

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
    feature_mode: str = "domain_specific",
) -> list[dict]:
    t = df["Time_s"].values.astype(float)
    t_start = t[0]
    t_end   = t[-1]

    rows = []
    win_start = t_start
    prev_window_mean: np.ndarray | None = None  # ← track previous window SOP

    while win_start + WINDOW_S <= t_end + 1e-9:
        win_end = win_start + WINDOW_S
        mask    = (t >= win_start) & (t < win_end)
        n_pts   = mask.sum()

        if n_pts >= MIN_SAMPLES_PER_WIN:
            win_df = df.loc[mask].reset_index(drop=True)

            # Pass prev_window_mean for cross-window TAP detection
            feats = extract_features(
                win_df, fs, s_ref,
                feature_mode=feature_mode,
                prev_window_mean=prev_window_mean,   # ← NEW
            )

            # Update rolling window mean for next iteration
            s123_win = win_df[["S1", "S2", "S3"]].values.astype(float)
            unit_win, dop_win = _unit_normalise(s123_win)
            gate_win = dop_win > DOP_GATE
            unit_win[~gate_win] = np.nan
            valid_win = ~np.any(np.isnan(unit_win), axis=1)
            if valid_win.sum() > 0:
                raw_mean = np.nanmean(unit_win[valid_win], axis=0)
                raw_norm = np.linalg.norm(raw_mean)
                prev_window_mean = raw_mean / raw_norm if raw_norm > 0 else None
            else:
                prev_window_mean = None

            row = {
                "source": file_info["source"],
                "event":  file_info["event"],
                "file":   file_info["path"].name,
                "t_start": round(win_start, 4),
                "split":   "",
            }
            row.update(feats)
            rows.append(row)

        win_start += WINDOW_S

    return rows


def extract_features(
    win_df: pd.DataFrame,
    fs: float,
    s_ref: np.ndarray,
    feature_mode: str = "domain_specific",
    prev_window_mean: np.ndarray | None = None,  # ← NEW parameter
) -> dict:
    s123 = win_df[["S1", "S2", "S3"]].values.astype(float)
    unit, dop = _unit_normalise(s123)
    gate  = dop > DOP_GATE
    unit_g = unit.copy()
    unit_g[~gate] = np.nan
    valid  = ~np.any(np.isnan(unit_g), axis=1)

    row = {}

    if feature_mode == "domain_specific":
        row.update(extract_s_parameter_features(s123))
        row['dop_mean'] = float(np.nanmean(dop))
        row['dop_std']  = float(np.nanstd(dop))
        row['dop_iqr']  = float(
            np.nanpercentile(dop, 75) - np.nanpercentile(dop, 25)
        )

        if valid.sum() >= 4 and not np.any(np.isnan(s_ref)):
            x = np.linalg.norm(unit_g[valid] - s_ref, axis=1)
            nperseg = max(4, min(int(fs), len(x)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                freqs_w, psd_w = welch(x, fs=fs, nperseg=nperseg)
            row.update(extract_80hz_features(freqs_w, psd_w))

            hz_per_bin    = freqs_w[1] - freqs_w[0] if len(freqs_w) > 1 else 1.0
            bin_center_80 = np.argmin(np.abs(freqs_w - 80.0))
            bin_range     = max(5, int(5.0 / hz_per_bin))
            bin_start     = max(0, bin_center_80 - bin_range)
            bin_end       = min(len(psd_w), bin_center_80 + bin_range + 1)
            fft_values    = psd_w[bin_start:bin_end]
            if len(fft_values) < 10:
                fft_values = np.pad(
                    fft_values, (0, 10 - len(fft_values)), mode='constant'
                )
            else:
                fft_values = fft_values[:10]
            for i, v in enumerate(fft_values):
                row[f'fft_80hz_bin_{i}'] = float(v) if np.isfinite(v) else 0.0
        else:
            row.update(extract_80hz_features(np.array([80.0]), np.array([0.0])))
            for i in range(10):
                row[f'fft_80hz_bin_{i}'] = 0.0

        # MB features (redesigned: no low-freq PSD, uses intra-window drift)
        row.update(extract_mb_features(s123, fs))

        # TAP features (redesigned: uses cross-window context)
        row.update(extract_tap_features(
            s123, unit_g, dop,
            prev_window_mean=prev_window_mean,  # ← pass context
        ))

        row = {k: (0.0 if not math.isfinite(v) else v) for k, v in row.items()}

    else:
        # ... "all" mode unchanged ...
        pass

    return row


# ---------------------------------------------------------------------------
# Dataset building with time-based split
# ---------------------------------------------------------------------------

def build_dataset(
    files: list[dict],
    logger: logging.Logger,
    feature_mode: str = "domain_specific",  # FIX: accept and propagate feature_mode
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build train/test DataFrames from all files for one source."""
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

        n_total = len(df)
        n_train = math.ceil(TRAIN_FRACTION * n_total)
        df_train_raw = df.iloc[:n_train].reset_index(drop=True)
        df_test_raw = df.iloc[n_train:].reset_index(drop=True)

        for subset, df_sub, split_label in [
            (train_rows, df_train_raw, "train"),
            (test_rows, df_test_raw, "test"),
        ]:
            windows = extract_windows_from_file(
                fi, df_sub, fs, s_ref, logger, feature_mode=feature_mode  # FIX
            )
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
    order = [e for e in EVENT_ORDER if e in present]
    extra = [e for e in present if e not in order]
    classes = order + extra
    mapping = {c: i for i, c in enumerate(classes)}
    return events.map(mapping).values.astype(int), classes


def build_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract features and labels from DataFrame."""
    feat_cols = ALL_FEATURE_NAMES  # Uses the module-level (possibly overridden) list

    available_cols = [col for col in feat_cols if col in df.columns]

    if len(available_cols) < len(feat_cols):
        missing = set(feat_cols) - set(df.columns)
        print(f"⚠ Warning: Missing {len(missing)} features: {missing}")
        feat_cols = available_cols

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
) -> tuple[dict, SimpleImputer, dict]:
    """Train three models: XGBoost, HGB, GB."""
    from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier
    from sklearn.utils.class_weight import compute_class_weight

    logger.info("Pre-processing: impute + scale …")
    imputer = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # NE-baseline normalization
    ne_mask_train = (y_train == 'NE')
    ne_samples = X_train_imp[ne_mask_train]

    if ne_samples.shape[0] > 0:
        ne_mean = ne_samples.mean(axis=0)
        ne_std = ne_samples.std(axis=0)
        ne_std[ne_std == 0] = 1

        logger.info("✓ NE-Baseline Normalization:")
        logger.info("  - NE samples in training: %d", ne_samples.shape[0])

        X_train_sc = (X_train_imp - ne_mean) / ne_std
        X_test_sc = (X_test_imp - ne_mean) / ne_std
        ne_scaler = {"mean": ne_mean, "std": ne_std}
    else:
        logger.warning("⚠ No NE samples found! Falling back to StandardScaler")
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train_imp)
        X_test_sc = scaler.transform(X_test_imp)
        ne_scaler = {"mean": np.zeros(X_train_imp.shape[1]),
                     "std": np.ones(X_train_imp.shape[1])}

    n_classes = len(classes)
    models = {}

    # Class weights with VB boost
    class_weights_balanced = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights_balanced)}

    if "VB" in classes:
        vb_idx = list(classes).index("VB")
        class_weight_dict[vb_idx] *= 2.0
        logger.info("✓ VB Class Boost Applied (weight ×2): %s", class_weight_dict)

    # ===== 1. XGBOOST =====
    logger.info("\n" + "=" * 60)
    logger.info("1. Training XGBoost")
    logger.info("=" * 60)
    t0 = time.time()

    le_map = {c: i for i, c in enumerate(classes)}
    y_train_enc = np.array([le_map[c] for c in y_train])
    y_test_enc = np.array([le_map[c] for c in y_test])

    class_weights_xgb = compute_optimal_weights(y_train, classes)
    sample_weights = np.array([class_weights_xgb[le_map[c]] for c in y_train])

    # FIX: use a proper validation split from training data (no test-set leakage)
    from sklearn.model_selection import train_test_split
    X_tr_xgb, X_val_xgb, y_tr_xgb, y_val_xgb, sw_tr, _ = train_test_split(
        X_train_sc, y_train_enc, sample_weights,
        test_size=0.1, random_state=RANDOM_SEED, stratify=y_train_enc
    )

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        colsample_bytree=0.8,
        early_stopping_rounds=20,
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
        verbosity=0,
        n_jobs=-1,
    )
    xgb.fit(
        X_tr_xgb, y_tr_xgb,
        sample_weight=sw_tr,
        eval_set=[(X_val_xgb, y_val_xgb)],
        verbose=False,
    )
    xgb_time = time.time() - t0
    logger.info("  ✓ XGBoost trained in %.2fs", xgb_time)

    models["XGBoost"] = {
        "clf": xgb,
        "X_train": X_train_sc,
        "X_test": X_test_sc,
        "le_map": le_map,
        "y_train_enc": y_train_enc,
        "y_test_enc": y_test_enc,
        "training_time": xgb_time,
    }

    # ===== 2. HISTOGRAM GRADIENT BOOSTING =====
    logger.info("\n" + "=" * 60)
    logger.info("2. Training Histogram Gradient Boosting (HGB)")
    logger.info("=" * 60)
    t0 = time.time()

    hgb = HistGradientBoostingClassifier(
        loss='log_loss',
        learning_rate=0.1,
        max_iter=100,
        max_leaf_nodes=31,
        max_depth=10,
        min_samples_leaf=20,
        l2_regularization=0.0,
        early_stopping='auto',
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=RANDOM_SEED,
        verbose=0,
    )
    hgb.fit(X_train_sc, y_train)
    hgb_time = time.time() - t0
    logger.info("  ✓ HGB trained in %.2fs", hgb_time)

    models["HGB"] = {
        "clf": hgb,
        "X_train": X_train_sc,
        "X_test": X_test_sc,
        "training_time": hgb_time,
    }

    # ===== 3. GRADIENT BOOSTING =====
    logger.info("\n" + "=" * 60)
    logger.info("3. Training Gradient Boosting (GB)")
    logger.info("=" * 60)
    t0 = time.time()

    gb = GradientBoostingClassifier(
        loss='log_loss',
        learning_rate=0.1,
        n_estimators=100,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=RANDOM_SEED,
        verbose=0,
    )
    gb.fit(X_train_sc, y_train)
    gb_time = time.time() - t0
    logger.info("  ✓ GB trained in %.2fs", gb_time)

    models["GB"] = {
        "clf": gb,
        "X_train": X_train_sc,
        "X_test": X_test_sc,
        "training_time": gb_time,
    }

    logger.info("\n✓ All 3 models trained  |  XGBoost: %.2fs  HGB: %.2fs  GB: %.2fs",
                xgb_time, hgb_time, gb_time)

    return models, imputer, ne_scaler


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_model(name, info, y_test, X_test_raw, classes, logger):
    clf = info["clf"]
    X_test = info["X_test"]

    if name == "XGBoost":
        y_pred_enc = clf.predict(X_test)
        inv_map = {v: k for k, v in info["le_map"].items()}
        y_pred = np.array([inv_map[i] for i in y_pred_enc])
        y_proba = clf.predict_proba(X_test)
        # XGBoost uses integer classes — reorder to match string `classes`
        xgb_classes = [inv_map[i] for i in clf.classes_]
        col_order = [xgb_classes.index(c) for c in classes]
        y_proba = y_proba[:, col_order]
    else:
        y_pred = clf.predict(X_test)
        # FIX: reorder proba columns from clf.classes_ order → `classes` order
        clf_class_list = list(clf.classes_)
        col_order = [clf_class_list.index(c) for c in classes]
        y_proba = clf.predict_proba(X_test)[:, col_order]
    
    # ... rest of evaluation unchanged

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_mic = f1_score(y_test, y_pred, average="micro", zero_division=0)

    y_bin = label_binarize(y_test, classes=classes)
    n_cls = len(classes)
    if n_cls == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])
    try:
        auc_macro = roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")
        auc_micro = roc_auc_score(y_bin, y_proba, average="micro", multi_class="ovr")
    except Exception:
        auc_macro = auc_micro = float("nan")

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    report = classification_report(y_test, y_pred, labels=classes,
                                   target_names=classes, zero_division=0,
                                   output_dict=True)

    logger.info("  [%s] Accuracy=%.4f | Precision=%.4f | Recall=%.4f | F1=%.4f",
                name, acc, prec, rec, f1_mac)

    return {
        "name": name,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1_mac,
        "f1_micro": f1_mic,
        "auc_macro": auc_macro,
        "auc_micro": auc_micro,
        "cm": cm,
        "report": report,
        "training_time": info.get("training_time", 0),
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
    """5-fold cross-validation."""
    from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_results = {}

    for name, info in models.items():
        X = info["X_train"]
        logger.info("  CV: %s (5-fold) …", name)

        if name == "XGBoost":
            clf_cv = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                eval_metric="mlogloss",
                random_state=RANDOM_SEED,
                verbosity=0,
                n_jobs=-1,
            )
            y = info["y_train_enc"]
        elif name == "HGB":
            clf_cv = HistGradientBoostingClassifier(
                loss='log_loss',
                learning_rate=0.1,
                max_iter=100,
                random_state=RANDOM_SEED,
            )
            y = y_train
        else:
            clf_cv = GradientBoostingClassifier(
                loss='log_loss',
                learning_rate=0.1,
                n_estimators=100,
                random_state=RANDOM_SEED,
            )
            y = y_train

        try:
            scores = cross_val_score(clf_cv, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
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


def plot_model_performance_comparison(
    results: list[dict],
    cv_results: dict,
    source: str,
    out_dir: Path,
    logger: logging.Logger,
) -> None:
    """Compare performance of XGBoost, HGB, and GB models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Model Performance Comparison — {source}", fontsize=14, fontweight="bold")

    model_names = [r["name"] for r in results]
    colors = {"XGBoost": "steelblue", "HGB": "darkorange", "GB": "green"}
    bar_colors = [colors.get(m, "gray") for m in model_names]

    ax = axes[0, 0]
    accuracies = [r["accuracy"] for r in results]
    bars = ax.bar(model_names, accuracies, color=bar_colors, alpha=0.8)
    ax.set_ylabel("Accuracy")
    ax.set_ylim([0.94, 1.0])
    ax.axhline(y=0.98, color='red', linestyle='--', alpha=0.5, label='Target: 98%')
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{acc:.4f}", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Accuracy")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax = axes[0, 1]
    f1_scores = [r["f1_macro"] for r in results]
    bars = ax.bar(model_names, f1_scores, color=bar_colors, alpha=0.8)
    ax.set_ylabel("F1-Score (Macro)")
    ax.set_ylim([0.94, 1.0])
    ax.axhline(y=0.98, color='red', linestyle='--', alpha=0.5, label='Target: 98%')
    for bar, f1 in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{f1:.4f}", ha="center", va="bottom", fontweight="bold")
    ax.set_title("F1-Score (Macro)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 0]
    training_times = [r.get("training_time", 0) for r in results]
    bars = ax.bar(model_names, training_times, color=bar_colors, alpha=0.8)
    ax.set_ylabel("Training Time (seconds)")
    for bar, time_val in zip(bars, training_times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{time_val:.2f}s", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Training Time")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    precisions = [r["precision_macro"] for r in results]
    recalls = [r["recall_macro"] for r in results]
    for i, name in enumerate(model_names):
        ax.scatter(recalls[i], precisions[i], s=300, alpha=0.7,
                   color=colors.get(name, "gray"), label=name)
        ax.annotate(name, (recalls[i], precisions[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    ax.set_xlabel("Recall (Macro)")
    ax.set_ylabel("Precision (Macro)")
    ax.set_xlim([0.94, 1.0])
    ax.set_ylim([0.94, 1.0])
    ax.set_title("Precision-Recall Trade-off")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left')

    plt.tight_layout()
    output_file = out_dir / "10_model_performance_comparison.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("  ✓ Saved %s", output_file.name)


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
    cm = result["cm"].astype(float)
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)

    name = result["name"]
    acc = result["accuracy"]
    safe_name = name.lower().replace(" ", "_")
    idx = {"XGBoost": "01", "HGB": "02", "GB": "03"}.get(name, "0X")

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
    y_bin = label_binarize(y_test, classes=classes)
    n_cls = len(classes)
    if n_cls == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])

    fig, ax = plt.subplots(figsize=(10, 8))
    linestyles = ["-", "--", ":"]
    colors_model = ["steelblue", "darkorange", "green"]

    for m_idx, res in enumerate(results):
        y_prob = res["y_proba"]
        ls = linestyles[m_idx % len(linestyles)]
        mc = colors_model[m_idx]
        mname = res["name"]

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
        valid_cls = sum(1 for i in range(n_cls) if len(np.unique(y_bin[:, i])) > 1)
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
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_model = ["steelblue", "darkorange", "green"]
    win_size = 10

    for m_idx, res in enumerate(results):
        y_pred = res["y_pred"]
        n = len(y_pred)
        running_f1 = []
        indices = []
        for end in range(win_size, n + 1):
            start = max(0, end - win_size)
            f1_v = f1_score(y_test[start:end], y_pred[start:end],
                            average="macro", zero_division=0)
            running_f1.append(f1_v)
            indices.append(end)

        if running_f1:
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
    from sklearn.metrics import silhouette_score

    n_clusters = min(5, len(classes))
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_2d = pca.fit_transform(X_test)

    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    km_labels = km.fit_predict(X_test)

    try:
        sil = silhouette_score(X_test, km_labels,
                               sample_size=min(5000, len(X_test)),
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
        f"K-Means Clustering (k={n_clusters}) — {source}\nSilhouette = {sil:.3f}",
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
    present_classes = [c for c in classes if c in train_df["event"].values
                       or c in test_df["event"].values]

    train_counts = {c: (train_df["event"] == c).sum() for c in present_classes}
    test_counts = {c: (test_df["event"] == c).sum() for c in present_classes}

    x = np.arange(len(present_classes))
    width = 0.35

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
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
    model_names = [r["name"] for r in results]

    x = np.arange(len(metrics))
    width = 0.25
    colors_model = ["steelblue", "darkorange", "green"]

    fig, ax = plt.subplots(figsize=(11, 6))
    for m_idx, res in enumerate(results):
        vals = [res[m] for m in metrics]
        err = [0, 0, 0, cv_results.get(res["name"], np.zeros(5)).std()]
        offset = (m_idx - 1) * width
        bars = ax.bar(x + offset, vals, width, label=res["name"],
                      color=colors_model[m_idx], alpha=0.85, yerr=err, capsize=4)
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


def save_model_comparison_table(
    results: list[dict],
    cv_results: dict,
    source: str,
    out_dir: Path,
    logger: logging.Logger,
) -> None:
    data = []
    for res in results:
        cv_scores = cv_results.get(res["name"], np.zeros(5))
        data.append({
            "Model": res["name"],
            "Accuracy": f"{res['accuracy']:.4f}",
            "Precision": f"{res['precision_macro']:.4f}",
            "Recall": f"{res['recall_macro']:.4f}",
            "F1-Score": f"{res['f1_macro']:.4f}",
            "AUC (Macro)": f"{res['auc_macro']:.4f}",
            "CV F1 (5-fold)": f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}",
            "Training Time (s)": f"{res.get('training_time', 0):.2f}",
        })

    df = pd.DataFrame(data)
    output_file = out_dir / "11_model_comparison_table.csv"
    df.to_csv(output_file, index=False)
    logger.info("✓ Saved %s", output_file.name)
    logger.info("\n%s", df.to_string(index=False))


def plot_feature_distributions(
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
    feature_importances: np.ndarray,
    active_scalar_names: list[str],  # FIX: accept active scalar names instead of hardcoding
    source: str,
    out_dir: Path,
    logger: logging.Logger,
) -> None:
    """Box plots of the top 6 scalar features by importance."""
    # FIX: use the passed-in scalar names so the indices are correct for the current feature set
    n_scalar = len(active_scalar_names)
    scalar_importances = feature_importances[:n_scalar]
    top6_local = np.argsort(scalar_importances)[::-1][:6]
    top6_names = [active_scalar_names[i] for i in top6_local]
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
    lines = [
        "NOVOPTEL PM1000 — ML Pipeline Performance Summary",
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
        lines.append(f"  CV F1 (5-fold)  : {cv.mean():.4f} ± {cv.std():.4f}")
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
    # FIX: was 'utf=8' (typo) — corrected to 'utf-8'
    (out_dir / "model_performance.txt").write_text(txt, encoding='utf-8')
    logger.info("  Saved model_performance.txt")


def save_feature_importance(
    feature_importances: np.ndarray,
    active_feature_names: list[str],  # FIX: accept active names explicitly
    out_dir: Path,
    logger: logging.Logger,
    top_n: int = 50,
) -> None:
    """Save top-N features by importance."""
    # Guard against length mismatch
    n = min(len(feature_importances), len(active_feature_names))
    importances = feature_importances[:n]
    names = active_feature_names[:n]

    order = np.argsort(importances)[::-1][:top_n]
    df_fi = pd.DataFrame({
        "rank": range(1, len(order) + 1),
        "feature": [names[i] for i in order],
        "importance": importances[order],
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
    active_feature_names: list[str],
    active_scalar_names: list[str],
    out_dir: Path,
    logger: logging.Logger,
) -> None:
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
        "n_features": len(active_feature_names),
        "n_scalar_features": len(active_scalar_names),
        "classes": classes,
        "models": {
            "XGBoost": {"n_estimators": 200, "early_stopping_rounds": 20},
            "HGB": {"max_iter": 100, "early_stopping": "auto"},
            "GB": {"n_estimators": 100, "max_depth": 5},
        },
    }
    with open(out_dir / "config.yaml", "w") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)
    logger.info("  Saved config.yaml")


def plot_domain_specific_features(
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
    feature_names: list[str],
    source: str,
    out_dir: Path,
    logger: logging.Logger,
) -> None:
    """Visualize which domain-specific features matter most for each event."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"Domain-Specific Features by Event Type — {source}",
                 fontsize=14, fontweight="bold")

    axes_flat = axes.flatten()
    cmap = plt.cm.Set2

    event_groups = {
        'NE':  ['S1_mean', 'S2_mean', 'S3_mean', 'dop_mean'],
        'FS':  ['S1_std', 'S2_std', 'S3_std', 'S1_range'],
        'VB':  ['vb_core_power', 'vb_snr_db', 'vb_dominance_ratio', 'vb_peak_proximity'],
        'MB':  ['S1_kurtosis', 'S2_kurtosis', 'S3_kurtosis', 'spectral_entropy'],
        'TAP': ['S1_diff_max', 'S2_diff_max', 'S3_diff_max', 'vb_prominence'],
    }

    for ax_idx, (event, key_features) in enumerate(event_groups.items()):
        ax = axes_flat[ax_idx]

        if event not in classes:
            ax.text(0.5, 0.5, f'{event}\n(Not in dataset)',
                    ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        mask = y_test == event
        event_data = X_test[mask]

        if len(event_data) == 0:
            ax.text(0.5, 0.5, f'{event}\n(No samples)',
                    ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        feat_indices = []
        for feat in key_features:
            try:
                idx = feature_names.index(feat)
                feat_indices.append(idx)
            except ValueError:
                pass

        if not feat_indices:
            ax.text(0.5, 0.5, f'{event}\n(Features not found)',
                    ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        data_to_plot = [event_data[:, idx] for idx in feat_indices]
        available_labels = [key_features[j] for j, _ in enumerate(feat_indices)]

        bp = ax.boxplot(data_to_plot, patch_artist=True, notch=True,
                        labels=[f.replace('_', '\n') for f in available_labels])
        for patch, color in zip(bp['boxes'],
                                cmap(np.linspace(0, 1, len(feat_indices)))):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(f'{event} ({len(event_data)} samples)', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_file = out_dir / "15_domain_specific_features_by_event.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("  ✓ Saved %s", output_file.name)


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
    p.add_argument(
        "--feature_mode", type=str, default="domain_specific",
        choices=["domain_specific", "all"],
        help="Feature extraction mode.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ===== FEATURE MODE — set once, used everywhere =====
    # FIX: was defined twice in the original; now defined once here and
    # propagated via function arguments rather than re-assigned later.
    FEATURE_MODE: str = args.feature_mode  # "domain_specific" or "all"

    global ALL_FEATURE_NAMES
    if FEATURE_MODE == "domain_specific":
        ALL_FEATURE_NAMES = DOMAIN_SPECIFIC_ALL_FEATURES
        active_scalar_names = DOMAIN_SPECIFIC_SCALAR_FEATURES
    else:
        ALL_FEATURE_NAMES = SCALAR_FEATURE_NAMES + [f"fft_bin_{i}" for i in range(N_FFT_BINS)]
        active_scalar_names = SCALAR_FEATURE_NAMES

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
    logger.info("Source       : %s  (tokens: %s)", source, source_tokens)
    logger.info("Data dir     : %s", args.data_dir)
    logger.info("Output dir   : %s", out_dir)
    logger.info("Feature mode : %s (%d features)", FEATURE_MODE, len(ALL_FEATURE_NAMES))
    logger.info("Random seed  : %d", RANDOM_SEED)

    np.random.seed(RANDOM_SEED)

    # ================================================================
    # 1. File discovery
    # ================================================================
    data_dir = args.data_dir
    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    files = discover_files(data_dir, source_tokens)
    if not files:
        logger.error(
            "No files found for source '%s' in %s.  Available sources: %s",
            source, data_dir,
            sorted({f['source'] for f in discover_files(
                data_dir,
                [t for tokens in SOURCE_ALIASES.values() for t in tokens]
            )}),
        )
        sys.exit(1)

    detected_events = sorted({fi["event"] for fi in files})
    logger.info("Files found: %d  |  Events: %s", len(files), detected_events)

    # ================================================================
    # 2. Build train / test DataFrames
    # ================================================================
    logger.info("\nBuilding dataset (feature extraction) …")
    t_build0 = time.time()
    # FIX: pass feature_mode so windows use the correct extraction branch
    train_df, test_df = build_dataset(files, logger, feature_mode=FEATURE_MODE)
    logger.info(
        "Dataset ready in %.1f s  |  train=%d windows  test=%d windows",
        time.time() - t_build0, len(train_df), len(test_df),
    )

    if len(train_df) == 0 or len(test_df) == 0:
        logger.error("Insufficient data after windowing.  Exiting.")
        sys.exit(1)

    all_events = sorted(set(train_df["event"].tolist() + test_df["event"].tolist()))
    classes = [e for e in EVENT_ORDER if e in all_events]
    classes += [e for e in all_events if e not in classes]
    logger.info("Classes: %s", classes)

    X_train_raw, y_train = build_xy(train_df)
    X_test_raw, y_test = build_xy(test_df)

    # ================================================================
    # 3. Train models
    # ================================================================
    logger.info("\nTraining models …")
    models, imputer, ne_scaler = train_models(
        X_train_raw, y_train, X_test_raw, y_test, classes, logger
    )

    # ================================================================
    # 4. Apply normalization to test / train data for evaluation & plots
    # ================================================================
    X_test_imp = imputer.transform(X_test_raw)
    ne_mean = ne_scaler["mean"]
    ne_std = ne_scaler["std"]
    X_test_scaled = (X_test_imp - ne_mean) / ne_std

    X_train_imp = imputer.transform(X_train_raw)
    X_train_sc = (X_train_imp - ne_mean) / ne_std

    # ================================================================
    # 5. Evaluate models
    # ================================================================
    logger.info("\nEvaluating models …")
    results = []
    for name, info in models.items():
        res = evaluate_model(name, info, y_test, X_test_raw, classes, logger)
        results.append(res)

    # ================================================================
    # 6. Cross-validation
    # ================================================================
    logger.info("\nRunning 5-fold CV on training set …")
    cv_results = cross_validate_models(models, X_train_sc, y_train, classes, logger)

    # ================================================================
    # 7. Feature importances (from XGBoost)
    # ================================================================
    xgb_importances = models["XGBoost"]["clf"].feature_importances_

    # ================================================================
    # 8. Generate plots
    # ================================================================
    logger.info("\nGenerating plots …")

    for res in results:
        plot_confusion_matrix(res, classes, source, plots_dir, logger)

    plot_roc_curves(results, classes, y_test, source, plots_dir, logger)
    plot_learning_curves(results, y_test, source, plots_dir, logger)
    plot_kmeans_clustering(X_test_scaled, y_test, classes, source, plots_dir, logger)
    plot_class_distribution(train_df, test_df, classes, source, plots_dir, logger)
    plot_model_comparison(results, cv_results, source, plots_dir, logger)

    # FIX: pass active_scalar_names so indices are correct for the current feature set
    plot_feature_distributions(
        X_test_scaled, y_test, classes,
        xgb_importances, active_scalar_names,
        source, plots_dir, logger,
    )

    plot_model_performance_comparison(results, cv_results, source, plots_dir, logger)

    if FEATURE_MODE == "domain_specific":
        plot_domain_specific_features(
            X_test_scaled, y_test, classes,
            ALL_FEATURE_NAMES,
            source, plots_dir, logger,
        )

    # ================================================================
    # 9. Save output files
    # ================================================================
    logger.info("\nSaving output files …")
    save_results_csv(test_df, results, classes, out_dir, logger)
    save_performance_txt(results, cv_results, classes, source, out_dir, logger)
    # FIX: pass active feature names explicitly
    save_feature_importance(xgb_importances, ALL_FEATURE_NAMES, out_dir, logger)
    save_model_comparison_table(results, cv_results, source, out_dir, logger)
    save_config_yaml(
        args, source, source_tokens,
        len(train_df), len(test_df), classes,
        ALL_FEATURE_NAMES, active_scalar_names,
        out_dir, logger,
    )

    # ================================================================
    # 10. Final summary
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("Output directory: %s", out_dir)
    logger.info("\nModel Performance Summary:")
    for res in results:
        logger.info("  %-15s  acc=%.4f  F1=%.4f  AUC=%.4f",
                    res["name"], res["accuracy"], res["f1_macro"], res["auc_macro"])
    logger.info("\nPlots saved to: %s", plots_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()