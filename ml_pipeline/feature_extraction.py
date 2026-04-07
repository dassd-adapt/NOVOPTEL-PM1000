"""
feature_extraction.py
=====================
Windowed feature extraction pipeline for the NOVOPTEL PM1000 SOP dataset.

Each ~100-second CSV file is segmented into overlapping windows, and a set of
25 features is computed per window.  The result is written to
``analysis/features_windowed.csv``.

PM1000 normalisation note
-------------------------
The PM1000 was configured in **Standard Normalisation** mode for all recordings:
S1, S2, S3 in every CSV are already normalised to the unit Poincaré sphere
(‖[S1, S2, S3]‖ ≈ 1).  S0 ≈ 1 throughout and must NOT be used to re-normalise
S1/S2/S3 — doing so would divide ~1 by ~1 and is conceptually incorrect.

This applies equally to SP sources (10GE, SP-AGIL, SP-PURE) and DP sources
(DPQAM16-200G, DPQPSK-200G).  A single defensive unit-normalisation step is
applied after loading to absorb sub-LSB floating-point noise.

Usage
-----
From the repository root::

    python analysis/feature_extraction.py

Or from inside the ``analysis/`` directory::

    python feature_extraction.py
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

WINDOW_S: float = 6.0       # window duration (seconds)
OVERLAP: float = 0.5        # fractional overlap between consecutive windows
MIN_SAMPLES: int = 100      # minimum raw points required to keep a window

MIN_FILE_SIZE_BYTES: int = 1_000_000   # 1 MB — skip truncated files

# DOP gating threshold
DOP_GATE: float = 0.2

# PSD band edges (Hz)
BAND_LOW = (1.0, 20.0)
BAND_MID = (20.0, 100.0)
BAND_HIGH = (100.0, 500.0)

# ---------------------------------------------------------------------------
# Output paths (resolved relative to this file so the script works from
# anywhere)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset-1603"
OUTPUT_CSV = REPO_ROOT / "analysis" / "features_windowed.csv"

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

    The PM1000 stores S1/S2/S3 in Standard Normalisation mode, so
    ‖[S1, S2, S3]‖ ≈ 1 already.  This step absorbs sub-LSB floating-point
    noise only; it does NOT divide by S0.

    Returns ``(unit_vecs, dop)`` where rows with DOP == 0 are set to NaN.
    DOP here is ‖[S1, S2, S3]‖ which equals the true optical DOP because
    the vectors are already S0-normalised by the instrument.
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
    # np.trapezoid was added in NumPy 2.0; fall back to np.trapz for older versions
    try:
        _trapz = np.trapezoid
    except AttributeError:
        _trapz = np.trapz  # type: ignore[attr-defined]
    return float(_trapz(psd[mask], freqs[mask]))

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

        # Parse filename tokens
        # Format: pm1000_sop_2kHz_1min_{SOURCE}_{EVENT}_1550_{DATE}_{REP}.csv
        stem = csv_path.stem
        parts = stem.split("_")
        # prefix tokens: pm1000, sop, 2kHz, 1min  → 4 tokens
        prefix_len = 4
        suffix_tokens = parts[-3:]   # ['1550', DATE, REP]
        if len(suffix_tokens) != 3 or suffix_tokens[0] != "1550":
            print(f"  [SKIP] Cannot parse filename: {csv_path.name}")
            continue

        date_str = suffix_tokens[1]
        rep_str = suffix_tokens[2]

        # Middle tokens are SOURCE and EVENT — split on first recognised event tag
        middle_tokens = parts[prefix_len:-3]
        event_tags = {"NE", "FS", "VB", "MB", "TAP"}
        event_idx = None
        for i, tok in enumerate(middle_tokens):
            if tok in event_tags:
                event_idx = i
                break

        if event_idx is None:
            print(f"  [SKIP] Cannot identify event tag: {csv_path.name}")
            continue

        source = "_".join(middle_tokens[:event_idx])
        event = middle_tokens[event_idx]

        records.append(
            {
                "path": csv_path,
                "source": source,
                "event": event,
                "date": date_str,
                "rep": rep_str,
            }
        )

    return records

# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def load_and_normalise(file_info: dict) -> pd.DataFrame | None:
    """Load a CSV and return a DataFrame with defensively unit-normalised S1, S2, S3.

    The PM1000 records in Standard Normalisation mode: S1, S2, S3 are already
    on the unit Poincaré sphere for ALL source types (SP and DP alike).
    S0 ≈ 1 throughout and is NOT used to re-normalise S1/S2/S3.

    A single defensive pass divides each row by ‖[S1, S2, S3]‖ to absorb
    sub-LSB floating-point noise; rows with zero norm are set to NaN.

    Returns ``None`` on read errors.
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

    # Defensive unit normalisation — same for all sources.
    # S0 is intentionally ignored here; the instrument already normalised
    # S1/S2/S3 to the unit sphere in Standard Normalisation mode.
    s123 = df[["S1", "S2", "S3"]].values.astype(float)
    norms = np.linalg.norm(s123, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)   # avoid divide-by-zero
    s123_unit = s123 / norms

    df["S1"] = s123_unit[:, 0]
    df["S2"] = s123_unit[:, 1]
    df["S3"] = s123_unit[:, 2]

    return df

# ---------------------------------------------------------------------------
# Feature extraction for a single window
# ---------------------------------------------------------------------------
def extract_window_features(
    win: pd.DataFrame,
    s_ref: np.ndarray,
) -> dict:
    """Compute all 25 features for a single window *win*.

    Parameters
    ----------
    win:
        Slice of the file DataFrame for this window.
    s_ref:
        (3,) unit vector — mean reference SOP computed from the **entire file**.
    """
    t = win["Time_s"].values.astype(float)
    s123 = win[["S1", "S2", "S3"]].values.astype(float)

    # Estimate sampling frequency from timestamps
    dt = np.diff(t)
    dt = dt[dt > 0]
    fs = 1.0 / np.median(dt) if len(dt) > 0 else 1.0

    # Unit vectors and DOP (‖[S1,S2,S3]‖ = true DOP because already S0-normalised)
    unit, dop = _unit_normalise(s123)

    # DOP features -------------------------------------------------------
    dop_mean = float(np.nanmean(dop))
    dop_std = float(np.nanstd(dop))
    dop_min = float(np.nanmin(dop))
    dop_max = float(np.nanmax(dop))

    # DOP gating
    gate = dop > DOP_GATE
    unit_gated = unit.copy()
    unit_gated[~gate] = np.nan

    # Step-angle features ------------------------------------------------
    n = len(unit_gated)
    if n >= 2:
        a = unit_gated[:-1]
        b = unit_gated[1:]
        valid = ~(np.any(np.isnan(a), axis=1) | np.any(np.isnan(b), axis=1))
        if valid.sum() > 0:
            angles = _geodesic_angle(a[valid], b[valid])
            step_mean = float(np.nanmean(angles))
            step_std = float(np.nanstd(angles))
            step_max = float(np.nanmax(angles))
            step_rms = float(np.sqrt(np.nanmean(angles ** 2)))
            step_p95 = float(np.nanpercentile(angles, 95))
            cum_arc = float(np.nansum(angles))
        else:
            step_mean = step_std = step_max = step_rms = step_p95 = cum_arc = 0.0
    else:
        step_mean = step_std = step_max = step_rms = step_p95 = cum_arc = 0.0

    # θ_ref features -----------------------------------------------------
    valid_rows = ~np.any(np.isnan(unit_gated), axis=1)
    if valid_rows.sum() > 0 and not np.any(np.isnan(s_ref)):
        ref_tiled = np.tile(s_ref, (valid_rows.sum(), 1))
        theta = _geodesic_angle(unit_gated[valid_rows], ref_tiled)
        theta_mean = float(np.nanmean(theta))
        theta_std = float(np.nanstd(theta))
        theta_max = float(np.nanmax(theta))
        theta_rms = float(np.sqrt(np.nanmean(theta ** 2)))
    else:
        theta_mean = theta_std = theta_max = theta_rms = 0.0

    # PSD / spectral features (x = ||ŝ - ŝ_ref||) -----------------------
    if valid_rows.sum() >= 2 and not np.any(np.isnan(s_ref)):
        x = np.linalg.norm(unit_gated[valid_rows] - s_ref, axis=1)
        nperseg = min(256, len(x))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            freqs, psd_vals = welch(x, fs=fs, nperseg=nperseg)
        if len(psd_vals) > 0:
            peak_idx = int(np.argmax(psd_vals))
            psd_peak_freq = float(freqs[peak_idx])
            psd_peak_power = float(psd_vals[peak_idx])
            bp_low = _band_power(freqs, psd_vals, *BAND_LOW)
            bp_mid = _band_power(freqs, psd_vals, *BAND_MID)
            bp_high = _band_power(freqs, psd_vals, *BAND_HIGH)
        else:
            psd_peak_freq = psd_peak_power = bp_low = bp_mid = bp_high = 0.0
    else:
        psd_peak_freq = psd_peak_power = bp_low = bp_mid = bp_high = 0.0

    # Stokes trajectory features -----------------------------------------
    s1_std = float(np.nanstd(s123[:, 0]))
    s2_std = float(np.nanstd(s123[:, 1]))
    s3_std = float(np.nanstd(s123[:, 2]))
    s1_range = float(np.nanmax(s123[:, 0]) - np.nanmin(s123[:, 0]))
    s2_range = float(np.nanmax(s123[:, 1]) - np.nanmin(s123[:, 1]))
    s3_range = float(np.nanmax(s123[:, 2]) - np.nanmin(s123[:, 2]))

    row = {
        # DOP (4)
        "dop_mean": dop_mean,
        "dop_std": dop_std,
        "dop_min": dop_min,
        "dop_max": dop_max,
        # Step-angle (6)
        "step_mean": step_mean,
        "step_std": step_std,
        "step_max": step_max,
        "step_rms": step_rms,
        "step_p95": step_p95,
        "cum_arc": cum_arc,
        # θ_ref (4)
        "theta_mean": theta_mean,
        "theta_std": theta_std,
        "theta_max": theta_max,
        "theta_rms": theta_rms,
        # PSD (5)
        "psd_peak_freq": psd_peak_freq,
        "psd_peak_power": psd_peak_power,
        "bp_low": bp_low,
        "bp_mid": bp_mid,
        "bp_high": bp_high,
        # Stokes trajectory (6)
        "s1_std": s1_std,
        "s2_std": s2_std,
        "s3_std": s3_std,
        "s1_range": s1_range,
        "s2_range": s2_range,
        "s3_range": s3_range,
    }

    # Replace any remaining NaN / inf with 0
    return {k: (0.0 if (not math.isfinite(v)) else v) for k, v in row.items()}

# ---------------------------------------------------------------------------
# Per-file windowing
# ---------------------------------------------------------------------------
def process_file(file_info: dict) -> list[dict]:
    """Segment one file into windows and extract features from each.

    Returns a list of row dicts ready to be appended to the output DataFrame.
    """  
    df = load_and_normalise(file_info)
    if df is None:
        return []

    t = df["Time_s"].values.astype(float)
    t_start_file = t[0]
    t_end_file = t[-1]
    duration = t_end_file - t_start_file

    if duration <= 0:
        return []

    # Compute mean reference SOP from the **entire file** (DOP-gated)
    s123_all = df[["S1", "S2", "S3"]].values.astype(float)
    unit_all, dop_all = _unit_normalise(s123_all)
    gate_all = dop_all > DOP_GATE
    unit_all[~gate_all] = np.nan

    valid_all = ~np.any(np.isnan(unit_all), axis=1)
    if valid_all.sum() > 0:
        raw_mean = np.nanmean(unit_all[valid_all], axis=0)
        ref_norm = np.linalg.norm(raw_mean)
        s_ref = raw_mean / ref_norm if ref_norm > 0 else np.full(3, np.nan)
    else:
        s_ref = np.full(3, np.nan)

    # Sliding window parameters
    step_s = WINDOW_S * (1.0 - OVERLAP)

    rows = []
    window_idx = 0
    win_t_start = t_start_file

    while win_t_start + WINDOW_S <= t_end_file + 1e-9:
        win_t_end = win_t_start + WINDOW_S
        mask = (t >= win_t_start) & (t < win_t_end)
        n_pts = mask.sum()

        if n_pts < MIN_SAMPLES:
            win_t_start += step_s
            window_idx += 1
            continue

        win_df = df.loc[mask].reset_index(drop=True)
        feats = extract_window_features(win_df, s_ref)

        row = {
            "source": file_info["source"],
            "event": file_info["event"],
            "date": file_info["date"],
            "replicate": file_info["rep"],
            "window_idx": window_idx,
            "t_start_s": round(win_t_start, 4),
            "t_end_s": round(win_t_end, 4),
        }
        row.update(feats)
        rows.append(row)

        win_t_start += step_s
        window_idx += 1

    return rows

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the full feature extraction pipeline."""  
    print("=" * 60)
    print("NOVOPTEL PM1000 — Windowed Feature Extraction")
    print("=" * 60)
    print(f"Dataset directory : {DATASET_DIR}")
    print(f"Output CSV        : {OUTPUT_CSV}")
    print(
        f"Window parameters : {WINDOW_S}s window, {int(OVERLAP * 100)}% overlap, "
        f"min {MIN_SAMPLES} samples"
    )
    print()

    # --- Discover files -------------------------------------------------
    print("Discovering CSV files …")
    file_list = discover_files(DATASET_DIR)
    print(f"  Found {len(file_list)} valid file(s)\n")

    if not file_list:
        print("No files to process.  Exiting.")
        return

    # --- Process files --------------------------------------------------
    all_rows: list[dict] = []
    summary: dict[str, dict] = {}  # event → {files, windows}

    for fi in file_list:
        event = fi["event"]
        if event not in summary:
            summary[event] = {"files": 0, "windows": 0}
        summary[event]["files"] += 1

        rows = process_file(fi)
        all_rows.extend(rows)
        summary[event]["windows"] += len(rows)

    # --- Write output ---------------------------------------------------
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "source", "event", "date", "replicate", "window_idx",
        "t_start_s", "t_end_s",
        # DOP
        "dop_mean", "dop_std", "dop_min", "dop_max",
        # Step-angle
        "step_mean", "step_std", "step_max", "step_rms", "step_p95", "cum_arc",
        # θ_ref
        "theta_mean", "theta_std", "theta_max", "theta_rms",
        # PSD
        "psd_peak_freq", "psd_peak_power", "bp_low", "bp_mid", "bp_high",
        # Stokes trajectory
        "s1_std", "s2_std", "s3_std", "s1_range", "s2_range", "s3_range",
    ]

    out_df = pd.DataFrame(all_rows, columns=columns)
    out_df.to_csv(OUTPUT_CSV, index=False)

    # --- Print summary --------------------------------------------------
    print("=== Feature Extraction Summary ===")
    print(f"{'Event':<8} {'Files':>6} {'Windows':>8}")
    print("-" * 26)
    total_windows = 0
    total_files = 0
    for event in sorted(summary):
        f = summary[event]["files"]
        w = summary[event]["windows"]
        print(f"{event:<8} {f:>6} {w:>8}")
        total_windows += w
        total_files += f

    print("-" * 26)
    print(f"{'TOTAL':<8} {total_files:>6} {total_windows:>8}")
    print()
    print(f"Total windows: {total_windows} across {total_files} files")
    print(f"Output: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
