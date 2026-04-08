"""
main.py  —  new_ml_pipeline_2
==============================
Orchestrator: feature extraction → source-normalisation → per-source LOSO.

Steps
-----
1. Auto-run feature extraction if features_1s_normalised.csv is missing.
2. Load features_1s_normalised.csv.
3. Print dataset summary.
4. Per-source LOSO analysis (RF, SVM, XGBoost).
5. Per-source LOSO learning curves.
6. Combined LOSO confusion matrix figures.
7. Print final summary table.

Usage (from the repository root)::

    python new_ml_pipeline_2/main.py
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import feature_extraction
from feature_extraction import FFT_BIN_COLS
from ml_classification import (
    run_per_source_loso,
    print_loso_summary_table,
    EVENTS,
    SOURCES,
    FEATURE_COLS,
)
from visualisation import (
    generate_loso_learning_curves,
    generate_all_plots,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "outputs")
FEATURES_CSV = os.path.join(SCRIPT_DIR, "features_1s_normalised.csv")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def load_features(csv_path: str):
    """Load features_1s_normalised.csv; return X, y_event, y_source."""
    print(f"Loading normalised features from {csv_path} ...")
    df = pd.read_csv(csv_path)
    df["event"]  = df["event"].str.upper()
    df["source"] = df["source"].str.upper()

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  WARNING: missing feature columns ({len(missing)}): {missing[:5]}...")

    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    X        = df[available_cols]
    y_event  = df["event"]
    y_source = df["source"]

    print(f"  {len(X)} windows x {len(available_cols)} features  "
          f"({y_event.nunique()} event classes, {y_source.nunique()} sources)")
    return X, y_event, y_source, available_cols


def print_dataset_summary(y_event: pd.Series, y_source: pd.Series) -> None:
    print("\n=== Dataset summary (windows per source x event) ===")
    df    = pd.DataFrame({"event": y_event, "source": y_source})
    pivot = df.pivot_table(index="source", columns="event",
                           aggfunc="size", fill_value=0)
    for ev in EVENTS:
        if ev not in pivot.columns:
            pivot[ev] = 0
    src_present = [s for s in SOURCES if s in pivot.index]
    ev_present  = [e for e in EVENTS  if e in pivot.columns]
    pivot = pivot.loc[src_present, ev_present] if src_present else pivot
    print(pivot.to_string())
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    warnings.filterwarnings("ignore")

    print("=" * 70)
    print("NOVOPTEL PM1000 — new_ml_pipeline_2")
    print("(Source-normalised features, per-source LOSO only)")
    print("=" * 70)

    # Step 1 — Auto-run feature extraction if CSV is missing
    if not os.path.exists(FEATURES_CSV):
        print(f"\nFeatures CSV not found: {FEATURES_CSV}")
        print("Running feature extraction (includes source-normalisation) ...\n")
        feature_extraction.main()

    if not os.path.exists(FEATURES_CSV):
        print(f"ERROR: Feature extraction did not produce {FEATURES_CSV}")
        sys.exit(1)

    # Step 2 — Load feature matrix
    X, y_event, y_source, feature_names = load_features(FEATURES_CSV)

    # Step 3 — Dataset summary
    print_dataset_summary(y_event, y_source)

    # Step 4 — Per-source LOSO analysis
    print("\n--- Step 4: Per-source LOSO analysis ---")
    per_source_results = run_per_source_loso(X, y_event, y_source, OUTPUT_DIR)

    # Step 5 — Per-source LOSO learning curves
    print("\n--- Step 5: LOSO learning curves ---")
    try:
        generate_loso_learning_curves(
            X, y_event, y_source, per_source_results, OUTPUT_DIR
        )
    except Exception as exc:
        print(f"  [WARN] Learning curves failed: {exc}")

    # Step 6 — Combined CM figures
    print("\n--- Step 6: Combined confusion matrix figures ---")
    try:
        generate_all_plots(per_source_results, OUTPUT_DIR)
    except Exception as exc:
        print(f"  [WARN] Combined CM figures failed: {exc}")

    # Step 7 — Final summary table
    print_loso_summary_table(per_source_results)

    print(f"\nDone. All outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
