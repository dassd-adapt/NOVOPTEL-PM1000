"""
main.py  —  new_ml_pipeline
============================
Orchestrator: calls feature extraction → ML classification → plots.

Steps
-----
1. Check for ``features_1s_windows.csv``; auto-run feature extraction if missing.
2. Load CSV, build X (41 features), y_event, y_source, date column.
3. Print dataset summary (windows per source × event).
4. ``run_full_pipeline()`` — global models (stratified 5-fold + LOSO for RF, SVM, XGB).
5. ``run_per_source_pipeline()`` — per-source models (stratified 5-fold + LODO).
6. ``generate_all_plots()`` — all visualisations.
7. Print final summary table.

Usage (from the repository root)::

    python new_ml_pipeline/main.py

Or from inside new_ml_pipeline/::

    python main.py
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

# Allow running from either the repo root or the new_ml_pipeline/ directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import feature_extraction
from ml_classification import (
    run_full_pipeline,
    run_per_source_pipeline,
    EVENTS,
    SOURCES,
    FEATURE_COLS,
)
from visualisation import generate_all_plots

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATASET_DIR = os.path.join(REPO_ROOT, "dataset-1603")
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "outputs")
FEATURES_CSV = os.path.join(SCRIPT_DIR, "features_1s_windows.csv")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def load_features(csv_path: str):
    """Load features_1s_windows.csv and return X, y_event, y_source, date, feature_names."""
    print(f"Loading windowed features from {csv_path} …")
    df = pd.read_csv(csv_path)

    # Normalise case for event/source labels
    df["event"]  = df["event"].str.upper()
    df["source"] = df["source"].str.upper()

    # Ensure all FEATURE_COLS are present
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  WARNING: missing feature columns: {missing}")

    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    X        = df[available_cols]
    y_event  = df["event"]
    y_source = df["source"]
    date_col = df["date"] if "date" in df.columns else pd.Series(["unknown"] * len(df))

    print(f"  {len(X)} windows × {len(available_cols)} features  "
          f"({y_event.nunique()} event classes, {y_source.nunique()} sources)")
    return X, y_event, y_source, date_col, available_cols


def print_dataset_summary(y_event: pd.Series, y_source: pd.Series) -> None:
    """Print count of windows per (source, event) combination."""
    print("\n=== Dataset summary (windows per source × event) ===")
    df    = pd.DataFrame({"event": y_event, "source": y_source})
    pivot = df.pivot_table(index="source", columns="event",
                           aggfunc="size", fill_value=0)
    for ev in EVENTS:
        if ev not in pivot.columns:
            pivot[ev] = 0
    src_present = [s for s in SOURCES if s in pivot.index]
    ev_present  = [e for e in EVENTS  if e in pivot.columns]
    pivot = pivot.loc[src_present, ev_present]
    print(pivot.to_string())
    print()

    missing = []
    for src in SOURCES:
        for ev in EVENTS:
            count = int(pivot.loc[src, ev]) if src in pivot.index and ev in pivot.columns else 0
            if count == 0:
                missing.append(f"{src}/{ev}")
    if missing:
        print("Missing (source, event) combinations:", ", ".join(missing))
    else:
        print("All (source, event) combinations present.")
    print()


def print_summary_table(results: dict, per_source_results: dict) -> None:
    """Print model / CV strategy / accuracy / macro-F1 table."""
    from sklearn.metrics import f1_score

    rows = []
    for clf_name in ("RF", "SVM", "XGBoost"):
        cv_key   = f"cv_{clf_name}"
        loso_key = f"loso_{clf_name}"

        if cv_key in results:
            r = results[cv_key]
            rows.append({
                "Model":       clf_name,
                "CV strategy": "Stratified 5-fold (global)",
                "Accuracy":    f"{r['mean_acc']:.3f} ± {r['std_acc']:.3f}",
                "Macro-F1":    f"{r['macro_f1']:.3f}",
            })
        if loso_key in results:
            r = results[loso_key]
            rows.append({
                "Model":       clf_name,
                "CV strategy": "Leave-One-Source-Out (global)",
                "Accuracy":    f"{r['overall_acc']:.3f}",
                "Macro-F1":    f"{r['macro_f1']:.3f}",
            })

    df = pd.DataFrame(rows)
    print("\n=== Global model performance summary ===")
    print(df.to_string(index=False))
    print()

    # Per-source summary
    if per_source_results:
        print("=== Per-source summary (stratified 5-fold CV accuracy) ===")
        header = f"{'Source':<20}" + "".join(f"{'RF':>10}{'SVM':>10}{'XGBoost':>10}")
        print(header)
        print("-" * (20 + 30))
        for src in SOURCES:
            if src not in per_source_results:
                continue
            line = f"{src:<20}"
            for clf_name in ("RF", "SVM", "XGBoost"):
                cv_res = per_source_results[src].get(clf_name, {}).get("strat_cv", {})
                acc = cv_res.get("mean_acc", float("nan"))
                line += f"{acc:>10.3f}"
            print(line)
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    warnings.filterwarnings("ignore")

    print("=" * 65)
    print("NOVOPTEL PM1000 — New ML Pipeline (1-second windows, TD+FD)")
    print("=" * 65)

    # Step 1 — Auto-run feature extraction if CSV is missing
    if not os.path.exists(FEATURES_CSV):
        print(f"\nFeatures CSV not found: {FEATURES_CSV}")
        print("Running feature extraction …\n")
        feature_extraction.main()

    if not os.path.exists(FEATURES_CSV):
        print(f"ERROR: Feature extraction did not produce {FEATURES_CSV}")
        sys.exit(1)

    # Step 2 — Load feature matrix
    X, y_event, y_source, date_col, feature_names = load_features(FEATURES_CSV)

    # Step 3 — Dataset summary
    print_dataset_summary(y_event, y_source)

    # Optionally save the feature matrix
    fm_path = os.path.join(OUTPUT_DIR, "feature_matrix_1s_windows.csv")
    fm_df   = X.copy()
    fm_df.insert(0, "date",   date_col.values)
    fm_df.insert(0, "event",  y_event.values)
    fm_df.insert(0, "source", y_source.values)
    fm_df.to_csv(fm_path, index=False)
    print(f"Feature matrix saved to {fm_path}")

    # Step 4 — Global classification (all sources)
    print("\n--- Step 4: Global classification (all sources combined) ---")
    results = run_full_pipeline(X, y_event, y_source, OUTPUT_DIR)

    # Step 5 — Per-source classification
    print("\n--- Step 5: Per-source classification (5 sources × 5 events) ---")
    per_source_results = run_per_source_pipeline(
        X, y_event, y_source, date_col, OUTPUT_DIR
    )

    # Step 6 — Visualisations
    print("\n--- Step 6: Generating visualisations ---")
    generate_all_plots(
        X, y_event, y_source, results, feature_names,
        DATASET_DIR, OUTPUT_DIR,
        per_source_results=per_source_results,
    )

    # Step 7 — Final summary table
    print_summary_table(results, per_source_results)

    print(f"\nDone. All outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
