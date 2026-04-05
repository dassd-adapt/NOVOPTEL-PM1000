"""
main.py
=======
Orchestrates the full NOVOPTEL PM1000 Stokes polarimeter ML pipeline end-to-end.

Usage (from the ml_pipeline/ directory):
    python main.py
"""

import os
import sys
import warnings

import pandas as pd

# Allow running from the ml_pipeline/ directory directly
sys.path.insert(0, os.path.dirname(__file__))

from ml_classification import build_feature_matrix, run_full_pipeline, EVENTS, SOURCES
from visualisation import generate_all_plots

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "dataset-1603"))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")


def print_dataset_summary(y_event, y_source):
    """Print count of files per (source, event) combination."""
    print("\n=== Dataset summary ===")
    df = pd.DataFrame({"event": y_event, "source": y_source})
    pivot = df.pivot_table(index="source", columns="event", aggfunc="size",
                           fill_value=0)
    # Reorder axes
    for ev in EVENTS:
        if ev not in pivot.columns:
            pivot[ev] = 0
    for src in SOURCES:
        if src not in pivot.index:
            pivot.loc[src] = 0
    pivot = pivot.loc[[s for s in SOURCES if s in pivot.index],
                      [e for e in EVENTS if e in pivot.columns]]
    print(pivot.to_string())
    print()

    # Missing combinations
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


def print_summary_table(results):
    """Print model/CV strategy/accuracy/macro-F1 table to console."""
    from sklearn.metrics import f1_score
    rows = []
    for name, cv_key in [("RF", "cv_rf"), ("SVM", "cv_svm")]:
        r = results[cv_key]
        f1 = f1_score(r["all_true"], r["all_pred"],
                      labels=EVENTS, average="macro", zero_division=0)
        rows.append({
            "Model": name,
            "CV strategy": "Stratified 5-fold",
            "Accuracy": f"{r['mean_acc']:.3f} ± {r['std_acc']:.3f}",
            "Macro-F1": f"{f1:.3f}",
        })
    for name, loso_key in [("RF", "loso_rf"), ("SVM", "loso_svm")]:
        r = results[loso_key]
        f1 = f1_score(r["all_true"], r["all_pred"],
                      labels=EVENTS, average="macro", zero_division=0)
        rows.append({
            "Model": name,
            "CV strategy": "Leave-One-Source-Out",
            "Accuracy": f"{r['overall_acc']:.3f}",
            "Macro-F1": f"{f1:.3f}",
        })

    df = pd.DataFrame(rows)
    print("\n=== Model performance summary ===")
    print(df.to_string(index=False))
    print()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    warnings.filterwarnings("ignore")

    # 1. Build feature matrix
    print("Building feature matrix …")
    X, y_event, y_source, feature_names = build_feature_matrix(
        DATASET_DIR, dop_thresh=0.2, t_start=0, t_end=60
    )
    print(f"  {len(X)} files × {len(feature_names)} features extracted.")

    # 2. Dataset summary
    print_dataset_summary(y_event, y_source)

    # 3. Save feature matrix
    fm_path = os.path.join(OUTPUT_DIR, "feature_matrix.csv")
    fm_df = X.copy()
    fm_df.insert(0, "event", y_event.values)
    fm_df.insert(0, "source", y_source.values)
    fm_df.to_csv(fm_path, index=False)
    print(f"Feature matrix saved to {fm_path}")

    # 4. Run ML models and cross-validation
    print("\nRunning ML models and cross-validation …")
    results = run_full_pipeline(X, y_event, y_source, OUTPUT_DIR)

    # 5. Print summary table
    print_summary_table(results)

    # 6. Generate and save all plots
    generate_all_plots(X, y_event, y_source, results, feature_names,
                       DATASET_DIR, OUTPUT_DIR)

    print(f"\nDone. All outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
