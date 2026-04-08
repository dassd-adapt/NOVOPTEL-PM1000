"""
ml_classification.py  —  new_ml_pipeline_2
===========================================
Per-source LOSO-only ML pipeline.  No global (all-sources-combined) pipeline.

Design:
* For each of the 5 sources:
  - LOSO CV: train on 4 sources, test on this source (RF, SVM, XGBoost)
  - Stratified 5-fold CV within source (RF, SVM, XGBoost) — comparison baseline
* Learning curves per source: vary number of training sources 1 → 4

Classifiers (same regularisation as new_ml_pipeline/):
  RF      — n_estimators=200, max_depth=10, min_samples_leaf=5
  SVM     — RBF, C=5, gamma=scale
  XGBoost — n_estimators=200, max_depth=6, lr=0.05, subsample=0.8
"""

from __future__ import annotations

import itertools
import os
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from xgboost import XGBClassifier

# Import FFT constants from feature_extraction in same directory
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extraction import FFT_BIN_COLS, FFT_N_BINS  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCES = ["SP-AGIL", "SP-PURE", "DPQAM16-200G", "DPQPSK-200G", "10GE"]
EVENTS  = ["NE", "FS", "VB", "MB", "TAP"]

# Full feature set: 34 TD + 9 Welch + 1001 FFT bins + 1 is_modulated = 1045
FEATURE_COLS: list[str] = [
    # TD — DOP (4)
    "dop_mean", "dop_std", "dop_min", "dop_max",
    # TD — DOP spread / modulation (3)
    "var_dop", "iqr_dop", "frac_dop_low",
    # TD — step-angle statistics (6)
    "step_mean", "step_std", "step_max", "step_rms", "step_p95", "step_p99",
    # TD — higher-order step statistics (2)
    "kurtosis_step", "skew_step",
    # TD — burst / arc / autocorr (3)
    "burst_count", "cum_arc", "step_autocorr_lag1",
    # TD — theta_ref features (6)
    "theta_mean", "theta_std", "theta_max", "theta_rms",
    "range_theta_ref", "p95_theta_ref",
    # TD — raw Stokes trajectory variability (6)
    "s1_std", "s2_std", "s3_std", "s1_range", "s2_range", "s3_range",
    # FD — Welch PSD scalars (9)
    "psd_peak_freq", "psd_peak_power",
    "bp_low", "bp_mid", "bp_high",
    "bp_ratio_mid_low", "psd_peak_sharpness",
    "spectral_entropy", "vb_snr_80hz_db",
    # Modulation flag (1)
    "is_modulated",
    # FD — raw FFT magnitude bins (1001)
    *FFT_BIN_COLS,
]

# ---------------------------------------------------------------------------
# Classifier factories
# ---------------------------------------------------------------------------

def _rf_factory() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42,
        )),
    ])


def _svm_factory() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     SVC(kernel="rbf", C=5, gamma="scale",
                        probability=True, random_state=42)),
    ])


def _xgb_factory() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=42, verbosity=0,
        )),
    ])


CLASSIFIERS = {
    "RF":      _rf_factory,
    "SVM":     _svm_factory,
    "XGBoost": _xgb_factory,
}

# ---------------------------------------------------------------------------
# Cross-validation helpers
# ---------------------------------------------------------------------------

def stratified_cv(clf_name: str, X_arr: np.ndarray, y_arr: np.ndarray,
                  n_splits: int = 5) -> dict:
    """Stratified k-fold CV within a single source."""
    le  = LabelEncoder().fit(y_arr)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_true: list = []
    all_pred: list = []
    accs: list[float] = []

    for train_idx, test_idx in skf.split(X_arr, y_arr):
        pipe = CLASSIFIERS[clf_name]()
        if clf_name == "XGBoost":
            pipe.fit(X_arr[train_idx], le.transform(y_arr[train_idx]))
            preds = le.inverse_transform(
                pipe.predict(X_arr[test_idx]).astype(int))
        else:
            pipe.fit(X_arr[train_idx], y_arr[train_idx])
            preds = pipe.predict(X_arr[test_idx])

        accs.append(accuracy_score(y_arr[test_idx], preds))
        all_true.extend(y_arr[test_idx])
        all_pred.extend(preds)

    mean_acc = float(np.mean(accs))
    std_acc  = float(np.std(accs))
    report   = classification_report(all_true, all_pred,
                                      labels=EVENTS, zero_division=0)
    cm       = confusion_matrix(all_true, all_pred, labels=EVENTS)
    macro_f1 = float(f1_score(all_true, all_pred,
                               labels=EVENTS, average="macro", zero_division=0))

    return {
        "mean_acc": mean_acc, "std_acc": std_acc, "macro_f1": macro_f1,
        "report": report, "cm": cm,
        "all_true": np.array(all_true), "all_pred": np.array(all_pred),
    }


# ---------------------------------------------------------------------------
# LOSO learning curve
# ---------------------------------------------------------------------------

def loso_learning_curve(
    clf_name: str,
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    source_arr: np.ndarray,
    test_source: str,
) -> list[tuple[int, float, float]]:
    """For test_source, vary number of training sources from 1 to 4.

    For each n_train_sources, enumerate all C(4, n) combinations of the other
    4 sources, train the model on each, test on test_source, and average acc.

    Returns list of (n_sources, mean_acc, std_acc).
    """
    le            = LabelEncoder().fit(y_arr)
    other_sources = [s for s in SOURCES if s != test_source]
    test_mask     = source_arr == test_source
    results: list[tuple[int, float, float]] = []

    for n in range(1, len(other_sources) + 1):
        accs = []
        for combo in itertools.combinations(other_sources, n):
            train_mask = np.isin(source_arr, list(combo))

            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue
            if len(np.unique(y_arr[train_mask])) < 2:
                continue

            pipe = CLASSIFIERS[clf_name]()
            try:
                if clf_name == "XGBoost":
                    pipe.fit(X_arr[train_mask], le.transform(y_arr[train_mask]))
                    preds = le.inverse_transform(
                        pipe.predict(X_arr[test_mask]).astype(int))
                else:
                    pipe.fit(X_arr[train_mask], y_arr[train_mask])
                    preds = pipe.predict(X_arr[test_mask])
                accs.append(accuracy_score(y_arr[test_mask], preds))
            except Exception:
                continue

        if accs:
            results.append((n, float(np.mean(accs)), float(np.std(accs))))

    return results


# ---------------------------------------------------------------------------
# Per-source LOSO analysis
# ---------------------------------------------------------------------------

def run_per_source_loso(
    X: pd.DataFrame,
    y_event: pd.Series,
    y_source: pd.Series,
    output_dir: str,
) -> dict:
    """For each of the 5 sources, run LOSO CV and stratified 5-fold CV.

    Returns nested dict:
      results[source][clf_name] = {
          "loso":    {accuracy, macro_f1, cm, report, all_true, all_pred},
          "strat_cv":{mean_acc, std_acc, macro_f1, cm, report, ...},
      }
    """
    per_source_dir = os.path.join(output_dir, "per_source")
    os.makedirs(per_source_dir, exist_ok=True)

    X_arr   = X.values.astype(float)
    y_ev    = y_event.values
    y_src   = y_source.values
    results: dict = {}

    for test_source in SOURCES:
        test_mask  = y_src == test_source
        train_mask = ~test_mask

        if test_mask.sum() == 0:
            print(f"  [SKIP] No test data for source {test_source}")
            continue

        print(f"\n--- LOSO: test source = {test_source} "
              f"({test_mask.sum()} test windows, {train_mask.sum()} train windows) ---")
        results[test_source] = {}

        X_train, y_train = X_arr[train_mask], y_ev[train_mask]
        X_test,  y_test  = X_arr[test_mask],  y_ev[test_mask]

        # Stratified 5-fold CV within this source
        n_splits = max(2, min(5, int(np.min(np.bincount(
            np.unique(y_ev[test_mask], return_inverse=True)[1])))))

        for clf_name in ("RF", "SVM", "XGBoost"):
            results[test_source][clf_name] = {}

            # --- LOSO ---
            le   = LabelEncoder().fit(y_ev)
            pipe = CLASSIFIERS[clf_name]()
            try:
                if clf_name == "XGBoost":
                    pipe.fit(X_train, le.transform(y_train))
                    preds = le.inverse_transform(
                        pipe.predict(X_test).astype(int))
                else:
                    pipe.fit(X_train, y_train)
                    preds = pipe.predict(X_test)

                loso_acc     = float(accuracy_score(y_test, preds))
                loso_macro_f1 = float(f1_score(y_test, preds, labels=EVENTS,
                                                average="macro", zero_division=0))
                loso_cm      = confusion_matrix(y_test, preds, labels=EVENTS)
                loso_report  = classification_report(y_test, preds,
                                                      labels=EVENTS, zero_division=0)
            except Exception as exc:
                print(f"  [ERROR] {clf_name} LOSO: {exc}")
                loso_acc = loso_macro_f1 = 0.0
                loso_cm  = np.zeros((len(EVENTS), len(EVENTS)), dtype=int)
                loso_report = ""
                preds = np.array([])

            results[test_source][clf_name]["loso"] = {
                "accuracy":  loso_acc,
                "macro_f1":  loso_macro_f1,
                "cm":        loso_cm,
                "report":    loso_report,
                "all_true":  y_test,
                "all_pred":  preds,
            }

            # Save LOSO CM and report
            safe_src = test_source.replace("/", "_").replace("-", "_")
            cm_path  = os.path.join(per_source_dir,
                                    f"cm_loso_{safe_src}_{clf_name}.png")
            rep_path = os.path.join(per_source_dir,
                                    f"report_loso_{safe_src}_{clf_name}.txt")

            try:
                from visualisation import plot_confusion_matrix
                plot_confusion_matrix(
                    loso_cm, EVENTS,
                    f"LOSO CM — {test_source} / {clf_name}\n"
                    f"(train on 4 sources, test on this one)",
                    cm_path,
                )
            except Exception:
                pass

            if loso_report:
                with open(rep_path, "w") as fh:
                    fh.write(f"Source: {test_source}  Model: {clf_name}  CV: LOSO\n\n")
                    fh.write(loso_report)

            # --- Stratified 5-fold within source ---
            try:
                cv_res = stratified_cv(clf_name, X_test, y_test,
                                       n_splits=n_splits)
            except Exception as exc:
                print(f"  [WARN] {clf_name} strat-CV for {test_source}: {exc}")
                cv_res = {
                    "mean_acc": 0.0, "std_acc": 0.0, "macro_f1": 0.0,
                    "report": "", "cm": np.zeros((len(EVENTS), len(EVENTS)), dtype=int),
                    "all_true": np.array([]), "all_pred": np.array([]),
                }

            results[test_source][clf_name]["strat_cv"] = cv_res

            print(f"  {clf_name:8s}  LOSO acc={loso_acc:.3f}  "
                  f"macro-F1={loso_macro_f1:.3f}  |  "
                  f"5-fold acc={cv_res['mean_acc']:.3f}±{cv_res['std_acc']:.3f}")

    return results


# ---------------------------------------------------------------------------
# Summary table printer
# ---------------------------------------------------------------------------

def print_loso_summary_table(results: dict) -> None:
    """Print per-source LOSO accuracy table."""
    header = (f"{'Source':<20}"
              f"{'RF (LOSO)':>12}{'SVM (LOSO)':>12}{'XGB (LOSO)':>12}"
              f"{'RF (5-fold)':>14}{'SVM (5-fold)':>14}{'XGB (5-fold)':>14}")
    print("\n=== Per-source LOSO accuracy (train on 4 sources, test on 1) ===")
    print(header)
    print("-" * len(header))

    for src in SOURCES:
        if src not in results:
            continue
        src_res = results[src]
        row = f"{src:<20}"
        for clf_name in ("RF", "SVM", "XGBoost"):
            loso_acc = src_res.get(clf_name, {}).get("loso", {}).get("accuracy", float("nan"))
            row += f"{loso_acc:>12.3f}"
        for clf_name in ("RF", "SVM", "XGBoost"):
            cv_acc = src_res.get(clf_name, {}).get("strat_cv", {}).get("mean_acc", float("nan"))
            row += f"{cv_acc:>14.3f}"
        print(row)
    print()
