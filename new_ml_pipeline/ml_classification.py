"""
ml_classification.py  —  new_ml_pipeline
=========================================
ML pipeline: feature matrix construction, preprocessing, global and
per-source classification with regularised classifiers.

Classifiers (all regularised to prevent overfitting):
* Random Forest  — max_depth=10, min_samples_leaf=5
* SVM (RBF)      — C=5 (reduced from 10)
* XGBoost        — max_depth=6, learning_rate=0.05, subsample=0.8

Cross-validation strategies:
* Stratified 5-fold CV — for standard model assessment
* Leave-One-Source-Out (LOSO) — global generalisation test
* Leave-One-Date-Out (LODO)   — within-source generalisation test
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCES = ["SP-AGIL", "SP-PURE", "DPQAM16-200G", "DPQPSK-200G", "10GE"]
EVENTS  = ["NE", "FS", "VB", "MB", "TAP"]

# All 41 features: 34 TD + 9 FD — computed unconditionally for every window
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
    # TD — θ_ref features (6)
    "theta_mean", "theta_std", "theta_max", "theta_rms",
    "range_theta_ref", "p95_theta_ref",
    # TD — raw Stokes trajectory variability (6)
    "s1_std", "s2_std", "s3_std", "s1_range", "s2_range", "s3_range",
    # FD — PSD features (9)
    "psd_peak_freq", "psd_peak_power",
    "bp_low", "bp_mid", "bp_high",
    "bp_ratio_mid_low", "psd_peak_sharpness",
    "spectral_entropy", "vb_snr_80hz_db",
    # Modulation flag — input feature (not a gate)
    "is_modulated",
]

# ---------------------------------------------------------------------------
# Classifier factories — regularised to prevent overfitting
# ---------------------------------------------------------------------------

def _rf_factory() -> Pipeline:
    """Regularised Random Forest — max_depth=10 prevents full memorisation."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     RandomForestClassifier(
            n_estimators=200,
            max_depth=10,        # prevents full memorisation
            min_samples_leaf=5,  # smooths decision boundaries
            random_state=42,
        )),
    ])


def _svm_factory() -> Pipeline:
    """Regularised SVM — C=5 (reduced from 10) to prevent overfitting."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     SVC(
            kernel="rbf",
            C=5,            # reduced from 10
            gamma="scale",
            probability=True,
            random_state=42,
        )),
    ])


def _xgb_factory(label_encoder: LabelEncoder | None = None) -> Pipeline:
    """Regularised XGBoost — subsample + colsample_bytree prevent overfitting."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )),
    ])


CLASSIFIERS = {
    "RF":      _rf_factory,
    "SVM":     _svm_factory,
    "XGBoost": _xgb_factory,
}

# ---------------------------------------------------------------------------
# Preprocessing helper
# ---------------------------------------------------------------------------

def _scale(X_arr: np.ndarray) -> np.ndarray:
    imp    = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    return scaler.fit_transform(imp.fit_transform(X_arr))


# ---------------------------------------------------------------------------
# Unsupervised: K-Means
# ---------------------------------------------------------------------------

def run_kmeans(X_scaled: np.ndarray, y_event: pd.Series, n_clusters: int = 5):
    km             = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    cluster_labels = km.fit_predict(X_scaled)

    label_map = {}
    for c in range(n_clusters):
        mask = cluster_labels == c
        if mask.sum() == 0:
            label_map[c] = "?"
            continue
        vals, counts = np.unique(y_event.values[mask], return_counts=True)
        label_map[c] = vals[np.argmax(counts)]

    predicted = np.array([label_map[c] for c in cluster_labels])
    purity    = float(np.mean(predicted == y_event.values))
    ari       = float(adjusted_rand_score(y_event.values, cluster_labels))
    return km, cluster_labels, label_map, purity, ari


def run_pca_kmeans(X_scaled: np.ndarray, y_event: pd.Series, n_components: int = 2):
    pca            = PCA(n_components=n_components, random_state=42)
    X_pca          = pca.fit_transform(X_scaled)
    km             = KMeans(n_clusters=5, random_state=42, n_init=20)
    cluster_labels = km.fit_predict(X_pca)
    ari            = float(adjusted_rand_score(y_event.values, cluster_labels))
    return pca, km, X_pca, cluster_labels, ari


# ---------------------------------------------------------------------------
# Cross-validation helpers
# ---------------------------------------------------------------------------

def _encode_labels(y_arr: np.ndarray, le: LabelEncoder) -> np.ndarray:
    return le.transform(y_arr)


def stratified_cv(clf_name: str, X_arr: np.ndarray, y_arr: np.ndarray,
                  n_splits: int = 5) -> dict:
    """Stratified k-fold CV.  Returns dict with mean_acc, std_acc, cm, report."""
    le  = LabelEncoder().fit(y_arr)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_true: list = []
    all_pred: list = []
    accs: list[float] = []

    for train_idx, test_idx in skf.split(X_arr, y_arr):
        factory = CLASSIFIERS[clf_name]
        pipe    = factory()

        if clf_name == "XGBoost":
            pipe.fit(X_arr[train_idx], le.transform(y_arr[train_idx]))
            raw_pred = pipe.predict(X_arr[test_idx])
            preds    = le.inverse_transform(raw_pred.astype(int))
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
        "mean_acc": mean_acc,
        "std_acc":  std_acc,
        "macro_f1": macro_f1,
        "report":   report,
        "cm":       cm,
        "all_true": np.array(all_true),
        "all_pred": np.array(all_pred),
    }


def loso_cv(clf_name: str, X_arr: np.ndarray, y_arr: np.ndarray,
            source_arr: np.ndarray) -> dict:
    """Leave-One-Source-Out CV — global generalisation test."""
    le      = LabelEncoder().fit(y_arr)
    sources = list(dict.fromkeys(source_arr))

    all_true: list = []
    all_pred: list = []
    per_source_acc: dict[str, float] = {}

    for src in sources:
        test_mask  = source_arr == src
        train_mask = ~test_mask
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        factory = CLASSIFIERS[clf_name]
        pipe    = factory()

        if clf_name == "XGBoost":
            pipe.fit(X_arr[train_mask], le.transform(y_arr[train_mask]))
            raw_pred = pipe.predict(X_arr[test_mask])
            preds    = le.inverse_transform(raw_pred.astype(int))
        else:
            pipe.fit(X_arr[train_mask], y_arr[train_mask])
            preds = pipe.predict(X_arr[test_mask])

        per_source_acc[src] = float(accuracy_score(y_arr[test_mask], preds))
        all_true.extend(y_arr[test_mask])
        all_pred.extend(preds)

    overall_acc = float(accuracy_score(all_true, all_pred))
    macro_f1    = float(f1_score(all_true, all_pred,
                                  labels=EVENTS, average="macro", zero_division=0))
    cm          = confusion_matrix(all_true, all_pred, labels=EVENTS)

    return {
        "per_source_acc": per_source_acc,
        "overall_acc":    overall_acc,
        "macro_f1":       macro_f1,
        "cm":             cm,
        "all_true":       np.array(all_true),
        "all_pred":       np.array(all_pred),
    }


def lodo_cv(clf_name: str, X_arr: np.ndarray, y_arr: np.ndarray,
            date_arr: np.ndarray) -> dict:
    """Leave-One-Date-Out CV — within-source generalisation test.

    Groups by the ``date`` column; leaves one recording date out at a time.
    """
    le    = LabelEncoder().fit(y_arr)
    dates = list(dict.fromkeys(date_arr))

    all_true: list = []
    all_pred: list = []
    per_date_acc: dict[str, float] = {}

    for d in dates:
        test_mask  = date_arr == d
        train_mask = ~test_mask
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        # Need at least 2 classes in training set
        if len(np.unique(y_arr[train_mask])) < 2:
            continue

        factory = CLASSIFIERS[clf_name]
        pipe    = factory()

        if clf_name == "XGBoost":
            y_train_enc = le.transform(y_arr[train_mask])
            pipe.fit(X_arr[train_mask], y_train_enc)
            raw_pred = pipe.predict(X_arr[test_mask])
            preds    = le.inverse_transform(raw_pred.astype(int))
        else:
            pipe.fit(X_arr[train_mask], y_arr[train_mask])
            preds = pipe.predict(X_arr[test_mask])

        per_date_acc[d] = float(accuracy_score(y_arr[test_mask], preds))
        all_true.extend(y_arr[test_mask])
        all_pred.extend(preds)

    if not all_true:
        return {
            "per_date_acc": {},
            "overall_acc":  0.0,
            "macro_f1":     0.0,
            "cm":           np.zeros((len(EVENTS), len(EVENTS)), dtype=int),
            "all_true":     np.array([]),
            "all_pred":     np.array([]),
        }

    overall_acc = float(accuracy_score(all_true, all_pred))
    macro_f1    = float(f1_score(all_true, all_pred,
                                  labels=EVENTS, average="macro", zero_division=0))
    cm          = confusion_matrix(all_true, all_pred, labels=EVENTS)

    return {
        "per_date_acc": per_date_acc,
        "overall_acc":  overall_acc,
        "macro_f1":     macro_f1,
        "cm":           cm,
        "all_true":     np.array(all_true),
        "all_pred":     np.array(all_pred),
    }


# ---------------------------------------------------------------------------
# Global pipeline (all sources combined)
# ---------------------------------------------------------------------------

def run_full_pipeline(X: pd.DataFrame, y_event: pd.Series, y_source: pd.Series,
                      output_dir: str) -> dict:
    """Run global classification: stratified 5-fold + LOSO for RF, SVM, XGBoost."""
    os.makedirs(output_dir, exist_ok=True)

    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    X_arr   = X.values.astype(float)
    X_imp   = imputer.fit_transform(X_arr)
    X_scaled = scaler.fit_transform(X_imp)

    y_ev  = y_event.values
    y_src = y_source.values

    results: dict = {}

    # Unsupervised
    km, cluster_labels, label_map, purity, ari_km = run_kmeans(X_scaled, y_event)
    results["kmeans"] = {
        "model": km, "cluster_labels": cluster_labels,
        "label_map": label_map, "purity": purity, "ari": ari_km,
    }
    print(f"[K-Means] Purity={purity:.3f}  ARI={ari_km:.3f}")

    pca2, km2, X_pca2, cl2, ari2 = run_pca_kmeans(X_scaled, y_event, n_components=2)
    pca3, km3, X_pca3, cl3, ari3 = run_pca_kmeans(X_scaled, y_event, n_components=3)
    results["pca2"] = {"pca": pca2, "km": km2, "X_pca": X_pca2, "cluster_labels": cl2, "ari": ari2}
    results["pca3"] = {"pca": pca3, "km": km3, "X_pca": X_pca3, "cluster_labels": cl3, "ari": ari3}
    print(f"[PCA2+KM] ARI={ari2:.3f}   [PCA3+KM] ARI={ari3:.3f}")

    # Stratified 5-fold CV
    for clf_name in ("RF", "SVM", "XGBoost"):
        print(f"Running stratified 5-fold CV for {clf_name} …")
        cv_result = stratified_cv(clf_name, X_arr, y_ev)
        results[f"cv_{clf_name}"] = cv_result
        print(f"  {clf_name} acc={cv_result['mean_acc']:.3f} ± {cv_result['std_acc']:.3f}  "
              f"macro-F1={cv_result['macro_f1']:.3f}")

    # LOSO CV
    for clf_name in ("RF", "SVM", "XGBoost"):
        print(f"Running LOSO CV for {clf_name} …")
        loso_result = loso_cv(clf_name, X_arr, y_ev, y_src)
        results[f"loso_{clf_name}"] = loso_result
        print(f"  {clf_name} LOSO acc={loso_result['overall_acc']:.3f}  "
              f"macro-F1={loso_result['macro_f1']:.3f}")
        print(f"      per-source: {loso_result['per_source_acc']}")

    # Save classification reports and final models
    le = LabelEncoder().fit(y_ev)
    for clf_name in ("RF", "SVM", "XGBoost"):
        factory = CLASSIFIERS[clf_name]
        pipe    = factory()
        if clf_name == "XGBoost":
            pipe.fit(X_arr, le.transform(y_ev))
        else:
            pipe.fit(X_arr, y_ev)

        if clf_name == "XGBoost":
            raw_preds = pipe.predict(X_arr)
            preds     = le.inverse_transform(raw_preds.astype(int))
        else:
            preds = pipe.predict(X_arr)

        rep = classification_report(y_ev, preds, labels=EVENTS, zero_division=0)
        with open(os.path.join(output_dir, f"classification_report_{clf_name}.txt"), "w") as fh:
            fh.write(f"{clf_name} — full dataset classification report\n\n{rep}")

        joblib.dump(pipe, os.path.join(output_dir, f"{clf_name.lower()}_model.pkl"))
        results[f"final_{clf_name}"] = pipe

    results["X_scaled"] = X_scaled
    results["X_pca2"]   = X_pca2
    results["X_pca3"]   = X_pca3
    results["imputer"]  = imputer
    results["scaler"]   = scaler

    return results


# ---------------------------------------------------------------------------
# Per-source pipeline (Requirement 4)
# ---------------------------------------------------------------------------

def run_per_source_pipeline(X: pd.DataFrame, y_event: pd.Series,
                             y_source: pd.Series, date_series: pd.Series,
                             output_dir: str) -> dict:
    """For each of the 5 sources, run stratified 5-fold + LODO CV for RF, SVM, XGBoost.

    Returns nested dict: per_source_results[source][clf_name][cv_strategy].
    """
    per_source_dir = os.path.join(output_dir, "per_source")
    os.makedirs(per_source_dir, exist_ok=True)

    per_source_results: dict = {}
    X_arr = X.values.astype(float)

    for src in SOURCES:
        mask = y_source.values == src
        if mask.sum() == 0:
            print(f"  [SKIP] No data for source {src}")
            continue

        X_src    = X_arr[mask]
        y_src_ev = y_event.values[mask]
        d_src    = date_series.values[mask]

        # Skip if fewer than 2 event classes
        unique_events = np.unique(y_src_ev)
        if len(unique_events) < 2:
            print(f"  [SKIP] {src}: fewer than 2 event classes, skipping")
            continue

        print(f"\n--- Per-source classification: {src} ({mask.sum()} windows) ---")
        per_source_results[src] = {}
        summary_rows = []

        for clf_name in ("RF", "SVM", "XGBoost"):
            per_source_results[src][clf_name] = {}

            # Stratified 5-fold CV
            n_splits = min(5, int(np.min(np.bincount(
                np.unique(y_src_ev, return_inverse=True)[1]
            ))))
            n_splits = max(2, n_splits)

            cv_res = stratified_cv(clf_name, X_src, y_src_ev, n_splits=n_splits)
            per_source_results[src][clf_name]["strat_cv"] = cv_res

            # Save confusion matrix
            from visualisation import plot_confusion_matrix  # local import to avoid circular
            cm_path = os.path.join(per_source_dir,
                                   f"confusion_matrix_{src}_{clf_name}.png")
            plot_confusion_matrix(
                cv_res["cm"], EVENTS,
                f"CM — {src} / {clf_name} (stratified 5-fold)",
                cm_path,
            )

            # Save classification report
            rep_path = os.path.join(per_source_dir,
                                    f"report_{src}_{clf_name}.txt")
            with open(rep_path, "w") as fh:
                fh.write(f"Source: {src}  Model: {clf_name}  CV: stratified 5-fold\n\n")
                fh.write(cv_res["report"])

            # LODO CV
            lodo_res = lodo_cv(clf_name, X_src, y_src_ev, d_src)
            per_source_results[src][clf_name]["lodo"] = lodo_res

            summary_rows.append({
                "Model":          clf_name,
                "CV strategy":    "Stratified 5-fold",
                "Accuracy":       f"{cv_res['mean_acc']:.3f} ± {cv_res['std_acc']:.3f}",
                "Macro-F1":       f"{cv_res['macro_f1']:.3f}",
            })
            summary_rows.append({
                "Model":          clf_name,
                "CV strategy":    "Leave-One-Date-Out",
                "Accuracy":       f"{lodo_res['overall_acc']:.3f}",
                "Macro-F1":       f"{lodo_res['macro_f1']:.3f}",
            })

            print(f"  {clf_name}  strat-CV acc={cv_res['mean_acc']:.3f}±{cv_res['std_acc']:.3f} "
                  f"macro-F1={cv_res['macro_f1']:.3f}   "
                  f"LODO acc={lodo_res['overall_acc']:.3f} "
                  f"macro-F1={lodo_res['macro_f1']:.3f}")

        # Print per-source summary table
        df_sum = pd.DataFrame(summary_rows)
        print(f"\n  Summary for {src}:")
        print(df_sum.to_string(index=False))

    return per_source_results
