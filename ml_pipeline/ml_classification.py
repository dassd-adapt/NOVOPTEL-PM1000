"""
ml_classification.py
====================
ML pipeline: feature matrix construction, preprocessing, unsupervised and
supervised models, cross-validation (stratified 5-fold + LOSO), and model saving.
"""

import os
import glob
import pickle
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, adjusted_rand_score, classification_report,
    confusion_matrix,
)

from feature_extraction import (
    parse_filename,
    load_csv,
    compute_source_baseline,
    extract_features,
)

SOURCES = ["SP-AGIL", "SP-PURE", "DPQAM16-200G", "DPQPSK-200G", "10GE"]
EVENTS = ["NE", "FS", "VB", "MB", "TAP"]


# ---------------------------------------------------------------------------
# 1. Build feature matrix
# ---------------------------------------------------------------------------

def _scan_csv_files(dataset_dir):
    """Return list of (filepath, meta_dict) for all valid PM1000 CSV files,
    scanning dataset_dir and its parent directory, deduplicating by basename."""
    found = {}
    search_dirs = [dataset_dir, os.path.dirname(dataset_dir)]
    for d in search_dirs:
        for fp in glob.glob(os.path.join(d, "pm1000_sop_*.csv")):
            bn = os.path.basename(fp)
            meta = parse_filename(bn)
            if meta is not None:
                found[bn] = (fp, meta)
    return list(found.values())


def build_feature_matrix(dataset_dir, dop_thresh=0.2, t_start=0, t_end=60):
    """Scan dataset_dir, compute per-source baselines, extract features for
    all files.

    Returns
    -------
    X : pd.DataFrame  — feature matrix (rows = files, cols = features)
    y_event : pd.Series — event labels
    y_source : pd.Series — source labels
    feature_names : list of str
    """
    all_files = _scan_csv_files(dataset_dir)

    # Group NE files by source
    ne_files_by_source = {s: [] for s in SOURCES}
    for fp, meta in all_files:
        if meta["event"].upper() == "NE":
            src = meta["source"].upper()
            if src in ne_files_by_source:
                ne_files_by_source[src].append(fp)

    # Compute per-source baselines and representative fs
    baselines = {}
    source_fs = {}
    for src in SOURCES:
        ne_fps = ne_files_by_source[src]
        if ne_fps:
            sref, floor_deg, mad_val = compute_source_baseline(
                ne_fps, dop_thresh=dop_thresh, t_start=t_start, t_end=t_end
            )
            # Estimate fs from first NE file
            _, fs_est = load_csv(ne_fps[0], t_start=t_start, t_end=t_end)
        else:
            sref = np.array([1.0, 0.0, 0.0])
            floor_deg = 0.0
            mad_val = 1.0
            fs_est = 1441.0
            warnings.warn(f"No NE files found for source {src}, using defaults.")
        baselines[src] = (sref, floor_deg, mad_val)
        source_fs[src] = fs_est

    # Extract features for every file
    rows = []
    labels_event = []
    labels_source = []

    for fp, meta in all_files:
        src = meta["source"].upper()
        event = meta["event"].upper()
        if src not in baselines:
            continue
        sref, floor_deg, mad_val = baselines[src]
        fs = source_fs[src]
        try:
            feats = extract_features(
                fp, sref, floor_deg, mad_val, fs,
                dop_thresh=dop_thresh, t_start=t_start, t_end=t_end,
            )
        except Exception as e:
            warnings.warn(f"Failed to extract features from {fp}: {e}")
            continue
        feats["_filepath"] = fp
        rows.append(feats)
        labels_event.append(event)
        labels_source.append(src)

    df_all = pd.DataFrame(rows)
    df_all = df_all.drop(columns=["_filepath"], errors="ignore")

    feature_names = [c for c in df_all.columns]
    X = df_all[feature_names]
    y_event = pd.Series(labels_event, name="event")
    y_source = pd.Series(labels_source, name="source")

    return X, y_event, y_source, feature_names


# ---------------------------------------------------------------------------
# 2. Preprocessing helper
# ---------------------------------------------------------------------------

def make_pipeline(clf):
    """Wrap classifier with median imputer + standard scaler."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


# ---------------------------------------------------------------------------
# 3. Unsupervised: K-Means
# ---------------------------------------------------------------------------

def run_kmeans(X_scaled, y_event, n_clusters=5):
    """Fit K-Means, map clusters to most frequent true label, report metrics."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    cluster_labels = km.fit_predict(X_scaled)

    # Map cluster id -> most frequent event label
    label_map = {}
    for c in range(n_clusters):
        mask = cluster_labels == c
        if mask.sum() == 0:
            label_map[c] = "?"
            continue
        vals, counts = np.unique(y_event[mask], return_counts=True)
        label_map[c] = vals[np.argmax(counts)]

    predicted = np.array([label_map[c] for c in cluster_labels])

    # Purity: fraction of samples assigned to their correct cluster majority
    purity = float(np.mean(predicted == y_event.values))
    ari = float(adjusted_rand_score(y_event.values, cluster_labels))

    return km, cluster_labels, label_map, purity, ari


def run_pca_kmeans(X_scaled, y_event, n_components=2):
    """PCA reduction then K-Means."""
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    km = KMeans(n_clusters=5, random_state=42, n_init=20)
    cluster_labels = km.fit_predict(X_pca)
    ari = float(adjusted_rand_score(y_event.values, cluster_labels))
    return pca, km, X_pca, cluster_labels, ari


# ---------------------------------------------------------------------------
# 4. Cross-validation helpers
# ---------------------------------------------------------------------------

def stratified_cv(clf_factory, X_arr, y_arr, n_splits=5):
    """Stratified k-fold CV.  Returns dict with mean_acc, std_acc, report."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_true = []
    all_pred = []
    accs = []

    for train_idx, test_idx in skf.split(X_arr, y_arr):
        pipe = clf_factory()
        pipe.fit(X_arr[train_idx], y_arr[train_idx])
        preds = pipe.predict(X_arr[test_idx])
        accs.append(accuracy_score(y_arr[test_idx], preds))
        all_true.extend(y_arr[test_idx])
        all_pred.extend(preds)

    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs))
    report = classification_report(all_true, all_pred,
                                   labels=EVENTS, zero_division=0)
    cm = confusion_matrix(all_true, all_pred, labels=EVENTS)
    return {
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "report": report,
        "cm": cm,
        "all_true": np.array(all_true),
        "all_pred": np.array(all_pred),
    }


def loso_cv(clf_factory, X_arr, y_arr, source_arr):
    """Leave-One-Source-Out CV.

    For each source, train on the other 4 sources, test on the left-out one.
    """
    sources = list(dict.fromkeys(source_arr))  # preserve order, unique
    all_true = []
    all_pred = []
    per_source_acc = {}

    for src in sources:
        test_mask = source_arr == src
        train_mask = ~test_mask

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        pipe = clf_factory()
        pipe.fit(X_arr[train_mask], y_arr[train_mask])
        preds = pipe.predict(X_arr[test_mask])
        per_source_acc[src] = float(accuracy_score(y_arr[test_mask], preds))
        all_true.extend(y_arr[test_mask])
        all_pred.extend(preds)

    overall_acc = float(accuracy_score(all_true, all_pred))
    cm = confusion_matrix(all_true, all_pred, labels=EVENTS)
    return {
        "per_source_acc": per_source_acc,
        "overall_acc": overall_acc,
        "cm": cm,
        "all_true": np.array(all_true),
        "all_pred": np.array(all_pred),
    }


# ---------------------------------------------------------------------------
# 5. Full pipeline runner
# ---------------------------------------------------------------------------

def run_full_pipeline(X, y_event, y_source, output_dir):
    """Run all models and CV, save outputs.  Returns results dict."""
    os.makedirs(output_dir, exist_ok=True)

    # ---- Preprocessing (for unsupervised; supervised uses Pipeline) ----
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_arr = X.values.astype(float)
    X_imp = imputer.fit_transform(X_arr)
    X_scaled = scaler.fit_transform(X_imp)

    y_ev = y_event.values
    y_src = y_source.values

    results = {}

    # ---- Unsupervised ----
    km, cluster_labels, label_map, purity, ari_km = run_kmeans(X_scaled, y_event)
    results["kmeans"] = {
        "model": km,
        "cluster_labels": cluster_labels,
        "label_map": label_map,
        "purity": purity,
        "ari": ari_km,
    }
    print(f"[K-Means] Purity={purity:.3f}  ARI={ari_km:.3f}")

    pca2, km2, X_pca2, cl2, ari2 = run_pca_kmeans(X_scaled, y_event, n_components=2)
    pca3, km3, X_pca3, cl3, ari3 = run_pca_kmeans(X_scaled, y_event, n_components=3)
    results["pca2"] = {"pca": pca2, "km": km2, "X_pca": X_pca2,
                       "cluster_labels": cl2, "ari": ari2}
    results["pca3"] = {"pca": pca3, "km": km3, "X_pca": X_pca3,
                       "cluster_labels": cl3, "ari": ari3}
    print(f"[PCA2+KM] ARI={ari2:.3f}   [PCA3+KM] ARI={ari3:.3f}")

    # ---- Supervised factories ----
    def rf_factory():
        return make_pipeline(
            RandomForestClassifier(n_estimators=200, random_state=42)
        )

    def svm_factory():
        return make_pipeline(
            SVC(kernel="rbf", C=10, gamma="scale",
                probability=True, random_state=42)
        )

    # ---- Stratified 5-fold CV ----
    print("Running stratified 5-fold CV for RF …")
    cv_rf = stratified_cv(rf_factory, X_arr, y_ev)
    print(f"  RF  acc={cv_rf['mean_acc']:.3f} ± {cv_rf['std_acc']:.3f}")

    print("Running stratified 5-fold CV for SVM …")
    cv_svm = stratified_cv(svm_factory, X_arr, y_ev)
    print(f"  SVM acc={cv_svm['mean_acc']:.3f} ± {cv_svm['std_acc']:.3f}")

    results["cv_rf"] = cv_rf
    results["cv_svm"] = cv_svm

    # ---- LOSO CV ----
    print("Running LOSO CV for RF …")
    loso_rf = loso_cv(rf_factory, X_arr, y_ev, y_src)
    print(f"  RF  LOSO overall acc={loso_rf['overall_acc']:.3f}")
    print(f"      per-source: {loso_rf['per_source_acc']}")

    print("Running LOSO CV for SVM …")
    loso_svm = loso_cv(svm_factory, X_arr, y_ev, y_src)
    print(f"  SVM LOSO overall acc={loso_svm['overall_acc']:.3f}")
    print(f"      per-source: {loso_svm['per_source_acc']}")

    results["loso_rf"] = loso_rf
    results["loso_svm"] = loso_svm

    # ---- Final models on all data ----
    print("Training final RF and SVM on all data …")
    final_rf = rf_factory()
    final_rf.fit(X_arr, y_ev)
    final_svm = svm_factory()
    final_svm.fit(X_arr, y_ev)

    preds_rf = final_rf.predict(X_arr)
    preds_svm = final_svm.predict(X_arr)

    rep_rf = classification_report(y_ev, preds_rf, labels=EVENTS, zero_division=0)
    rep_svm = classification_report(y_ev, preds_svm, labels=EVENTS, zero_division=0)

    with open(os.path.join(output_dir, "classification_report_RF.txt"), "w") as f:
        f.write("Random Forest — full dataset classification report\n\n")
        f.write(rep_rf)

    with open(os.path.join(output_dir, "classification_report_SVM.txt"), "w") as f:
        f.write("SVM — full dataset classification report\n\n")
        f.write(rep_svm)

    joblib.dump(final_rf, os.path.join(output_dir, "rf_model.pkl"))
    joblib.dump(final_svm, os.path.join(output_dir, "svm_model.pkl"))

    results["final_rf"] = final_rf
    results["final_svm"] = final_svm
    results["X_scaled"] = X_scaled
    results["X_pca2"] = X_pca2
    results["X_pca3"] = X_pca3
    results["imputer"] = imputer
    results["scaler"] = scaler

    return results
