"""
visualisation.py  —  new_ml_pipeline
======================================
All plots for the redesigned NOVOPTEL PM1000 ML pipeline.

New plots added vs the original ml_pipeline:
* ``plot_learning_curves_improved()`` — 3-panel (RF, SVM, XGB) with overfitting
  gap line, ±1 std shaded bands, 10 training-size points, reference lines at 0.8/0.9
* ``plot_per_source_confusion_matrices()`` — per-source CM caller (files are saved
  directly in ``run_per_source_pipeline`` via ``plot_confusion_matrix()``)
* ``plot_per_source_event_accuracy()`` — combined bar chart: accuracy per source
  per model (3 groups of 5 bars, one group per model)

Updated plots:
* ``plot_confusion_matrices()`` — now includes XGBoost CMs
* ``plot_per_source_accuracy()`` — now shows 3 bars (RF, SVM, XGBoost) per source
* ``generate_all_plots()`` — calls all new functions
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import seaborn as sns
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Colour scheme
# ---------------------------------------------------------------------------

EVENT_COLOURS = {
    "NE":  "#808080",
    "FS":  "#FF8C00",
    "VB":  "#2CA02C",
    "MB":  "#9467BD",
    "TAP": "#1F77B4",
}
EVENTS_ORDER = ["NE", "FS", "VB", "MB", "TAP"]

SOURCES = ["SP-AGIL", "SP-PURE", "DPQAM16-200G", "DPQPSK-200G", "10GE"]
SOURCE_MARKERS = {
    "SP-AGIL":      "o",
    "SP-PURE":      "s",
    "DPQAM16-200G": "^",
    "DPQPSK-200G":  "D",
    "10GE":         "X",
}

FEATURE_CATEGORIES = {
    "DOP":         ["dop_mean", "dop_std", "dop_min", "dop_max",
                    "var_dop", "iqr_dop", "frac_dop_low"],
    "SOP motion":  [
        "theta_mean", "theta_std", "theta_max", "theta_rms",
        "range_theta_ref", "p95_theta_ref",
        "step_mean", "step_std", "step_max", "step_rms",
        "step_p95", "step_p99", "kurtosis_step", "skew_step",
        "burst_count", "cum_arc", "step_autocorr_lag1",
    ],
    "Frequency":   ["psd_peak_freq", "psd_peak_power",
                    "bp_low", "bp_mid", "bp_high",
                    "bp_ratio_mid_low", "psd_peak_sharpness",
                    "spectral_entropy", "vb_snr_80hz_db"],
    "Stokes":      ["s1_std", "s2_std", "s3_std", "s1_range", "s2_range", "s3_range"],
    "Modulation":  ["is_modulated"],
}
CAT_COLOURS = {
    "DOP":        "#1f77b4",
    "SOP motion": "#ff7f0e",
    "Frequency":  "#2ca02c",
    "Stokes":     "#d62728",
    "Modulation": "#9467bd",
}

MODEL_COLOURS = {
    "RF":      "#1f77b4",
    "SVM":     "#ff7f0e",
    "XGBoost": "#2ca02c",
}


def _feature_category(fname: str) -> str:
    for cat, names in FEATURE_CATEGORIES.items():
        if fname in names:
            return cat
    return "Other"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Basic distribution plots
# ---------------------------------------------------------------------------

def plot_dop_distributions(X: pd.DataFrame, y_event: pd.Series,
                           y_source: pd.Series, output_dir: str) -> None:
    _ensure_dir(output_dir)
    df = X[["dop_mean"]].copy()
    df["event"]  = y_event.values
    df["source"] = y_source.values

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = {e: EVENT_COLOURS[e] for e in EVENTS_ORDER if e in df["event"].unique()}
    sns.violinplot(data=df, x="event", y="dop_mean", order=EVENTS_ORDER,
                   palette=palette, inner="box", ax=ax, cut=0)
    ax.set_title("DOP distribution per event type")
    ax.set_xlabel("Event")
    ax.set_ylabel("Mean DOP")
    ax.set_ylim([-0.05, 1.1])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dop_distributions.png"), dpi=120)
    plt.close()


def plot_step_distributions(X: pd.DataFrame, y_event: pd.Series,
                            y_source: pd.Series, output_dir: str) -> None:
    _ensure_dir(output_dir)
    df = X[["step_mean"]].copy()
    df["event"]  = y_event.values
    df["source"] = y_source.values

    fig, ax = plt.subplots(figsize=(12, 6))
    source_list = [s for s in SOURCES if s in df["source"].unique()]
    event_list  = [e for e in EVENTS_ORDER if e in df["event"].unique()]
    n_src  = len(source_list)
    n_ev   = len(event_list)
    width  = 0.8 / n_src
    x_pos  = np.arange(n_ev)
    src_palette = sns.color_palette("tab10", n_src)

    for i, src in enumerate(source_list):
        sub  = df[df["source"] == src]
        data = [sub[sub["event"] == ev]["step_mean"].dropna().values for ev in event_list]
        pos  = x_pos + (i - n_src / 2 + 0.5) * width
        ax.boxplot(data, positions=pos, widths=width * 0.8,
                   patch_artist=True,
                   boxprops=dict(facecolor=(*src_palette[i], 0.6)),
                   medianprops=dict(color="black"),
                   whiskerprops=dict(color="grey"),
                   capprops=dict(color="grey"),
                   flierprops=dict(marker=".", markersize=3, alpha=0.4))

    ax.set_xticks(x_pos)
    ax.set_xticklabels(event_list)
    ax.set_xlabel("Event")
    ax.set_ylabel("Mean step angle (deg)")
    ax.set_title("Mean step angle per event type, grouped by source")
    handles = [mpatches.Patch(color=src_palette[i], label=s)
               for i, s in enumerate(source_list)]
    ax.legend(handles=handles, title="Source", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "step_distributions.png"), dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# PCA scatter
# ---------------------------------------------------------------------------

def _fit_pca(X_scaled: np.ndarray, n_components: int):
    pca    = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X_scaled)
    return pca, coords


def _confidence_ellipse(x: np.ndarray, y: np.ndarray, ax, n_std: float = 2.0,
                        **kwargs) -> None:
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
        return
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    vx, vy   = eigenvectors[:, 0]
    theta    = np.degrees(np.arctan2(vy, vx))
    width    = 2 * n_std * np.sqrt(np.abs(eigenvalues[0]))
    height   = 2 * n_std * np.sqrt(np.abs(eigenvalues[1]))
    ellipse  = Ellipse(xy=(np.mean(x), np.mean(y)),
                       width=width, height=height, angle=theta,
                       linewidth=1.5, fill=False, **kwargs)
    ax.add_patch(ellipse)


def plot_pca_scatter_2d(X_scaled: np.ndarray, y_event: pd.Series,
                        y_source: pd.Series, output_dir: str) -> None:
    _ensure_dir(output_dir)
    _pca, coords = _fit_pca(X_scaled, 2)
    events  = y_event.values
    sources = y_source.values
    unique_events  = [e for e in EVENTS_ORDER if e in events]
    unique_sources = [s for s in SOURCES     if s in sources]

    fig, ax = plt.subplots(figsize=(9, 7))
    for ev in unique_events:
        for src in unique_sources:
            mask = (events == ev) & (sources == src)
            if not mask.any():
                continue
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=EVENT_COLOURS[ev],
                       marker=SOURCE_MARKERS.get(src, "o"),
                       s=50, alpha=0.7, edgecolors="none")

    for ev in unique_events:
        mask = events == ev
        if mask.sum() >= 3:
            _confidence_ellipse(coords[mask, 0], coords[mask, 1], ax,
                                n_std=2.0, edgecolor=EVENT_COLOURS[ev],
                                linestyle="--", alpha=0.8)

    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("PCA scatter (PC1 vs PC2) — colour=event, shape=source")
    event_handles  = [mpatches.Patch(color=EVENT_COLOURS[e], label=e) for e in unique_events]
    source_handles = [plt.scatter([], [], c="grey", marker=SOURCE_MARKERS.get(s, "o"),
                                  s=60, label=s) for s in unique_sources]
    leg1 = ax.legend(handles=event_handles, title="Event", loc="upper left", fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=source_handles, title="Source", loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_scatter_2d.png"), dpi=120)
    plt.close()


def plot_pca_scatter_3d(X_scaled: np.ndarray, y_event: pd.Series,
                        y_source: pd.Series, output_dir: str) -> None:
    _ensure_dir(output_dir)
    _pca, coords = _fit_pca(X_scaled, 3)
    events = y_event.values
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")
    unique_events = [e for e in EVENTS_ORDER if e in events]
    for ev in unique_events:
        mask = events == ev
        ax.scatter(coords[mask, 0], coords[mask, 1], coords[mask, 2],
                   c=EVENT_COLOURS[ev], label=ev, s=30, alpha=0.7)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title("PCA 3D scatter — colour=event")
    ax.legend(title="Event", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_scatter_3d.png"), dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# K-Means
# ---------------------------------------------------------------------------

def plot_kmeans_clusters_2d(X_scaled: np.ndarray, y_event: pd.Series,
                             cluster_labels: np.ndarray, label_map: dict,
                             output_dir: str) -> None:
    _ensure_dir(output_dir)
    _pca, coords = _fit_pca(X_scaled, 2)
    n_clusters      = len(label_map)
    cluster_palette = sns.color_palette("tab10", n_clusters)
    events          = y_event.values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for c in range(n_clusters):
        mask = cluster_labels == c
        ax1.scatter(coords[mask, 0], coords[mask, 1],
                    color=cluster_palette[c],
                    label=f"Cluster {c} ({label_map[c]})", s=30, alpha=0.7)
    ax1.set_title("K-Means cluster assignments")
    ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2")
    ax1.legend(fontsize=7, title="Cluster")

    unique_events = [e for e in EVENTS_ORDER if e in events]
    for ev in unique_events:
        mask = events == ev
        ax2.scatter(coords[mask, 0], coords[mask, 1],
                    color=EVENT_COLOURS[ev], label=ev, s=30, alpha=0.7)
    ax2.set_title("True event labels")
    ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
    ax2.legend(fontsize=7, title="Event")

    plt.suptitle("K-Means clusters vs true labels in PCA 2D space")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "kmeans_clusters_2d.png"), dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Confusion matrices
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm: np.ndarray, labels: list, title: str,
                          outpath: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm / row_sums, 0.0)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                ax=ax, vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j + 0.5, i + 0.72, f"({cm[i, j]})",
                    ha="center", va="center", fontsize=7, color="grey")
    ax.set_title(title)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close()


def plot_confusion_matrices(results: dict, output_dir: str) -> None:
    """Save confusion matrices for all three classifiers (stratified CV + LOSO)."""
    _ensure_dir(output_dir)
    labels = EVENTS_ORDER

    for clf_name in ("RF", "SVM", "XGBoost"):
        cv_key   = f"cv_{clf_name}"
        loso_key = f"loso_{clf_name}"

        if cv_key in results:
            plot_confusion_matrix(
                results[cv_key]["cm"], labels,
                f"Confusion matrix — {clf_name} (stratified 5-fold)",
                os.path.join(output_dir, f"confusion_matrix_{clf_name}.png"),
            )
        if loso_key in results:
            plot_confusion_matrix(
                results[loso_key]["cm"], labels,
                f"Confusion matrix — {clf_name} (LOSO)",
                os.path.join(output_dir, f"confusion_matrix_LOSO_{clf_name}.png"),
            )


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(final_rf, feature_names: list, output_dir: str) -> None:
    _ensure_dir(output_dir)
    rf_clf      = final_rf.named_steps["clf"]
    importances = rf_clf.feature_importances_
    indices     = np.argsort(importances)[::-1][:15]

    names_top = [feature_names[i] for i in indices]
    imp_top   = importances[indices]
    colours   = [CAT_COLOURS.get(_feature_category(n), "#7f7f7f") for n in names_top]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(names_top))[::-1], imp_top, color=colours)
    ax.set_yticks(range(len(names_top))[::-1])
    ax.set_yticklabels(names_top, fontsize=9)
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title("Top 15 feature importances (Random Forest)")
    cat_handles = [mpatches.Patch(color=CAT_COLOURS[c], label=c)
                   for c in CAT_COLOURS if c in {_feature_category(n) for n in names_top}]
    ax.legend(handles=cat_handles, title="Category", loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Feature correlation
# ---------------------------------------------------------------------------

def plot_feature_correlation(X: pd.DataFrame, output_dir: str) -> None:
    _ensure_dir(output_dir)
    corr = X.apply(pd.to_numeric, errors="coerce").corr()
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax,
                xticklabels=True, yticklabels=True,
                annot=False, linewidths=0.3)
    ax.set_title("Feature Pearson correlation matrix (41 features)")
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    ax.tick_params(axis="y", rotation=0,  labelsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_correlation.png"), dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Per-source LOSO accuracy (global LOSO — 3 bars per source)
# ---------------------------------------------------------------------------

def plot_per_source_accuracy(results: dict, output_dir: str) -> None:
    """Bar chart: LOSO test accuracy per source for RF, SVM, XGBoost."""
    _ensure_dir(output_dir)
    sources = SOURCES

    accs: dict[str, list[float]] = {clf: [] for clf in ("RF", "SVM", "XGBoost")}
    for clf_name in ("RF", "SVM", "XGBoost"):
        loso_key = f"loso_{clf_name}"
        if loso_key in results:
            for s in sources:
                accs[clf_name].append(
                    results[loso_key]["per_source_acc"].get(s, 0.0)
                )
        else:
            accs[clf_name] = [0.0] * len(sources)

    x     = np.arange(len(sources))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (clf_name, colour) in enumerate(MODEL_COLOURS.items()):
        ax.bar(x + (i - 1) * width, accs[clf_name], width,
               label=clf_name, color=colour)
    ax.set_xticks(x)
    ax.set_xticklabels(sources, rotation=20, ha="right")
    ax.set_ylim([0, 1.05])
    ax.set_ylabel("LOSO test accuracy")
    ax.set_title("LOSO test accuracy per source (RF vs SVM vs XGBoost)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_source_accuracy.png"), dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Improved learning curves (Requirement 3)
# ---------------------------------------------------------------------------

def plot_learning_curves_improved(X_arr: np.ndarray, y_ev: np.ndarray,
                                   output_dir: str) -> None:
    """3-panel learning curve figure: RF, SVM, XGBoost.

    For each model:
    * 10 training size points from 10% to 100%
    * Stratified 5-fold CV at each point
    * Training accuracy AND CV accuracy with ±1 std shaded bands
    * Overfitting gap line (training − CV accuracy) as dashed line
    * Horizontal reference lines at 0.8 and 0.9
    * Subplot title shows model name and final CV accuracy
    Saved as learning_curves_improved.png at dpi=150.
    """
    from sklearn.model_selection import learning_curve, StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from ml_classification import CLASSIFIERS, EVENTS

    _ensure_dir(output_dir)

    train_sizes = np.linspace(0.1, 1.0, 10)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    le = LabelEncoder().fit(y_ev)

    for ax, clf_name in zip(axes, ("RF", "SVM", "XGBoost")):
        estimator = CLASSIFIERS[clf_name]()

        if clf_name == "XGBoost":
            y_fit = le.transform(y_ev)
        else:
            y_fit = y_ev

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_sz, train_scores, val_scores = learning_curve(
                estimator, X_arr, y_fit,
                train_sizes=train_sizes,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring="accuracy",
                shuffle=True,
                random_state=42,
                n_jobs=-1,
            )

        train_mean = train_scores.mean(axis=1)
        train_std  = train_scores.std(axis=1)
        val_mean   = val_scores.mean(axis=1)
        val_std    = val_scores.std(axis=1)
        gap        = train_mean - val_mean   # overfitting gap

        final_cv_acc = val_mean[-1]

        # Training accuracy
        ax.plot(train_sz, train_mean, "o-", color="#1f77b4", label="Training accuracy")
        ax.fill_between(train_sz, train_mean - train_std, train_mean + train_std,
                        alpha=0.15, color="#1f77b4")

        # CV accuracy
        ax.plot(train_sz, val_mean, "o-", color="#ff7f0e", label="CV accuracy")
        ax.fill_between(train_sz, val_mean - val_std, val_mean + val_std,
                        alpha=0.15, color="#ff7f0e")

        # Overfitting gap line
        ax.plot(train_sz, gap, "--", color="#d62728", linewidth=1.5,
                label="Gap (train − CV)")

        # Reference lines
        ax.axhline(0.9, color="grey", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.axhline(0.8, color="grey", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.text(train_sz[0], 0.905, "0.9", fontsize=7, color="grey", va="bottom")
        ax.text(train_sz[0], 0.805, "0.8", fontsize=7, color="grey", va="bottom")

        ax.set_title(f"{clf_name}\n(final CV acc = {final_cv_acc:.3f})", fontsize=11)
        ax.set_xlabel("Training set size")
        if ax == axes[0]:
            ax.set_ylabel("Accuracy")
        ax.set_ylim([0.0, 1.05])
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Learning curves — stratified 5-fold CV (10 training sizes, ±1 std)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_curves_improved.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Per-source event accuracy bar chart (Requirement 4)
# ---------------------------------------------------------------------------

def plot_per_source_event_accuracy(per_source_results: dict,
                                    output_dir: str) -> None:
    """Combined bar chart: accuracy per source per model (stratified CV).

    3 groups of 5 bars (one group per model: RF, SVM, XGBoost).
    """
    _ensure_dir(output_dir)

    clf_names = [c for c in ("RF", "SVM", "XGBoost") if any(
        c in per_source_results.get(s, {}) for s in SOURCES
    )]
    sources_present = [s for s in SOURCES if s in per_source_results]

    if not sources_present or not clf_names:
        return

    n_models  = len(clf_names)
    n_sources = len(sources_present)
    x         = np.arange(n_sources)
    width     = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(13, 6))
    palette = sns.color_palette("tab10", n_models)

    for i, clf_name in enumerate(clf_names):
        accs = []
        for src in sources_present:
            cv_res = per_source_results.get(src, {}).get(clf_name, {}).get("strat_cv", {})
            accs.append(cv_res.get("mean_acc", 0.0))
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, accs, width, label=clf_name, color=palette[i])

    ax.set_xticks(x)
    ax.set_xticklabels(sources_present, rotation=20, ha="right")
    ax.set_ylim([0, 1.05])
    ax.set_ylabel("Stratified 5-fold CV accuracy")
    ax.set_title("Per-source event classification accuracy (stratified 5-fold CV)")
    ax.axhline(0.8, color="grey", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.axhline(0.9, color="grey", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.legend(title="Model")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_source_event_accuracy.png"), dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Stub (for compatibility)
# ---------------------------------------------------------------------------

def plot_dop_vs_time_overlay(*args, **kwargs) -> None:
    print("  [SKIP] dop_vs_time_overlay — not available in windowed pipeline.")


# ---------------------------------------------------------------------------
# generate_all_plots — master caller
# ---------------------------------------------------------------------------

def generate_all_plots(X: pd.DataFrame, y_event: pd.Series, y_source: pd.Series,
                       results: dict, feature_names: list,
                       dataset_dir: str, output_dir: str,
                       per_source_results: dict | None = None) -> None:
    print("Generating plots …")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        X_scaled = results["X_scaled"]
        km_res   = results["kmeans"]

        plot_dop_distributions(X, y_event, y_source, output_dir)
        print("  ✓ dop_distributions.png")

        plot_step_distributions(X, y_event, y_source, output_dir)
        print("  ✓ step_distributions.png")

        plot_pca_scatter_2d(X_scaled, y_event, y_source, output_dir)
        print("  ✓ pca_scatter_2d.png")

        plot_pca_scatter_3d(X_scaled, y_event, y_source, output_dir)
        print("  ✓ pca_scatter_3d.png")

        plot_kmeans_clusters_2d(X_scaled, y_event,
                                km_res["cluster_labels"], km_res["label_map"],
                                output_dir)
        print("  ✓ kmeans_clusters_2d.png")

        plot_confusion_matrices(results, output_dir)
        print("  ✓ confusion_matrix_{RF,SVM,XGBoost,LOSO_RF,LOSO_SVM,LOSO_XGBoost}.png")

        plot_feature_importance(results["final_RF"], feature_names, output_dir)
        print("  ✓ feature_importance.png")

        plot_feature_correlation(X, output_dir)
        print("  ✓ feature_correlation.png")

        plot_per_source_accuracy(results, output_dir)
        print("  ✓ per_source_accuracy.png")

        plot_learning_curves_improved(X_scaled, y_event.values, output_dir)
        print("  ✓ learning_curves_improved.png")

        if per_source_results:
            plot_per_source_event_accuracy(per_source_results,
                                           os.path.join(output_dir, "per_source"))
            print("  ✓ per_source/per_source_event_accuracy.png")
