"""
visualisation.py
================
All plots for the NOVOPTEL PM1000 Stokes polarimeter classification pipeline.
All figures are saved to outputs/.
"""

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

# Consistent colour scheme
EVENT_COLOURS = {
    "NE": "#808080",   # grey
    "FS": "#FF8C00",   # orange
    "VB": "#2CA02C",   # green
    "MB": "#9467BD",   # purple
    "TAP": "#1F77B4",  # blue
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
    "DOP":       ["mean_dop", "std_dop", "median_dop", "frac_dop_above"],
    "SOP motion": [
        "max_theta_ref", "mean_theta_ref", "rms_theta_ref", "std_theta_ref",
        "max_step", "mean_step", "rms_step", "std_step",
        "cum_arc", "excess_cum_arc", "frac_above_floor",
        "fs_jump_count", "tap_spike_energy", "mb_drift_slope",
    ],
    "Frequency": ["vb_bp_narrow", "vb_bp_wide", "vb_peak_freq", "vb_peak_prominence"],
    "Intensity": ["mean_s0", "std_s0", "cv_s0"],
}
CAT_COLOURS = {
    "DOP": "#1f77b4",
    "SOP motion": "#ff7f0e",
    "Frequency": "#2ca02c",
    "Intensity": "#d62728",
}


def _feature_category(fname):
    for cat, names in FEATURE_CATEGORIES.items():
        if fname in names:
            return cat
    return "Other"


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. DOP distributions (violin per event, coloured by source)
# ---------------------------------------------------------------------------

def plot_dop_distributions(X, y_event, y_source, output_dir):
    _ensure_dir(output_dir)
    df = X[["mean_dop"]].copy()
    df["event"] = y_event.values
    df["source"] = y_source.values

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = {e: EVENT_COLOURS[e] for e in EVENTS_ORDER if e in df["event"].unique()}
    sns.violinplot(data=df, x="event", y="mean_dop", order=EVENTS_ORDER,
                   palette=palette, inner="box", ax=ax, cut=0)
    ax.set_title("DOP distribution per event type")
    ax.set_xlabel("Event")
    ax.set_ylabel("Mean DOP")
    ax.set_ylim([-0.05, 1.1])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dop_distributions.png"), dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# 2. Step angle box plot per event, grouped by source
# ---------------------------------------------------------------------------

def plot_step_distributions(X, y_event, y_source, output_dir):
    _ensure_dir(output_dir)
    df = X[["mean_step"]].copy()
    df["event"] = y_event.values
    df["source"] = y_source.values

    fig, ax = plt.subplots(figsize=(12, 6))
    source_list = [s for s in SOURCES if s in df["source"].unique()]
    event_list = [e for e in EVENTS_ORDER if e in df["event"].unique()]
    n_src = len(source_list)
    n_ev = len(event_list)
    width = 0.8 / n_src
    x_pos = np.arange(n_ev)
    src_palette = sns.color_palette("tab10", n_src)

    for i, src in enumerate(source_list):
        sub = df[df["source"] == src]
        data_per_event = [sub[sub["event"] == ev]["mean_step"].dropna().values
                          for ev in event_list]
        positions = x_pos + (i - n_src / 2 + 0.5) * width
        bp = ax.boxplot(data_per_event, positions=positions, widths=width * 0.8,
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
# Helper: PCA transform
# ---------------------------------------------------------------------------

def _fit_pca(X_scaled, n_components):
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X_scaled)
    return pca, coords


def _confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    """Draw a covariance confidence ellipse on ax."""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
        return
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    vx, vy = eigenvectors[:, 0]
    theta = np.degrees(np.arctan2(vy, vx))
    width, height = 2 * n_std * np.sqrt(np.abs(eigenvalues))
    ellipse = Ellipse(xy=(np.mean(x), np.mean(y)),
                      width=width, height=height, angle=theta,
                      linewidth=1.5, fill=False, **kwargs)
    ax.add_patch(ellipse)


# ---------------------------------------------------------------------------
# 3. PCA scatter 2D
# ---------------------------------------------------------------------------

def plot_pca_scatter_2d(X_scaled, y_event, y_source, output_dir):
    _ensure_dir(output_dir)
    _pca, coords = _fit_pca(X_scaled, 2)

    events = y_event.values
    sources = y_source.values
    unique_events = [e for e in EVENTS_ORDER if e in events]
    unique_sources = [s for s in SOURCES if s in sources]

    fig, ax = plt.subplots(figsize=(9, 7))
    for ev in unique_events:
        for src in unique_sources:
            mask = (events == ev) & (sources == src)
            if not mask.any():
                continue
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=EVENT_COLOURS[ev],
                       marker=SOURCE_MARKERS.get(src, "o"),
                       s=50, alpha=0.7, edgecolors="none",
                       label=f"{ev}/{src}")

    # 95% confidence ellipses per event
    for ev in unique_events:
        mask = events == ev
        if mask.sum() >= 3:
            _confidence_ellipse(coords[mask, 0], coords[mask, 1], ax,
                                n_std=2.0, edgecolor=EVENT_COLOURS[ev],
                                linestyle="--", alpha=0.8)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA scatter (PC1 vs PC2) — colour=event, shape=source")

    event_handles = [mpatches.Patch(color=EVENT_COLOURS[e], label=e)
                     for e in unique_events]
    source_handles = [plt.scatter([], [], c="grey", marker=SOURCE_MARKERS.get(s, "o"),
                                  s=60, label=s) for s in unique_sources]
    legend1 = ax.legend(handles=event_handles, title="Event",
                        loc="upper left", fontsize=8)
    ax.add_artist(legend1)
    ax.legend(handles=source_handles, title="Source",
              loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_scatter_2d.png"), dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# 4. PCA scatter 3D
# ---------------------------------------------------------------------------

def plot_pca_scatter_3d(X_scaled, y_event, y_source, output_dir):
    _ensure_dir(output_dir)
    _pca, coords = _fit_pca(X_scaled, 3)

    events = y_event.values
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    unique_events = [e for e in EVENTS_ORDER if e in events]
    for ev in unique_events:
        mask = events == ev
        ax.scatter(coords[mask, 0], coords[mask, 1], coords[mask, 2],
                   c=EVENT_COLOURS[ev], label=ev, s=30, alpha=0.7)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA 3D scatter — colour=event")
    ax.legend(title="Event", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_scatter_3d.png"), dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# 5. K-Means clusters vs true labels in PCA 2D space
# ---------------------------------------------------------------------------

def plot_kmeans_clusters_2d(X_scaled, y_event, cluster_labels, label_map, output_dir):
    _ensure_dir(output_dir)
    _pca, coords = _fit_pca(X_scaled, 2)

    n_clusters = len(label_map)
    cluster_palette = sns.color_palette("tab10", n_clusters)
    events = y_event.values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: K-Means assignments
    for c in range(n_clusters):
        mask = cluster_labels == c
        ax1.scatter(coords[mask, 0], coords[mask, 1],
                    color=cluster_palette[c], label=f"Cluster {c} ({label_map[c]})",
                    s=30, alpha=0.7)
    ax1.set_title("K-Means cluster assignments")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.legend(fontsize=7, title="Cluster")

    # Right: true event labels
    unique_events = [e for e in EVENTS_ORDER if e in events]
    for ev in unique_events:
        mask = events == ev
        ax2.scatter(coords[mask, 0], coords[mask, 1],
                    color=EVENT_COLOURS[ev], label=ev, s=30, alpha=0.7)
    ax2.set_title("True event labels")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.legend(fontsize=7, title="Event")

    plt.suptitle("K-Means clusters vs true labels in PCA 2D space")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "kmeans_clusters_2d.png"), dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# 6 & 7. Confusion matrices (stratified CV)
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm, labels, title, outpath):
    fig, ax = plt.subplots(figsize=(7, 6))
    # Normalise rows so diagonal shows per-class recall
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                ax=ax, vmin=0, vmax=1)
    # Overlay raw counts
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j + 0.5, i + 0.72, f"({cm[i, j]})",
                    ha="center", va="center", fontsize=7, color="grey")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close()


def plot_confusion_matrices(results, output_dir):
    _ensure_dir(output_dir)
    from ml_classification import EVENTS
    labels = EVENTS

    plot_confusion_matrix(
        results["cv_rf"]["cm"], labels,
        "Confusion matrix — RF (stratified 5-fold)",
        os.path.join(output_dir, "confusion_matrix_RF.png"),
    )
    plot_confusion_matrix(
        results["cv_svm"]["cm"], labels,
        "Confusion matrix — SVM (stratified 5-fold)",
        os.path.join(output_dir, "confusion_matrix_SVM.png"),
    )
    plot_confusion_matrix(
        results["loso_rf"]["cm"], labels,
        "Confusion matrix — RF (LOSO)",
        os.path.join(output_dir, "confusion_matrix_LOSO_RF.png"),
    )
    plot_confusion_matrix(
        results["loso_svm"]["cm"], labels,
        "Confusion matrix — SVM (LOSO)",
        os.path.join(output_dir, "confusion_matrix_LOSO_SVM.png"),
    )


# ---------------------------------------------------------------------------
# 10. Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(final_rf, feature_names, output_dir):
    _ensure_dir(output_dir)
    rf_clf = final_rf.named_steps["clf"]
    importances = rf_clf.feature_importances_
    indices = np.argsort(importances)[::-1][:15]

    names_top = [feature_names[i] for i in indices]
    imp_top = importances[indices]
    colours = [CAT_COLOURS.get(_feature_category(n), "#7f7f7f") for n in names_top]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(range(len(names_top))[::-1], imp_top, color=colours)
    ax.set_yticks(range(len(names_top))[::-1])
    ax.set_yticklabels(names_top, fontsize=9)
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title("Top 15 feature importances (Random Forest)")

    cat_handles = [mpatches.Patch(color=CAT_COLOURS[c], label=c)
                   for c in CAT_COLOURS]
    ax.legend(handles=cat_handles, title="Category",
              loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# 11. Feature correlation heatmap
# ---------------------------------------------------------------------------

def plot_feature_correlation(X, output_dir):
    _ensure_dir(output_dir)
    corr = X.apply(pd.to_numeric, errors="coerce").corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax,
                xticklabels=True, yticklabels=True,
                annot=False, linewidths=0.3)
    ax.set_title("Feature Pearson correlation matrix")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_correlation.png"), dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# 12. Per-source LOSO accuracy bar chart
# ---------------------------------------------------------------------------

def plot_per_source_accuracy(results, output_dir):
    _ensure_dir(output_dir)
    sources = list(results["loso_rf"]["per_source_acc"].keys())
    rf_accs = [results["loso_rf"]["per_source_acc"].get(s, 0) for s in sources]
    svm_accs = [results["loso_svm"]["per_source_acc"].get(s, 0) for s in sources]

    x = np.arange(len(sources))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, rf_accs, width, label="RF", color="#1f77b4")
    ax.bar(x + width / 2, svm_accs, width, label="SVM", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(sources, rotation=20, ha="right")
    ax.set_ylim([0, 1.05])
    ax.set_ylabel("Test accuracy")
    ax.set_title("LOSO test accuracy per source (RF vs SVM)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_source_accuracy.png"), dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# 13. Learning curves (GroupKFold keyed on source — LOSO-style)
# ---------------------------------------------------------------------------

def plot_learning_curves(final_rf, final_svm, X_arr, y_event, y_source, output_dir):
    """Plot learning curves for RF and SVM using GroupKFold keyed on source.

    GroupKFold ensures train/test folds never share the same optical source,
    giving an honest LOSO-style generalisation estimate as a function of
    training size. The gap between train and CV accuracy is annotated to
    diagnose overfitting.
    """
    from sklearn.model_selection import learning_curve, GroupKFold
    from sklearn.metrics import f1_score

    _ensure_dir(output_dir)

    y_arr = y_event.values if hasattr(y_event, "values") else np.array(y_event)
    groups = y_source.values if hasattr(y_source, "values") else np.array(y_source)

    n_samples = len(y_arr)
    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    cv = GroupKFold(n_splits=n_splits)

    train_sizes_rel = np.linspace(0.1, 1.0, 15)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    models = [
        (final_rf,  "Learning Curve — RF (LOSO-style, grouped by source)",  axes[0]),
        (final_svm, "Learning Curve — SVM (LOSO-style, grouped by source)", axes[1]),
    ]

    for model, title, ax in models:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_sizes_abs, train_scores, cv_scores = learning_curve(
                model, X_arr, y_arr,
                train_sizes=train_sizes_rel,
                cv=cv,
                groups=groups,
                scoring="accuracy",
                n_jobs=-1,
                error_score=0.0,
            )

        train_mean = train_scores.mean(axis=1)
        train_std  = train_scores.std(axis=1)
        cv_mean    = cv_scores.mean(axis=1)
        cv_std     = cv_scores.std(axis=1)

        # Training score — blue
        ax.plot(train_sizes_abs, train_mean, "o-", color="#1f77b4",
                linewidth=1.8, markersize=4, label="Training accuracy")
        ax.fill_between(train_sizes_abs,
                        train_mean - train_std, train_mean + train_std,
                        alpha=0.15, color="#1f77b4")

        # CV score — orange
        ax.plot(train_sizes_abs, cv_mean, "s-", color="#ff7f0e",
                linewidth=1.8, markersize=4, label="CV accuracy (GroupKFold by source)")
        ax.fill_between(train_sizes_abs,
                        cv_mean - cv_std, cv_mean + cv_std,
                        alpha=0.15, color="#ff7f0e")

        # Reference line
        final_cv = cv_mean[-1]
        ax.axhline(final_cv, color="#ff7f0e", linestyle="--",
                   linewidth=1.0, alpha=0.6, label=f"Final CV acc = {final_cv:.2f}")

        # Gap annotation
        gap = train_mean[-1] - cv_mean[-1]
        mid_y = (train_mean[-1] + cv_mean[-1]) / 2
        ax.annotate(
            f"Gap: {gap:.2f}",
            xy=(train_sizes_abs[-1], mid_y),
            xytext=(-60, 0), textcoords="offset points",
            fontsize=9, color="dimgrey",
            arrowprops=dict(arrowstyle="->", color="dimgrey"),
        )

        # Per-class F1 at full data
        full_preds = model.predict(X_arr)
        class_f1 = f1_score(y_arr, full_preds,
                            labels=EVENTS_ORDER, average=None, zero_division=0)
        table_text = "Full-data per-class F1:\n" + "  ".join(
            f"{ev}:{f1:.2f}" for ev, f1 in zip(EVENTS_ORDER, class_f1)
        )
        ax.text(0.02, 0.04, table_text, transform=ax.transAxes,
                fontsize=7.5, verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Number of training samples")
        ax.set_ylabel("Accuracy")
        y_lo = max(0.0, min(cv_mean.min(), train_mean.min()) - 0.12)
        ax.set_ylim([y_lo, 1.05])
        ax.set_xlim([train_sizes_abs[0] * 0.9, train_sizes_abs[-1] * 1.05])
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle(
        f"Learning curves — {n_samples} samples, grouped by source (LOSO-style CV)\n"
        "Large gap → source-specific SOP bias; consider source-normalised features",
        fontsize=9, color="grey"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_curves.png"), dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# 14. DOP vs time overlay (one subplot per source)
# ---------------------------------------------------------------------------

def plot_dop_vs_time_overlay(dataset_dir, output_dir, t_start=0, t_end=60,
                             dop_thresh=0.2):
    """Load one representative file per (source, event) combo and overlay DOP(t)."""
    import glob as _glob
    from feature_extraction import parse_filename, load_csv, compute_dop

    _ensure_dir(output_dir)

    # Gather files
    all_files = {}
    search_dirs = [dataset_dir, os.path.dirname(dataset_dir)]
    for d in search_dirs:
        for fp in _glob.glob(os.path.join(d, "pm1000_sop_*.csv")):
            bn = os.path.basename(fp)
            meta = parse_filename(bn)
            if meta:
                key = (meta["source"].upper(), meta["event"].upper())
                all_files.setdefault(key, fp)  # keep first occurrence

    sources = ["SP-AGIL", "SP-PURE", "DPQAM16-200G", "DPQPSK-200G", "10GE"]
    events = ["NE", "FS", "VB", "MB", "TAP"]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    for ax, src in zip(axes, sources):
        for ev in events:
            fp = all_files.get((src, ev))
            if fp is None:
                continue
            try:
                df, _fs = load_csv(fp, t_start=t_start, t_end=t_end)
                dop = compute_dop(df)
                ax.plot(df["Time_s"].values, dop,
                        color=EVENT_COLOURS[ev], alpha=0.7,
                        label=ev, linewidth=0.8)
            except Exception:
                pass
        ax.set_title(src, fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylim([-0.05, 1.1])
        ax.axhline(dop_thresh, color="k", linestyle=":", linewidth=0.8, alpha=0.5)
    axes[0].set_ylabel("DOP")

    handles = [mpatches.Patch(color=EVENT_COLOURS[e], label=e) for e in events]
    fig.legend(handles=handles, title="Event", loc="upper right", fontsize=8)
    plt.suptitle("DOP vs time per source (all events overlaid)", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dop_vs_time_overlay.png"),
                dpi=120, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Master function: generate all plots
# ---------------------------------------------------------------------------

def generate_all_plots(X, y_event, y_source, results, feature_names,
                       dataset_dir, output_dir):
    print("Generating plots …")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        X_scaled = results["X_scaled"]
        km_res = results["kmeans"]

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
        print("  ✓ confusion_matrix_{RF,SVM,LOSO_RF,LOSO_SVM}.png")

        plot_feature_importance(results["final_rf"], feature_names, output_dir)
        print("  ✓ feature_importance.png")

        plot_feature_correlation(X, output_dir)
        print("  ✓ feature_correlation.png")

        plot_per_source_accuracy(results, output_dir)
        print("  ✓ per_source_accuracy.png")

        X_arr = X.values.astype(float)
        plot_learning_curves(results["final_rf"], results["final_svm"],
                             X_arr, y_event, y_source, output_dir)
        print("  ✓ learning_curves.png")

        plot_dop_vs_time_overlay(dataset_dir, output_dir)
        print("  ✓ dop_vs_time_overlay.png")
