"""
visualisation.py  —  new_ml_pipeline_2
========================================
Plots for the per-source LOSO pipeline.

Functions
---------
plot_confusion_matrix()          — single CM heatmap
plot_cm_all_sources()            — 1×5 combined CM figure for one model
generate_loso_learning_curves()  — per-source learning curve plots
generate_all_plots()             — top-level caller
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from ml_classification import loso_learning_curve

# ---------------------------------------------------------------------------
# Colour constants
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
MODEL_COLOURS = {
    "RF":      "#1f77b4",
    "SVM":     "#ff7f0e",
    "XGBoost": "#2ca02c",
}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Single confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    title: str,
    save_path: str,
) -> None:
    """Save a normalised confusion matrix heatmap."""
    _ensure_dir(os.path.dirname(save_path))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm  = np.where(row_sums > 0, cm / row_sums, 0.0)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            vmin=0, vmax=1, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title, fontsize=10)
        plt.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Combined 1×5 CM figure (all sources, one model)
# ---------------------------------------------------------------------------

def plot_cm_all_sources(
    per_source_results: dict,
    clf_name: str,
    output_dir: str,
) -> None:
    """Save a 1×5 grid of LOSO confusion matrices — one per source."""
    fig, axes = plt.subplots(1, len(SOURCES), figsize=(4 * len(SOURCES), 4.5))
    if len(SOURCES) == 1:
        axes = [axes]

    for ax, src in zip(axes, SOURCES):
        if src not in per_source_results:
            ax.set_visible(False)
            continue
        cm = per_source_results[src].get(clf_name, {}).get("loso", {}).get(
            "cm", np.zeros((len(EVENTS_ORDER), len(EVENTS_ORDER)), dtype=int))
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm  = np.where(row_sums > 0, cm / row_sums, 0.0)
        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=EVENTS_ORDER, yticklabels=EVENTS_ORDER,
            vmin=0, vmax=1, ax=ax, cbar=False,
        )
        acc = per_source_results[src].get(clf_name, {}).get("loso", {}).get(
            "accuracy", float("nan"))
        ax.set_title(f"{src}\nacc={acc:.3f}", fontsize=9)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)

    fig.suptitle(f"LOSO Confusion Matrices — {clf_name}", fontsize=12)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"cm_loso_all_sources_{clf_name}.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Per-source learning curves
# ---------------------------------------------------------------------------

def plot_loso_learning_curve_source(
    source: str,
    clf_results: dict,
    strat_cv_accs: dict,
    output_dir: str,
) -> None:
    """3-subplot figure (RF, SVM, XGBoost) for one source.

    clf_results  : {clf_name: [(n_sources, mean_acc, std_acc), ...]}
    strat_cv_accs: {clf_name: float} — stratified 5-fold CV accuracy (baseline)
    """
    _ensure_dir(output_dir)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle(f"LOSO Learning Curves — {source}", fontsize=13)

    for ax, clf_name in zip(axes, ("RF", "SVM", "XGBoost")):
        data = clf_results.get(clf_name, [])
        cv_acc = strat_cv_accs.get(clf_name, float("nan"))

        if data:
            ns     = [d[0] for d in data]
            means  = [d[1] for d in data]
            stds   = [d[2] for d in data]
            color  = MODEL_COLOURS[clf_name]

            ax.plot(ns, means, "o-", color=color, lw=2, label="LOSO acc")
            ax.fill_between(ns,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            alpha=0.2, color=color)

            # Annotate final point
            ax.annotate(f"{means[-1]:.3f}",
                        xy=(ns[-1], means[-1]),
                        xytext=(5, 5), textcoords="offset points", fontsize=8)

        # Stratified 5-fold baseline
        if not np.isnan(cv_acc):
            ax.axhline(cv_acc, color="orange", lw=1.5, ls="--",
                       label=f"5-fold CV={cv_acc:.3f}")

        ax.axhline(0.8, color="grey", lw=0.8, ls=":", alpha=0.6)
        ax.axhline(0.9, color="grey", lw=0.8, ls=":", alpha=0.6)
        ax.set_xlim(0.5, 4.5)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("Number of training sources")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{source} — {clf_name}\n(LOSO accuracy vs n training sources)",
                     fontsize=9)
        ax.legend(fontsize=8)

    plt.tight_layout()
    safe_src  = source.replace("/", "_").replace("-", "_")
    save_path = os.path.join(output_dir, f"learning_curve_loso_{safe_src}.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def generate_loso_learning_curves(
    X: pd.DataFrame,
    y_event: pd.Series,
    y_source: pd.Series,
    per_source_results: dict,
    output_dir: str,
) -> None:
    """Compute and plot LOSO learning curves for all 5 sources."""
    _ensure_dir(output_dir)
    X_arr   = X.values.astype(float)
    y_arr   = y_event.values
    src_arr = y_source.values

    # Per-source figures
    for src in SOURCES:
        if src not in per_source_results:
            continue
        print(f"  Computing learning curves for {src} ...")
        clf_results: dict = {}
        for clf_name in ("RF", "SVM", "XGBoost"):
            print(f"    {clf_name} ...", end=" ", flush=True)
            curve = loso_learning_curve(clf_name, X_arr, y_arr, src_arr, src)
            clf_results[clf_name] = curve
            print("done")

        strat_cv_accs = {
            clf: per_source_results[src].get(clf, {}).get(
                "strat_cv", {}).get("mean_acc", float("nan"))
            for clf in ("RF", "SVM", "XGBoost")
        }
        plot_loso_learning_curve_source(src, clf_results, strat_cv_accs, output_dir)

    # Combined 5×3 summary figure
    _plot_combined_learning_curves(X_arr, y_arr, src_arr,
                                   per_source_results, output_dir)


def _plot_combined_learning_curves(
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    src_arr: np.ndarray,
    per_source_results: dict,
    output_dir: str,
) -> None:
    """5×3 grid: rows=sources, cols=models."""
    present_sources = [s for s in SOURCES if s in per_source_results]
    n_src = len(present_sources)
    if n_src == 0:
        return

    fig, axes = plt.subplots(n_src, 3, figsize=(12, 3.5 * n_src), squeeze=False)
    fig.suptitle("LOSO Learning Curves — All Sources", fontsize=13)

    for row_idx, src in enumerate(present_sources):
        for col_idx, clf_name in enumerate(("RF", "SVM", "XGBoost")):
            ax = axes[row_idx][col_idx]
            curve = loso_learning_curve(clf_name, X_arr, y_arr, src_arr, src)
            cv_acc = per_source_results[src].get(clf_name, {}).get(
                "strat_cv", {}).get("mean_acc", float("nan"))
            color = MODEL_COLOURS[clf_name]

            if curve:
                ns    = [d[0] for d in curve]
                means = [d[1] for d in curve]
                stds  = [d[2] for d in curve]
                ax.plot(ns, means, "o-", color=color, lw=1.5)
                ax.fill_between(ns,
                                [m - s for m, s in zip(means, stds)],
                                [m + s for m, s in zip(means, stds)],
                                alpha=0.2, color=color)

            if not np.isnan(cv_acc):
                ax.axhline(cv_acc, color="orange", lw=1.2, ls="--")

            ax.axhline(0.8, color="grey", lw=0.6, ls=":", alpha=0.6)
            ax.axhline(0.9, color="grey", lw=0.6, ls=":", alpha=0.6)
            ax.set_xlim(0.5, 4.5)
            ax.set_ylim(0, 1.05)
            ax.set_xticks([1, 2, 3, 4])
            if col_idx == 0:
                ax.set_ylabel(src, fontsize=8)
            if row_idx == 0:
                ax.set_title(clf_name, fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "learning_curves_loso_all_sources.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Top-level plot generator
# ---------------------------------------------------------------------------

def generate_all_plots(
    per_source_results: dict,
    output_dir: str,
) -> None:
    """Generate all LOSO visualisations."""
    _ensure_dir(output_dir)

    # Combined CM figures per model
    for clf_name in ("RF", "SVM", "XGBoost"):
        try:
            plot_cm_all_sources(per_source_results, clf_name, output_dir)
        except Exception as exc:
            print(f"  [WARN] CM all sources {clf_name}: {exc}")
