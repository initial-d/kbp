"""
Evaluation Metrics and Visualization
=====================================
Utilities for computing and plotting paper metrics:
  - Layer-wise AUROC curves (Figure 1)
  - Effective rank vs. gradient SNR scatter (Figure 2)
  - Detection variance vs. boundary distance (Figure 3)
  - Probe AUROC vs. training size (Figure 6)
  - Cross-model AUROC vs. alignment size (Figure 8)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# AUROC utilities
# ------------------------------------------------------------------

def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute Area Under the ROC Curve."""
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


def compute_auroc_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Compute AUROC with bootstrap confidence interval.

    Returns
    -------
    (auroc, ci_low, ci_high)
    """
    from sklearn.metrics import roc_auc_score

    auroc = float(roc_auc_score(labels, scores))
    rng = np.random.RandomState(42)

    boot_aurocs = []
    N = len(scores)
    for _ in range(n_bootstrap):
        idx = rng.choice(N, size=N, replace=True)
        try:
            boot_auroc = roc_auc_score(labels[idx], scores[idx])
            boot_aurocs.append(boot_auroc)
        except ValueError:
            pass  # Skip if only one class in bootstrap sample

    alpha = 1 - confidence
    ci_low = float(np.percentile(boot_aurocs, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_aurocs, 100 * (1 - alpha / 2)))
    return auroc, ci_low, ci_high


def paired_ttest_auroc(
    scores_a: List[float],
    scores_b: List[float],
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """
    Paired t-test comparing two sets of per-seed AUROC values.

    Used in Table 1: "‡ denotes p < 0.001 vs. Truthfulness Probe
    (paired t-test across 5 seed pairs, Bonferroni-corrected)."
    """
    t_stat, p_val = stats.ttest_rel(scores_a, scores_b, alternative=alternative)
    return float(t_stat), float(p_val)


# ------------------------------------------------------------------
# Boundary distance analysis (Section 6.3, Figure 3)
# ------------------------------------------------------------------

def compute_detection_variance_vs_margin(
    probe_scores: np.ndarray,
    margins: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute detection variance across paraphrases as a function of
    decision boundary margin.

    Paper Section 6.3:
      "Detection variance is highest for queries near the boundary
       (margin < 0.5) and decays monotonically."

    Parameters
    ----------
    probe_scores : (N, K) — scores for N queries across K paraphrases
    margins : (N,) — geometric distances to decision boundary

    Returns
    -------
    dict with 'margins', 'variances', 'spearman_rho', 'spearman_pval'
    """
    # Variance across K paraphrases per query
    variances = probe_scores.var(axis=1)  # (N,)

    # Spearman correlation between margin and variance
    rho, pval = stats.spearmanr(margins, variances)

    logger.info(
        f"Boundary distance analysis: ρs={rho:.3f} (p={pval:.4e}), "
        f"N={len(margins)} queries"
    )

    return {
        "margins": margins,
        "variances": variances,
        "spearman_rho": float(rho),
        "spearman_pval": float(pval),
    }


# ------------------------------------------------------------------
# Plotting (matplotlib)
# ------------------------------------------------------------------

def plot_layerwise_auroc(
    layer_results: Dict[str, Dict[int, Tuple[float, float]]],
    total_layers: int,
    best_baseline_auroc: float = 80.3,
    save_path: Optional[str] = None,
    title: str = "Layer-wise Probe AUROC (PopQA)",
) -> None:
    """
    Reproduce Figure 1: layer-wise AUROC curves.

    Parameters
    ----------
    layer_results : dict of model_name → {layer: (mean_auroc, std_auroc)}
    total_layers : int (for normalized depth x-axis)
    best_baseline_auroc : float (dashed line, paper: 80.3 for Truthfulness Probe)
    save_path : str, optional
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        logger.warning("matplotlib not installed. Cannot plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    markers = ["o", "s", "^", "D"]

    for (model_name, results), color, marker in zip(
        layer_results.items(), colors, markers
    ):
        layers = sorted(results.keys())
        depths = [l / total_layers for l in layers]
        means = [results[l][0] for l in layers]
        stds = [results[l][1] for l in layers]

        ax.plot(
            depths,
            means,
            label=model_name,
            color=color,
            marker=marker,
            markersize=4,
            linewidth=2,
        )
        ax.fill_between(
            depths,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.15,
            color=color,
        )

    # Best baseline dashed line
    ax.axhline(
        y=best_baseline_auroc,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"Best baseline (Truth. Probe, AUROC={best_baseline_auroc})",
    )

    # Shaded region: 60-75% depth (paper-predicted peak)
    ax.axvspan(0.60, 0.75, alpha=0.08, color="gold", label="60–75% depth (predicted peak)")

    ax.set_xlabel("Layer index (normalized)", fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.45, 0.95)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Figure saved to {save_path}")
    plt.show()


def plot_erank_vs_snr(
    task_names: List[str],
    eranks: List[float],
    snrs: List[float],
    model_labels: Optional[List[str]] = None,
    spearman_rhos: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Reproduce Figure 2: Activation effective rank vs. gradient SNR.

    Paper: ρs = -0.81 (Llama), -0.82 (Qwen), pooled -0.83.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib not installed. Cannot plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    # Color code by regime
    from kbp.effective_rank import ERANK_KS_MAX, ERANK_KD_MIN
    regime_colors = {"KS": "#2196F3", "Mixed": "#FF9800", "KD": "#F44336"}

    for i, (name, er, snr) in enumerate(zip(task_names, eranks, snrs)):
        if er <= ERANK_KS_MAX:
            color = regime_colors["KS"]
        elif er > ERANK_KD_MIN:
            color = regime_colors["KD"]
        else:
            color = regime_colors["Mixed"]

        marker = "o" if (model_labels is None or model_labels[i] == "Llama-3-8B") else "s"
        ax.scatter(er, snr, c=color, marker=marker, s=60, zorder=3, alpha=0.85)
        ax.annotate(
            name, (er, snr), textcoords="offset points", xytext=(5, 3), fontsize=7
        )

    # OLS log-linear fit
    log_snrs = np.log(np.maximum(snrs, 1e-6))
    z = np.polyfit(eranks, log_snrs, 1)
    x_fit = np.linspace(min(eranks) - 1, max(eranks) + 1, 100)
    y_fit = np.exp(np.polyval(z, x_fit))
    ax.plot(x_fit, y_fit, "k--", linewidth=1.2, alpha=0.6, label="OLS log-linear fit")

    # Regime boundary lines
    ax.axvline(x=ERANK_KS_MAX, color="gray", linestyle=":", linewidth=1)
    ax.axvline(x=ERANK_KD_MIN, color="gray", linestyle=":", linewidth=1)

    # Legend
    legend_elements = [
        mpatches.Patch(color=regime_colors["KS"], label="Knowledge-Sufficient"),
        mpatches.Patch(color=regime_colors["Mixed"], label="Mixed"),
        mpatches.Patch(color=regime_colors["KD"], label="Knowledge-Deficient"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)

    ax.set_yscale("log")
    ax.set_xlabel("Activation Effective Rank", fontsize=12)
    ax.set_ylabel("Gradient SNR (log scale)", fontsize=12)

    if spearman_rhos:
        rho_str = ", ".join(f"{k}: ρs={v:.2f}" for k, v in spearman_rhos.items())
        ax.set_title(f"Effective Rank vs. Gradient SNR\n({rho_str})", fontsize=11)
    else:
        ax.set_title("Activation Effective Rank vs. Gradient SNR", fontsize=12)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Figure saved to {save_path}")
    plt.show()


def plot_boundary_distance_vs_variance(
    margins: np.ndarray,
    variances: np.ndarray,
    model_names: Optional[List[str]] = None,
    spearman_rhos: Optional[List[float]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Reproduce Figure 3: detection variance vs. boundary distance."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    colors = ["#2196F3", "#FF5722"]
    if isinstance(margins[0], np.ndarray):
        # Multiple models
        for i, (m, v) in enumerate(zip(margins, variances)):
            label = model_names[i] if model_names else f"Model {i}"
            rho_str = f" (ρs={spearman_rhos[i]:.2f})" if spearman_rhos else ""
            ax.scatter(m, v, alpha=0.3, s=10, c=colors[i], label=label + rho_str)
    else:
        ax.scatter(margins, variances, alpha=0.3, s=10, c=colors[0])

    ax.set_xlabel("Distance to decision boundary (margin)", fontsize=12)
    ax.set_ylabel("Detection variance across paraphrases", fontsize=12)
    ax.set_title("Prompt Sensitivity vs. Boundary Distance", fontsize=12)
    if model_names:
        ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def print_results_table(
    results: Dict[str, float],
    title: str = "AUROC Results",
    bold_key: Optional[str] = None,
) -> None:
    """Pretty-print a results table."""
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")
    for method, auroc in sorted(results.items(), key=lambda x: -x[1]):
        prefix = "→ " if method == bold_key else "  "
        print(f"{prefix}{method:<35} {auroc:.4f}")
    print(f"{'─' * 50}\n")
