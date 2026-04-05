"""
Effective Rank Estimator
========================
Computes the effective rank of the hidden-state covariance matrix as a
label-free proxy for knowledge deficiency (H2).

Paper Equation 1:
    erank(C(ℓ)) = tr(C(ℓ)) / ‖C(ℓ)‖₂ = Σᵢλᵢ / λ₁

Paper Section 6.2:
    "Knowledge-deficient tasks show systematically higher effective rank
     than knowledge-sufficient tasks."

    Spearman correlation with gradient SNR: ρs ≈ −0.82 (per model)

Paper Appendix H (Theoretical Note):
    Low erank → distribution concentrated on a low-dimensional submanifold
               → coherent knowledge retrieval pathway
    High erank → diffuse activation across weight columns
               → knowledge-deficient confabulation

Reference: Roy & Vetterli (2007) "The effective rank: A measure of
           effective dimensionality." EUSIPCO.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# Regime thresholds from Table 3 (for 8B-scale models)
ERANK_KS_MAX = 22.0       # Knowledge-Sufficient: erank ≤ 22
ERANK_KD_MIN = 34.0       # Knowledge-Deficient:  erank > 34
ERANK_FT_THRESHOLD = 38.0  # Fine-tuning investment threshold (Section 7.3)


@dataclass
class EffectiveRankResult:
    """Result of effective rank computation for one task."""

    erank: float
    """Mean effective rank over bootstrap samples."""

    erank_std: float
    """Standard deviation over bootstrap samples."""

    regime: str
    """'KS', 'KD', or 'Mixed' based on paper Table 3 thresholds."""

    n_queries: int
    """Number of queries used."""

    eigenvalues: Optional[np.ndarray] = None
    """Top eigenvalues of the covariance matrix."""

    @property
    def sft_recommended(self) -> bool:
        """
        True if SFT is expected to yield usable gradient signal.

        Paper Section 7.3:
          "if erank > τft (8B-scale: τft=38), prioritize pretraining-data
           supplementation over SFT; otherwise, SFT signal is sufficient."
        """
        return self.erank <= ERANK_FT_THRESHOLD


def compute_effective_rank(H: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute the effective rank of the covariance matrix of H.

    erank(C) = tr(C) / ‖C‖₂ = Σᵢλᵢ / λ₁

    Parameters
    ----------
    H : (N, d) array of ℓ2-normalized hidden states

    Returns
    -------
    erank : float
    eigenvalues : (d,) array, sorted descending
    """
    # Center the hidden states
    H_centered = H - H.mean(axis=0, keepdims=True)

    # Covariance matrix C = (1/N) HᵀH, shape (d, d)
    N = H_centered.shape[0]
    C = (H_centered.T @ H_centered) / N

    # Eigenvalues (use eigvalsh for symmetric matrix, faster)
    eigenvalues = np.linalg.eigvalsh(C)  # ascending order
    eigenvalues = eigenvalues[::-1]      # descending order
    eigenvalues = np.maximum(eigenvalues, 0.0)  # numerical stability

    # erank = trace / spectral norm = Σλᵢ / λ₁
    trace = eigenvalues.sum()
    lambda_1 = eigenvalues[0]

    if lambda_1 < 1e-10:
        logger.warning("Degenerate covariance matrix (λ₁ ≈ 0); returning erank=1.0")
        return 1.0, eigenvalues

    erank = trace / lambda_1
    return float(erank), eigenvalues


class EffectiveRankEstimator:
    """
    Estimates effective rank with bootstrap uncertainty quantification.

    Paper Appendix E:
      "The estimate stabilizes at N ≈ 128; our default of N = 256
       provides a comfortable margin."
      "Kendall's τ between rankings at ℓ* and adjacent layers exceeds
       0.91 for both models."

    Example
    -------
    >>> estimator = EffectiveRankEstimator(n_queries=256, n_bootstrap=20)
    >>> result = estimator.estimate(hidden_states)
    >>> print(f"erank = {result.erank:.1f} ± {result.erank_std:.1f}")
    >>> print(f"Regime: {result.regime}")
    >>> print(f"SFT recommended: {result.sft_recommended}")
    """

    def __init__(
        self,
        n_queries: int = 256,
        n_bootstrap: int = 20,
        random_state: int = 42,
    ):
        self.n_queries = n_queries
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.RandomState(random_state)

    def estimate(self, H: np.ndarray) -> EffectiveRankResult:
        """
        Estimate effective rank with bootstrap confidence.

        Parameters
        ----------
        H : (N, d) array of ℓ2-normalized hidden states.
            If N > n_queries, subsample to n_queries.

        Returns
        -------
        EffectiveRankResult
        """
        N = H.shape[0]
        n = min(N, self.n_queries)

        bootstrap_eranks = []
        for _ in range(self.n_bootstrap):
            indices = self.rng.choice(N, size=n, replace=True)
            H_boot = H[indices]
            erank, _ = compute_effective_rank(H_boot)
            bootstrap_eranks.append(erank)

        # Full estimate
        if N > n:
            indices = self.rng.choice(N, size=n, replace=False)
            H_sub = H[indices]
        else:
            H_sub = H
        erank, eigenvalues = compute_effective_rank(H_sub)

        erank_mean = float(np.mean(bootstrap_eranks))
        erank_std = float(np.std(bootstrap_eranks))
        regime = self._classify_regime(erank_mean)

        return EffectiveRankResult(
            erank=erank_mean,
            erank_std=erank_std,
            regime=regime,
            n_queries=n,
            eigenvalues=eigenvalues[:50],  # store top-50
        )

    @staticmethod
    def _classify_regime(erank: float) -> str:
        """Classify into KS / Mixed / KD regime per Table 3 thresholds."""
        if erank <= ERANK_KS_MAX:
            return "KS"
        elif erank > ERANK_KD_MIN:
            return "KD"
        else:
            return "Mixed"

    def calibrate_thresholds(
        self,
        H_ks: np.ndarray,
        H_kd: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Calibrate τ_lo and τ_hi from reference KS and KD hidden states.

        Implements Algorithm 1 / Appendix F.2 threshold calibration:
            τ_lo = r_ref,hi + 0.4 × (r_ref,lo − r_ref,hi)
            τ_hi = r_ref,hi + 0.7 × (r_ref,lo − r_ref,hi)

        Parameters
        ----------
        H_ks : (N, d) hidden states from knowledge-sufficient queries (e.g., PopQA-High)
        H_kd : (N, d) hidden states from knowledge-deficient queries (e.g., PopQA-Low)

        Returns
        -------
        tau_lo, tau_hi : calibrated thresholds
        """
        result_ks = self.estimate(H_ks)
        result_kd = self.estimate(H_kd)

        r_ref_hi = result_ks.erank   # from high-frequency (KS) reference
        r_ref_lo = result_kd.erank   # from low-frequency (KD) reference

        tau_lo = r_ref_hi + 0.4 * (r_ref_lo - r_ref_hi)
        tau_hi = r_ref_hi + 0.7 * (r_ref_lo - r_ref_hi)

        logger.info(
            f"Calibrated thresholds: τ_lo={tau_lo:.2f}, τ_hi={tau_hi:.2f} "
            f"(r_ref,KS={r_ref_hi:.2f}, r_ref,KD={r_ref_lo:.2f})"
        )
        return tau_lo, tau_hi

    def predict_unsupervised(
        self,
        H: np.ndarray,
        tau_lo: float,
        tau_hi: float,
    ) -> Tuple[str, float]:
        """
        Classify a task as KS / KD / Uncertain (Algorithm 1, Step 3b).

        Parameters
        ----------
        H : (N, d) hidden states for pilot queries
        tau_lo, tau_hi : calibrated thresholds

        Returns
        -------
        label : 'KS', 'KD', or 'UNCERTAIN'
        erank : estimated effective rank
        """
        result = self.estimate(H)
        erank = result.erank

        if erank < tau_lo:
            return "KS", erank
        elif erank > tau_hi:
            return "KD", erank
        else:
            return "UNCERTAIN", erank

    def convergence_analysis(
        self,
        H: np.ndarray,
        n_values: Optional[List[int]] = None,
        n_bootstrap: int = 20,
    ) -> Dict[int, Tuple[float, float]]:
        """
        Reproduce Figure 7: effective rank estimate vs. number of queries.

        Returns
        -------
        dict mapping n → (mean_erank, std_erank)
        """
        if n_values is None:
            n_values = [32, 64, 128, 256, 512, min(1024, len(H))]

        results = {}
        N = len(H)
        for n in n_values:
            n = min(n, N)
            eranks = []
            for _ in range(n_bootstrap):
                idx = self.rng.choice(N, size=n, replace=True)
                erank, _ = compute_effective_rank(H[idx])
                eranks.append(erank)
            results[n] = (float(np.mean(eranks)), float(np.std(eranks)))
            logger.debug(
                f"  N={n:5d}: erank={results[n][0]:.2f}±{results[n][1]:.2f}"
            )
        return results

    def layer_stability_analysis(
        self,
        hidden_states: Dict[int, np.ndarray],
        best_layer: int,
        window: int = 3,
    ) -> Dict[Tuple[int, int], float]:
        """
        Reproduce Table 13: Kendall's τ for task effective rank ordering
        across layers near ℓ*.

        Paper Appendix E:
          "Kendall's τ between rankings at ℓ* and adjacent layers exceeds
           0.91 for both models."

        Parameters
        ----------
        hidden_states : dict of layer → (N, d) arrays, for each task
        best_layer : int
        window : int, check layers ℓ* ± window

        Returns
        -------
        dict of (layer_a, layer_b) → kendall_tau
        """
        # Estimate erank for each task at each layer
        eranks_by_layer = {}
        task_ids = list(hidden_states.keys())  # tasks

        for layer in [best_layer - window, best_layer, best_layer + window]:
            eranks = []
            for task_id in task_ids:
                if layer in hidden_states[task_id]:
                    result = self.estimate(hidden_states[task_id][layer])
                    eranks.append(result.erank)
            eranks_by_layer[layer] = eranks

        results = {}
        for other_layer in [best_layer - window, best_layer + window]:
            tau, pval = stats.kendalltau(
                eranks_by_layer[best_layer], eranks_by_layer[other_layer]
            )
            results[(other_layer, best_layer)] = float(tau)
            logger.info(f"Kendall τ between layer {other_layer} and {best_layer}: {tau:.3f}")

        return results


def compute_gradient_snr(
    model: "torch.nn.Module",
    X_train: "torch.Tensor",
    y_train: "torch.Tensor",
    n_steps: int = 5,
    batch_size: int = 8,
) -> float:
    """
    Compute gradient signal-to-noise ratio in the early training regime.

    Paper Section 5.3:
        SNR = ‖ḡ‖²_F / (1/T Σ_t ‖∇L_t − ḡ‖²_F)

    where ḡ is the mean gradient over T steps and ∇L_t is the per-step gradient.

    Parameters
    ----------
    model : nn.Module (a small fine-tuning adapter or the full model)
    X_train : (N, d) input tensor
    y_train : (N,) label tensor
    n_steps : int, T in the formula (paper: 5 for main, 50 for robustness)
    batch_size : int (paper: 8)

    Returns
    -------
    snr : float
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        raise ImportError("PyTorch is required for compute_gradient_snr. Install with: pip install torch")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    gradients = []  # list of flattened gradient vectors
    N = X_train.shape[0]

    for step in range(n_steps):
        idx = torch.randperm(N)[:batch_size]
        x_batch = X_train[idx]
        y_batch = y_train[idx]

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = F.cross_entropy(logits, y_batch)
        loss.backward()

        # Collect gradient vector
        grad_vec = []
        for param in model.parameters():
            if param.grad is not None:
                grad_vec.append(param.grad.detach().cpu().flatten())
        gradients.append(torch.cat(grad_vec))

    gradients = torch.stack(gradients)  # (T, P)
    g_bar = gradients.mean(dim=0)      # mean gradient
    noise = gradients - g_bar.unsqueeze(0)  # (T, P)

    signal_sq = (g_bar ** 2).sum()
    noise_sq = (noise ** 2).sum() / n_steps

    snr = (signal_sq / (noise_sq + 1e-10)).item()
    model.eval()
    return float(snr)


def spearman_erank_vs_snr(
    eranks: List[float],
    snrs: List[float],
    n_bootstrap: int = 10000,
) -> Dict[str, float]:
    """
    Compute Spearman correlation between effective rank and gradient SNR.

    Reproduces the correlation reported in Figure 2 and Section 6.2.

    Paper: ρs = −0.81 (Llama-3-8B), −0.82 (Qwen3-8B) per model (N=12 tasks)
           ρs = −0.83 pooled (N=24 task-model pairs)
           95% bootstrap CI: [−0.91, −0.71]

    Parameters
    ----------
    eranks : list of float
    snrs : list of float (on log scale for the OLS fit)
    n_bootstrap : int, for CI estimation

    Returns
    -------
    dict with 'rho', 'pval', 'ci_low', 'ci_high'
    """
    eranks = np.array(eranks)
    snrs = np.array(snrs)

    rho, pval = stats.spearmanr(eranks, snrs)

    # Block bootstrap CI (treat pairs as the unit)
    rng = np.random.RandomState(42)
    boot_rhos = []
    N = len(eranks)
    for _ in range(n_bootstrap):
        idx = rng.choice(N, size=N, replace=True)
        boot_rho, _ = stats.spearmanr(eranks[idx], snrs[idx])
        boot_rhos.append(boot_rho)

    ci_low = float(np.percentile(boot_rhos, 2.5))
    ci_high = float(np.percentile(boot_rhos, 97.5))

    logger.info(
        f"Spearman ρs = {rho:.3f} (p={pval:.4f}), "
        f"95% CI [{ci_low:.3f}, {ci_high:.3f}]"
    )

    return {
        "rho": float(rho),
        "pval": float(pval),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "boot_rhos": boot_rhos,
    }


def classify_sft_viability(
    erank: float,
    tau_ft: float = ERANK_FT_THRESHOLD,
) -> Dict[str, object]:
    """
    Classify whether SFT is viable based on effective rank.

    Paper Section 7.3:
      "if erank > τft (8B-scale: τft=38), prioritize pretraining-data
       supplementation over SFT; otherwise, SFT signal is sufficient."

    Returns
    -------
    dict with 'recommendation', 'erank', 'tau_ft', 'confidence'
    """
    if erank <= tau_ft:
        recommendation = "SFT_VIABLE"
        rationale = (
            f"erank={erank:.1f} ≤ τft={tau_ft}: model has coherent retrieval "
            "pathways; gradient signal is sufficient for convergence."
        )
    else:
        recommendation = "PRETRAINING_DATA_NEEDED"
        rationale = (
            f"erank={erank:.1f} > τft={tau_ft}: activation dispersion is too "
            "high; SFT gradient signal is likely too noisy. Prioritize adding "
            "pretraining data for this domain."
        )

    return {
        "recommendation": recommendation,
        "erank": erank,
        "tau_ft": tau_ft,
        "rationale": rationale,
    }
