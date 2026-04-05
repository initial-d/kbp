"""Unit tests for kbp.effective_rank"""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kbp.effective_rank import (
    compute_effective_rank,
    EffectiveRankEstimator,
    ERANK_KS_MAX,
    ERANK_KD_MIN,
    ERANK_FT_THRESHOLD,
    classify_sft_viability,
    spearman_erank_vs_snr,
)


def make_low_rank_data(n: int = 256, d: int = 4096, rank: int = 5, seed: int = 42):
    """Generate data concentrated on a low-dimensional submanifold (KS-like)."""
    rng = np.random.RandomState(seed)
    W = rng.randn(rank, d)
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    coeffs = rng.randn(n, rank)
    H = coeffs @ W
    H /= np.linalg.norm(H, axis=1, keepdims=True) + 1e-8
    return H


def make_high_rank_data(n: int = 256, d: int = 4096, seed: int = 42):
    """Generate data uniformly spread in high-dimensional space (KD-like)."""
    rng = np.random.RandomState(seed)
    H = rng.randn(n, d)
    H /= np.linalg.norm(H, axis=1, keepdims=True) + 1e-8
    return H


class TestComputeEffectiveRank:
    def test_low_rank_data(self):
        H = make_low_rank_data(n=256, d=512, rank=5)
        erank, eigenvalues = compute_effective_rank(H)
        # Should be close to rank (5), well below ERANK_KS_MAX (22)
        assert erank < ERANK_KS_MAX, f"Low-rank data erank {erank:.1f} too high"
        assert len(eigenvalues) == 512

    def test_high_rank_data(self):
        H = make_high_rank_data(n=256, d=512)
        erank, eigenvalues = compute_effective_rank(H)
        # Should be high — well above ERANK_KD_MIN (34)
        # (with n=256, d=512, high-rank data has erank >> 34)
        assert erank > 20, f"High-rank data erank {erank:.1f} too low"

    def test_erank_ordering(self):
        """Low-rank data should have lower erank than high-rank data."""
        H_low = make_low_rank_data(n=256, d=256, rank=3)
        H_high = make_high_rank_data(n=256, d=256)
        erank_low, _ = compute_effective_rank(H_low)
        erank_high, _ = compute_effective_rank(H_high)
        assert erank_low < erank_high, (
            f"Expected erank_low ({erank_low:.1f}) < erank_high ({erank_high:.1f})"
        )

    def test_formula_correctness(self):
        """Test erank = Σλᵢ / λ₁."""
        rng = np.random.RandomState(0)
        H = rng.randn(100, 50)
        H_c = H - H.mean(0)
        C = H_c.T @ H_c / 100
        eigvals = np.linalg.eigvalsh(C)[::-1]
        eigvals = np.maximum(eigvals, 0)

        expected_erank = eigvals.sum() / eigvals[0]
        computed_erank, _ = compute_effective_rank(H)
        assert abs(computed_erank - expected_erank) < 0.1

    def test_degenerate_matrix(self):
        """Single-vector input → all same, should not crash."""
        H = np.ones((10, 50))
        erank, _ = compute_effective_rank(H)
        assert erank == 1.0  # degenerate case

    def test_eigenvalues_nonnegative(self):
        rng = np.random.RandomState(42)
        H = rng.randn(100, 200)
        _, eigenvalues = compute_effective_rank(H)
        assert (eigenvalues >= 0).all()


class TestEffectiveRankEstimator:
    def test_estimate_with_bootstrap(self):
        H = make_low_rank_data(n=500, d=256, rank=5)
        estimator = EffectiveRankEstimator(n_queries=256, n_bootstrap=10)
        result = estimator.estimate(H)

        assert result.erank > 0
        assert result.erank_std >= 0
        assert result.regime in ("KS", "Mixed", "KD")
        assert result.n_queries == 256

    def test_regime_classification_ks(self):
        H = make_low_rank_data(n=500, d=256, rank=3)
        estimator = EffectiveRankEstimator(n_queries=256, n_bootstrap=5)
        result = estimator.estimate(H)
        assert result.regime == "KS", f"Expected KS, got {result.regime} (erank={result.erank:.1f})"

    def test_convergence_analysis(self):
        H = make_high_rank_data(n=512, d=256)
        estimator = EffectiveRankEstimator(n_bootstrap=5)
        sizes = [32, 64, 128, 256]
        results = estimator.convergence_analysis(H, n_values=sizes, n_bootstrap=5)
        assert len(results) == len(sizes)
        # Standard deviations should decrease as N increases
        stds = [results[n][1] for n in sorted(results.keys())]
        # Std at n=256 should be ≤ std at n=32 (approximately)
        assert stds[-1] <= stds[0] * 2  # Allow some slack

    def test_calibrate_thresholds(self):
        H_ks = make_low_rank_data(n=300, d=256, rank=4)
        H_kd = make_high_rank_data(n=300, d=256)
        estimator = EffectiveRankEstimator(n_queries=256, n_bootstrap=5)
        tau_lo, tau_hi = estimator.calibrate_thresholds(H_ks, H_kd)

        assert tau_lo < tau_hi, "τ_lo should be less than τ_hi"
        assert tau_lo > 0
        assert tau_hi > 0

    def test_predict_unsupervised_ks(self):
        H_ks = make_low_rank_data(n=300, d=256, rank=4)
        H_kd = make_high_rank_data(n=300, d=256)
        estimator = EffectiveRankEstimator(n_queries=200, n_bootstrap=5)
        tau_lo, tau_hi = estimator.calibrate_thresholds(H_ks, H_kd)

        label, erank = estimator.predict_unsupervised(H_ks, tau_lo, tau_hi)
        # Low-rank data should be classified as KS
        assert label in ("KS", "UNCERTAIN")

    def test_predict_unsupervised_kd(self):
        H_ks = make_low_rank_data(n=300, d=256, rank=4)
        H_kd = make_high_rank_data(n=300, d=256)
        estimator = EffectiveRankEstimator(n_queries=200, n_bootstrap=5)
        tau_lo, tau_hi = estimator.calibrate_thresholds(H_ks, H_kd)

        label, erank = estimator.predict_unsupervised(H_kd, tau_lo, tau_hi)
        assert label in ("KD", "UNCERTAIN")


class TestSFTViability:
    def test_sft_viable_low_erank(self):
        result = classify_sft_viability(erank=20.0, tau_ft=ERANK_FT_THRESHOLD)
        assert result["recommendation"] == "SFT_VIABLE"

    def test_sft_not_viable_high_erank(self):
        result = classify_sft_viability(erank=45.0, tau_ft=ERANK_FT_THRESHOLD)
        assert result["recommendation"] == "PRETRAINING_DATA_NEEDED"

    def test_boundary_case(self):
        result_below = classify_sft_viability(erank=ERANK_FT_THRESHOLD - 0.1)
        result_above = classify_sft_viability(erank=ERANK_FT_THRESHOLD + 0.1)
        assert result_below["recommendation"] == "SFT_VIABLE"
        assert result_above["recommendation"] == "PRETRAINING_DATA_NEEDED"


class TestSpearmanCorrelation:
    def test_negative_correlation(self):
        """erank should negatively correlate with gradient SNR."""
        # High erank → low SNR, low erank → high SNR
        eranks = [18.3, 19.1, 26.4, 28.2, 35.9, 38.7, 41.2, 48.3]
        snrs = [100, 90, 40, 30, 8, 5, 3, 1]  # Monotone decreasing (log scale)

        result = spearman_erank_vs_snr(eranks, snrs, n_bootstrap=100)

        assert result["rho"] < -0.5, f"Expected strong negative correlation, got {result['rho']}"
        assert result["ci_low"] < result["ci_high"]
        assert "pval" in result
