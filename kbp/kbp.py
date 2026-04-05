"""
Knowledge Boundary Probe — Main Pipeline
=========================================
Implements the full KBP inference pipeline (Algorithm 1 in the paper).

Two modes:
  1. Supervised  — logistic regression probe w (requires labeled pilot data)
  2. Unsupervised — effective rank r = erank(C(ℓ*)) (256 unlabeled queries)

Paper Figure 4 (KBP inference pipeline):
  Query q
    → LLM Forward Pass
    → Layer hook at ℓ* → h(ℓ*)(q) ∈ ℝᵈ
    → ℓ2-Normalize → h̃
    → [Supervised]   s = σ(wᵀh̃ + b)   → KS / KD + margin
    → [Unsupervised] r = erank(C(ℓ*)) → KS / KD / UNCERTAIN
  <0.3% latency overhead
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore

from kbp.effective_rank import (
    ERANK_FT_THRESHOLD,
    EffectiveRankEstimator,
    EffectiveRankResult,
    classify_sft_viability,
)
from kbp.extractor import ExtractionConfig, HiddenStateExtractor, HiddenStateOutput
from kbp.probe import LayerWiseProbeTrainer, LinearProbe

logger = logging.getLogger(__name__)


@dataclass
class KBPResult:
    """
    Result from a single KBP inference call.

    Fields
    ------
    label : str
        "KNOWLEDGE_SUFFICIENT", "KNOWLEDGE_DEFICIENT", or "UNCERTAIN"
    score : float
        Supervised mode: P(KS) ∈ [0, 1]
        Unsupervised mode: None
    margin : float
        Supervised mode: geometric distance to decision boundary.
        Correlates with prompt sensitivity (low margin → high sensitivity).
        Unsupervised mode: None
    erank : float
        Unsupervised mode: estimated effective rank.
        Supervised mode: None
    mode : str
        "supervised" or "unsupervised"
    """

    label: str
    score: Optional[float] = None
    margin: Optional[float] = None
    erank: Optional[float] = None
    mode: str = "supervised"

    @property
    def is_knowledge_sufficient(self) -> bool:
        return self.label == "KNOWLEDGE_SUFFICIENT"

    @property
    def is_knowledge_deficient(self) -> bool:
        return self.label == "KNOWLEDGE_DEFICIENT"

    @property
    def should_retrieve(self) -> bool:
        """True if RAG retrieval is recommended."""
        return self.label in ("KNOWLEDGE_DEFICIENT", "UNCERTAIN")

    def __repr__(self) -> str:
        if self.mode == "supervised":
            return (
                f"KBPResult(label={self.label!r}, "
                f"score={self.score:.3f}, margin={self.margin:.3f})"
            )
        else:
            return (
                f"KBPResult(label={self.label!r}, "
                f"erank={self.erank:.2f})"
            )


class KBP:
    """
    Knowledge Boundary Probe.

    Detects whether an LLM query lies within the model's parametric knowledge,
    using the geometry of intermediate-layer hidden states.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model identifier.
    best_layer : int, optional
        ℓ* — the probe layer. If None, determined via layer-wise sweep.
    mode : str
        "supervised" (default) or "unsupervised".
    extraction_config : ExtractionConfig, optional

    Example
    -------
    >>> kbp = KBP.from_pretrained("meta-llama/Meta-Llama-3-8B")
    >>> kbp.fit(train_queries, train_labels)
    >>> result = kbp.predict("Who wrote the novel 1984?")
    >>> print(result.label, result.score)
    """

    # Paper-reported best layers: layer 23 for both Llama-3-8B and Qwen3-8B
    KNOWN_BEST_LAYERS = {
        "llama-3-8b": 23,
        "llama3": 23,
        "qwen3-8b": 23,
        "qwen3": 23,
    }

    def __init__(
        self,
        model_name_or_path: str,
        best_layer: Optional[int] = None,
        mode: str = "supervised",
        extraction_config: Optional[ExtractionConfig] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name_or_path
        self.mode = mode
        self.extraction_config = extraction_config or ExtractionConfig()

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Lazy-loaded components
        self._extractor: Optional[HiddenStateExtractor] = None
        self._probe: Optional[LinearProbe] = None
        self._erank_estimator: Optional[EffectiveRankEstimator] = None
        self._tau_lo: Optional[float] = None
        self._tau_hi: Optional[float] = None
        self._best_layer: Optional[int] = best_layer

        self._is_fitted = False

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        mode: str = "supervised",
        device: Optional[str] = None,
        **kwargs,
    ) -> "KBP":
        """
        Create a KBP instance from a pretrained model.

        The model weights are loaded lazily on first extraction call.
        """
        instance = cls(model_name_or_path, mode=mode, device=device, **kwargs)
        return instance

    @property
    def extractor(self) -> HiddenStateExtractor:
        """Lazy-load the hidden state extractor."""
        if self._extractor is None:
            self._extractor = HiddenStateExtractor(
                self.model_name,
                config=self.extraction_config,
                device=self.device,
            )
        return self._extractor

    @property
    def best_layer(self) -> int:
        """Return ℓ* — must be set via fit() or explicitly."""
        if self._best_layer is None:
            raise RuntimeError(
                "best_layer is not set. Call fit() first or pass best_layer "
                "to the constructor."
            )
        return self._best_layer

    # ------------------------------------------------------------------
    # Supervised mode
    # ------------------------------------------------------------------

    def fit(
        self,
        queries: List[str],
        labels: List[int],
        sweep_layers: bool = False,
        layer_range: Optional[Tuple[int, int]] = None,
        n_seeds: int = 5,
    ) -> "KBP":
        """
        Train the KBP probe.

        Parameters
        ----------
        queries : list of str
        labels : list of int — 1=KS (knowledge-sufficient), 0=KD (deficient)
        sweep_layers : bool
            If True, run the full layer-wise sweep to find ℓ* (slower).
            If False, use the paper-recommended layer for this model.
        layer_range : (start, end), optional
            Layer range for sweep. Defaults to ExtractionConfig range.
        n_seeds : int
        """
        labels_arr = np.array(labels)

        if not sweep_layers:
            # Use paper-recommended layer
            self._best_layer = self._get_default_best_layer()
            logger.info(f"Using paper-recommended ℓ* = {self._best_layer}")

            # Extract hidden states at best layer only
            layers = [self._best_layer]
            output = self.extractor.extract(queries, layers=layers)
            X = output.get_layer(self._best_layer).numpy()

            # Train probe with 70/30 split, averaged over seeds
            trainer = LayerWiseProbeTrainer(n_seeds=n_seeds)
            probe, auroc = trainer.fit_best_probe(
                {self._best_layer: X}, labels_arr, self._best_layer
            )
            self._probe = probe
            logger.info(f"Probe trained: AUROC={auroc:.4f}")
        else:
            # Full layer-wise sweep
            start = layer_range[0] if layer_range else self.extraction_config.layer_start
            end = layer_range[1] if layer_range else self.extraction_config.layer_end
            layers = list(range(start, end + 1))

            output = self.extractor.extract(queries, layers=layers)
            hidden_states = {
                l: output.get_layer(l).numpy() for l in layers
            }

            trainer = LayerWiseProbeTrainer(n_seeds=n_seeds)
            n_layers = self.extractor.n_layers
            lw_results = trainer.fit_all_layers(
                hidden_states, labels_arr, total_layers=n_layers
            )
            self._best_layer, best_auroc = lw_results.best_layer()
            logger.info(
                f"Layer sweep complete. Best layer: {self._best_layer} "
                f"(AUROC={best_auroc:.4f})"
            )
            print(lw_results.summary_table())

            # Fit final probe at best layer
            probe, _ = trainer.fit_best_probe(
                hidden_states, labels_arr, self._best_layer
            )
            self._probe = probe

        self._is_fitted = True
        return self

    def predict(self, query: Union[str, List[str]]) -> Union[KBPResult, List[KBPResult]]:
        """
        Predict knowledge sufficiency for one or more queries.

        Returns KBPResult (or list of KBPResult for batch input).
        """
        if isinstance(query, str):
            results = self._predict_batch([query])
            return results[0]
        else:
            return self._predict_batch(query)

    def _predict_batch(self, queries: List[str]) -> List[KBPResult]:
        if self.mode == "supervised":
            return self._predict_supervised(queries)
        else:
            return self._predict_unsupervised(queries)

    def _predict_supervised(self, queries: List[str]) -> List[KBPResult]:
        self._check_fitted()
        output = self.extractor.extract(queries, layers=[self.best_layer])
        X = output.get_layer(self.best_layer).numpy()

        scores = self._probe.predict_proba(X)
        margins = self._probe.margin(X)

        results = []
        for i, (score, margin) in enumerate(zip(scores, margins)):
            label = (
                "KNOWLEDGE_SUFFICIENT" if score >= 0.5 else "KNOWLEDGE_DEFICIENT"
            )
            results.append(
                KBPResult(
                    label=label,
                    score=float(score),
                    margin=float(margin),
                    mode="supervised",
                )
            )
        return results

    def _predict_unsupervised(self, queries: List[str]) -> List[KBPResult]:
        """
        Unsupervised prediction: compute erank for the query set.

        Note: erank is a *task-level* metric, not per-query.
        All queries in the batch receive the same label.
        """
        if self._tau_lo is None or self._tau_hi is None:
            raise RuntimeError(
                "Unsupervised mode requires calibration. Call calibrate() first."
            )

        estimator = self._get_erank_estimator()
        output = self.extractor.extract(queries, layers=[self.best_layer])
        H = output.get_layer(self.best_layer).numpy()

        label, erank = estimator.predict_unsupervised(H, self._tau_lo, self._tau_hi)

        return [
            KBPResult(label=label, erank=float(erank), mode="unsupervised")
            for _ in queries
        ]

    def calibrate(
        self,
        pilot_queries: List[str],
        reference_ks_queries: Optional[List[str]] = None,
        reference_kd_queries: Optional[List[str]] = None,
    ) -> "KBP":
        """
        Calibrate unsupervised thresholds τ_lo and τ_hi.

        Paper Appendix F.2:
          Thresholds calibrated on a small unlabeled pilot set (256 queries)
          by comparing to reference effective ranks from PopQA high- and
          low-frequency splits.

        Parameters
        ----------
        pilot_queries : list of str (256 queries from target domain)
        reference_ks_queries : list of str
            Unlabeled queries from a knowledge-sufficient reference (e.g., PopQA-High).
            If None, uses the pilot queries with a default threshold.
        reference_kd_queries : list of str
            Unlabeled queries from a knowledge-deficient reference (e.g., PopQA-Low).
        """
        estimator = self._get_erank_estimator()

        if reference_ks_queries is not None and reference_kd_queries is not None:
            output_ks = self.extractor.extract(
                reference_ks_queries, layers=[self.best_layer]
            )
            output_kd = self.extractor.extract(
                reference_kd_queries, layers=[self.best_layer]
            )
            H_ks = output_ks.get_layer(self.best_layer).numpy()
            H_kd = output_kd.get_layer(self.best_layer).numpy()
            self._tau_lo, self._tau_hi = estimator.calibrate_thresholds(H_ks, H_kd)
        else:
            # Default thresholds based on paper Table 3 (8B-scale models)
            logger.warning(
                "No reference queries provided. Using paper-default thresholds "
                f"τ_lo={ERANK_KS_MAX}, τ_hi={34.0} (Table 3, 8B-scale models)."
            )
            self._tau_lo = ERANK_KS_MAX
            self._tau_hi = 34.0

        if self._best_layer is None:
            self._best_layer = self._get_default_best_layer()

        self._is_fitted = True
        return self

    def assess_sft_viability(
        self,
        queries: List[str],
        tau_ft: float = ERANK_FT_THRESHOLD,
    ) -> Dict:
        """
        Predict whether SFT will yield usable gradient signal on a new task.

        Paper Section 7.3:
          "if erank > τft (8B-scale: τft=38), prioritize pretraining-data
           supplementation over SFT; otherwise, SFT signal is sufficient."

        Paper result: 11/12 tasks correctly classified (binomial p=0.003).

        Parameters
        ----------
        queries : list of str — at least 256 queries from target task
        tau_ft : float — fine-tuning viability threshold

        Returns
        -------
        dict with 'recommendation', 'erank', 'tau_ft', 'rationale'
        """
        if self._best_layer is None:
            self._best_layer = self._get_default_best_layer()

        estimator = self._get_erank_estimator()
        output = self.extractor.extract(queries[:256], layers=[self.best_layer])
        H = output.get_layer(self.best_layer).numpy()

        result = estimator.estimate(H)
        decision = classify_sft_viability(result.erank, tau_ft)
        decision["erank_std"] = result.erank_std
        return decision

    # ------------------------------------------------------------------
    # Cross-model Procrustes alignment (Appendix O)
    # ------------------------------------------------------------------

    def align_to(
        self,
        target_kbp: "KBP",
        alignment_queries: List[str],
    ) -> "LinearProbe":
        """
        Align this probe's weight vector to another model's hidden-state space
        via Procrustes analysis (Appendix O).

        Paper Appendix O.2:
          "After Procrustes alignment on 200 queries, AUROC recovers to
           77.8–79.4 — within 8–9 points of the in-domain ceiling and well
           above all output-layer baselines."

        Parameters
        ----------
        target_kbp : KBP instance for the target model
        alignment_queries : list of str (≥150 unlabeled queries)

        Returns
        -------
        Aligned LinearProbe for the target model.
        """
        from scipy.linalg import orthogonal_procrustes

        self._check_fitted()

        logger.info(
            f"Procrustes alignment with {len(alignment_queries)} queries..."
        )

        # Extract hidden states from both models
        src_output = self.extractor.extract(
            alignment_queries, layers=[self.best_layer]
        )
        tgt_output = target_kbp.extractor.extract(
            alignment_queries, layers=[target_kbp.best_layer]
        )

        H_src = src_output.get_layer(self.best_layer).numpy()  # (N, d)
        H_tgt = tgt_output.get_layer(target_kbp.best_layer).numpy()  # (N, d)

        # R* = argmin_{RᵀR=I} Σ ‖H_tgt_i − R H_src_i‖²
        R, _ = orthogonal_procrustes(H_src, H_tgt)  # (d, d)

        # Aligned probe: ϕ(h_tgt) = σ(wᵀ R*ᵀ h̃_tgt + b)
        # Equivalent to rotating the weight vector: w_aligned = R w
        w_src = self._probe.weight_vector()
        b_src = self._probe.bias()

        w_aligned = R.T @ w_src  # (d,)

        # Build an aligned probe with the rotated weight
        aligned_probe = LinearProbe(
            C=self._probe.C,
            normalization=self._probe.normalization,
        )
        # Manually set the fitted classifier
        from sklearn.linear_model import LogisticRegression as LR
        clf = LR(C=self._probe.C, solver="lbfgs", max_iter=1)
        # Create a mock fitted classifier
        clf.classes_ = np.array([0, 1])
        clf.coef_ = w_aligned.reshape(1, -1)
        clf.intercept_ = np.array([b_src])
        clf.n_iter_ = np.array([1])
        aligned_probe._clf = clf
        aligned_probe._is_fitted = True

        logger.info("Procrustes alignment complete.")
        return aligned_probe

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save KBP state (probe + metadata) to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        state = {
            "model_name": self.model_name,
            "best_layer": self._best_layer,
            "mode": self.mode,
            "tau_lo": self._tau_lo,
            "tau_hi": self._tau_hi,
            "is_fitted": self._is_fitted,
        }
        with open(path / "kbp_state.pkl", "wb") as f:
            pickle.dump(state, f)

        if self._probe is not None:
            self._probe.save(path / "probe.pkl")

        logger.info(f"KBP saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path], device: Optional[str] = None) -> "KBP":
        """Load KBP from disk. Model weights are loaded lazily."""
        path = Path(path)
        with open(path / "kbp_state.pkl", "rb") as f:
            state = pickle.load(f)

        kbp = cls(
            model_name_or_path=state["model_name"],
            best_layer=state["best_layer"],
            mode=state["mode"],
            device=device,
        )
        kbp._tau_lo = state["tau_lo"]
        kbp._tau_hi = state["tau_hi"]
        kbp._is_fitted = state["is_fitted"]

        probe_path = path / "probe.pkl"
        if probe_path.exists():
            kbp._probe = LinearProbe.load(probe_path)

        logger.info(f"KBP loaded from {path}")
        return kbp

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_default_best_layer(self) -> int:
        """Return paper-recommended ℓ* for known models."""
        name_lower = self.model_name.lower()
        for key, layer in self.KNOWN_BEST_LAYERS.items():
            if key in name_lower:
                return layer
        logger.warning(
            f"Unknown model '{self.model_name}'. Defaulting to layer 23 as ℓ*. "
            "Run fit(sweep_layers=True) to find the optimal layer."
        )
        return 23

    def _get_erank_estimator(self) -> EffectiveRankEstimator:
        if self._erank_estimator is None:
            self._erank_estimator = EffectiveRankEstimator(n_queries=256)
        return self._erank_estimator

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("KBP is not fitted. Call fit() or calibrate() first.")
        if self.mode == "supervised" and self._probe is None:
            raise RuntimeError(
                "Supervised mode requires a trained probe. Call fit() first."
            )
