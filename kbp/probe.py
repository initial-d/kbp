"""
Linear Probe for Knowledge Sufficiency Detection
=================================================
Trains a logistic regression classifier on ℓ2-normalized last-token
hidden states to predict knowledge sufficiency (KS vs KD).

Paper Section 5.3:
  "For H1, we train logistic regression probes on ℓ2-normalized last-token
   hidden states at each layer, using 70% of the PopQA high/low-frequency
   split for training and 30% for evaluation."

Paper Table 9 (Probe Architecture Ablation):
  Linear (logistic): AUROC 88.4, <0.3% latency
  MLP (2-layer):     AUROC 89.1, ~1.1% latency  ← marginal gain, 3.7× cost
  SVM (RBF):         AUROC 88.7, ~0.8% latency
  k-NN (k=15):       AUROC 84.3, ~3.2% latency
  → Linear is the practical optimum.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

# Constants from paper
LABEL_KNOWLEDGE_SUFFICIENT = 1
LABEL_KNOWLEDGE_DEFICIENT = 0


@dataclass
class ProbeResult:
    """Result from layer-wise probe evaluation."""

    layer: int
    auroc: float
    auroc_std: float
    n_train: int
    n_test: int
    seed: int
    predictions: Optional[np.ndarray] = None
    scores: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None

    @property
    def normalized_depth(self) -> Optional[float]:
        return None  # Set externally when total layers known


@dataclass
class LayerWiseResults:
    """Results across all layers (for H1 analysis)."""

    results: Dict[int, List[ProbeResult]] = field(default_factory=dict)
    """layer_idx → list of results across seeds."""

    model_name: str = ""
    total_layers: int = 0

    def best_layer(self) -> Tuple[int, float]:
        """Return (layer_idx, mean_auroc) for the layer with highest mean AUROC."""
        means = {
            layer: np.mean([r.auroc for r in results])
            for layer, results in self.results.items()
        }
        best = max(means, key=means.get)
        return best, means[best]

    def mean_auroc(self, layer: int) -> float:
        return float(np.mean([r.auroc for r in self.results[layer]]))

    def std_auroc(self, layer: int) -> float:
        return float(np.std([r.auroc for r in self.results[layer]]))

    def summary_table(self) -> str:
        """Return a formatted table of layer-wise results."""
        lines = [
            f"{'Layer':>6} {'Depth':>8} {'AUROC':>10} {'Std':>8}",
            "-" * 40,
        ]
        for layer in sorted(self.results.keys()):
            depth = layer / self.total_layers if self.total_layers > 0 else 0
            mean = self.mean_auroc(layer)
            std = self.std_auroc(layer)
            lines.append(f"{layer:>6} {depth:>8.2f} {mean:>10.3f} {std:>8.3f}")
        best, best_auroc = self.best_layer()
        lines.append(f"\nBest layer: {best} (AUROC={best_auroc:.3f})")
        return "\n".join(lines)


class LinearProbe:
    """
    Logistic regression probe for knowledge sufficiency detection.

    Implements the KBP supervised detection head (Algorithm 1, Steps 3a).

    Parameters
    ----------
    C : float
        Inverse regularization strength (paper: C=1.0 via cross-validation).
    solver : str
        Solver for logistic regression (paper: 'lbfgs').
    max_iter : int
        Max iterations (paper: 1000).
    normalization : str
        Input normalization: 'l2' (default), 'l1', 'none'.
    architecture : str
        'linear' (default), 'mlp', 'svm', 'knn'.
    """

    def __init__(
        self,
        C: float = 1.0,
        solver: str = "lbfgs",
        max_iter: int = 1000,
        normalization: str = "l2",
        architecture: str = "linear",
        knn_k: int = 15,
        mlp_hidden: int = 256,
        random_state: int = 42,
    ):
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.normalization = normalization
        self.architecture = architecture
        self.knn_k = knn_k
        self.mlp_hidden = mlp_hidden
        self.random_state = random_state

        self._clf = None
        self._is_fitted = False

    def _build_classifier(self):
        if self.architecture == "linear":
            return LogisticRegression(
                C=self.C,
                solver=self.solver,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
        elif self.architecture == "svm":
            return SVC(
                C=self.C,
                kernel="rbf",
                probability=True,
                random_state=self.random_state,
            )
        elif self.architecture == "knn":
            return KNeighborsClassifier(
                n_neighbors=self.knn_k,
                metric="cosine",
                n_jobs=-1,
            )
        elif self.architecture == "mlp":
            from sklearn.neural_network import MLPClassifier

            return MLPClassifier(
                hidden_layer_sizes=(self.mlp_hidden,),
                activation="relu",
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        if self.normalization == "l2":
            return normalize(X, norm="l2")
        elif self.normalization == "l1":
            return normalize(X, norm="l1")
        elif self.normalization == "none":
            return X
        else:
            raise ValueError(f"Unknown normalization: {self.normalization}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearProbe":
        """
        Fit the probe.

        Parameters
        ----------
        X : (N, d) hidden states
        y : (N,) binary labels — 1=KS, 0=KD
        """
        X_norm = self._normalize(X)
        self._clf = self._build_classifier()
        self._clf.fit(X_norm, y)
        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of KNOWLEDGE_SUFFICIENT, shape (N,)."""
        self._check_fitted()
        X_norm = self._normalize(X)
        proba = self._clf.predict_proba(X_norm)
        ks_idx = list(self._clf.classes_).index(LABEL_KNOWLEDGE_SUFFICIENT)
        return proba[:, ks_idx]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions (1=KS, 0=KD), shape (N,)."""
        scores = self.predict_proba(X)
        return (scores >= 0.5).astype(int)

    def margin(self, X: np.ndarray) -> np.ndarray:
        """
        Return signed margin |wᵀh̃ + b| / ‖w‖ for each sample.

        The margin score correlates with prompt sensitivity (Section 6.3):
        queries near the decision boundary (low margin) are highly sensitive
        to prompt rephrasing; queries far from it are robustly classified.

        Only meaningful for linear architecture.
        """
        self._check_fitted()
        if self.architecture != "linear":
            raise NotImplementedError(
                "Margin score is only defined for the linear (logistic) probe."
            )
        X_norm = self._normalize(X)
        # For LogisticRegression: decision_function = wᵀx + b
        raw_margin = self._clf.decision_function(X_norm)  # (N,)
        # Normalize by ‖w‖ to get geometric distance to hyperplane
        w_norm = np.linalg.norm(self._clf.coef_[0])
        return np.abs(raw_margin) / w_norm

    def weight_vector(self) -> np.ndarray:
        """Return the probe weight vector w, shape (d,)."""
        self._check_fitted()
        if self.architecture != "linear":
            raise NotImplementedError("Weight vector is only defined for linear probe.")
        return self._clf.coef_[0]

    def bias(self) -> float:
        """Return the probe bias b."""
        self._check_fitted()
        if self.architecture != "linear":
            raise NotImplementedError("Bias is only defined for linear probe.")
        return float(self._clf.intercept_[0])

    def auroc(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute AUROC on a test set."""
        scores = self.predict_proba(X)
        return float(roc_auc_score(y, scores))

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Probe is not fitted. Call fit() first.")

    def save(self, path: Union[str, Path]) -> None:
        """Save probe to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "clf": self._clf,
            "C": self.C,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "normalization": self.normalization,
            "architecture": self.architecture,
            "is_fitted": self._is_fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"Probe saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LinearProbe":
        """Load probe from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        probe = cls(
            C=state["C"],
            solver=state["solver"],
            max_iter=state["max_iter"],
            normalization=state["normalization"],
            architecture=state["architecture"],
        )
        probe._clf = state["clf"]
        probe._is_fitted = state["is_fitted"]
        return probe


class LayerWiseProbeTrainer:
    """
    Trains and evaluates probes across all layers to find ℓ*.

    Reproduces Figure 1 and Tables 7/8 from the paper.

    Paper Section 6.1:
      "The best AUROC is achieved at layers 22–24 for both Llama-3-8B
       and Qwen3-8B, consistent with our prediction that knowledge-relevant
       features are encoded at ≈65% of total depth."
    """

    def __init__(
        self,
        probe_kwargs: Optional[dict] = None,
        train_ratio: float = 0.70,
        n_seeds: int = 5,
    ):
        self.probe_kwargs = probe_kwargs or {}
        self.train_ratio = train_ratio
        self.n_seeds = n_seeds

    def fit_all_layers(
        self,
        hidden_states: Dict[int, np.ndarray],
        labels: np.ndarray,
        total_layers: int = 32,
        model_name: str = "",
    ) -> LayerWiseResults:
        """
        Train and evaluate probes at every layer.

        Parameters
        ----------
        hidden_states : dict
            layer_idx → (N, d) array of hidden states
        labels : (N,) binary array — 1=KS, 0=KD
        total_layers : int
            Total number of layers in model (for normalized depth calculation)
        model_name : str

        Returns
        -------
        LayerWiseResults
        """
        results = LayerWiseResults(model_name=model_name, total_layers=total_layers)
        layers = sorted(hidden_states.keys())
        logger.info(
            f"Training probes on {len(layers)} layers × {self.n_seeds} seeds"
        )

        for layer in layers:
            X = hidden_states[layer]
            layer_results = []

            for seed in range(self.n_seeds):
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    labels,
                    train_size=self.train_ratio,
                    stratify=labels,
                    random_state=seed,
                )

                probe = LinearProbe(**self.probe_kwargs, random_state=seed)
                probe.fit(X_train, y_train)
                auroc = probe.auroc(X_test, y_test)

                layer_results.append(
                    ProbeResult(
                        layer=layer,
                        auroc=auroc,
                        auroc_std=0.0,
                        n_train=len(X_train),
                        n_test=len(X_test),
                        seed=seed,
                    )
                )

            mean_auroc = np.mean([r.auroc for r in layer_results])
            std_auroc = np.std([r.auroc for r in layer_results])
            depth = layer / total_layers
            logger.info(
                f"  Layer {layer:3d} ({depth:.2%}): AUROC={mean_auroc:.4f}±{std_auroc:.4f}"
            )

            results.results[layer] = layer_results

        best_layer, best_auroc = results.best_layer()
        logger.info(
            f"\nBest layer: {best_layer} "
            f"({best_layer / total_layers:.1%} depth), "
            f"AUROC={best_auroc:.4f}"
        )
        return results

    def fit_best_probe(
        self,
        hidden_states: Dict[int, np.ndarray],
        labels: np.ndarray,
        best_layer: int,
        seed: int = 0,
    ) -> Tuple[LinearProbe, float]:
        """
        Train a single probe at the best layer for deployment.

        Returns (fitted_probe, test_auroc).
        """
        X = hidden_states[best_layer]
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, train_size=self.train_ratio, stratify=labels, random_state=seed
        )
        probe = LinearProbe(**self.probe_kwargs, random_state=seed)
        probe.fit(X_train, y_train)
        auroc = probe.auroc(X_test, y_test)
        logger.info(f"Fitted probe at layer {best_layer}: AUROC={auroc:.4f}")
        return probe, auroc


def compare_probes_architectures(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Compare probe architectures (Table 9 in paper).

    Returns dict of architecture → AUROC.
    """
    architectures = ["linear", "svm", "knn", "mlp"]
    results = {}
    for arch in architectures:
        probe = LinearProbe(architecture=arch, random_state=random_state)
        probe.fit(X_train, y_train)
        auroc = probe.auroc(X_test, y_test)
        results[arch] = auroc
        logger.info(f"  {arch:10s}: AUROC={auroc:.4f}")
    return results


def compute_auroc_vs_training_size(
    X: np.ndarray,
    y: np.ndarray,
    sizes: Optional[List[int]] = None,
    n_seeds: int = 5,
) -> Dict[int, Tuple[float, float]]:
    """
    Reproduce Figure 6: probe AUROC vs. number of training examples.

    Paper Appendix C.2:
      "Performance saturates rapidly: 500 examples already recover 95%
       of the gain from full training, and even 100 examples achieve
       AUROC > 0.83 for both models."

    Returns
    -------
    dict mapping training size → (mean_auroc, std_auroc)
    """
    if sizes is None:
        sizes = [50, 100, 200, 500, 1000, 2000, 5000, len(X)]

    results = {}
    for size in sizes:
        if size >= len(X):
            size = len(X)
        aurocs = []
        for seed in range(n_seeds):
            if size < len(X):
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, train_size=size, stratify=y, random_state=seed
                )
            else:
                X_tr, y_tr = X, y
                X_te, y_te = X, y  # degenerate; only for logging

            probe = LinearProbe(random_state=seed)
            probe.fit(X_tr, y_tr)
            aurocs.append(probe.auroc(X_te, y_te))

        results[size] = (float(np.mean(aurocs)), float(np.std(aurocs)))
        logger.info(f"  n={size:6d}: AUROC={results[size][0]:.4f}±{results[size][1]:.4f}")

    return results
