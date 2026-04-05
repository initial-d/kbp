"""Unit tests for kbp.probe"""

import numpy as np
import pytest
from sklearn.datasets import make_classification

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kbp.probe import (
    LinearProbe,
    LayerWiseProbeTrainer,
    compare_probes_architectures,
    compute_auroc_vs_training_size,
    LABEL_KNOWLEDGE_SUFFICIENT,
    LABEL_KNOWLEDGE_DEFICIENT,
)


@pytest.fixture
def synthetic_data():
    """Generate synthetic KS/KD hidden states with linear separability."""
    X, y = make_classification(
        n_samples=1000,
        n_features=128,
        n_informative=20,
        n_redundant=5,
        random_state=42,
    )
    # Normalize (as paper does)
    from sklearn.preprocessing import normalize
    X = normalize(X, norm="l2")
    return X, y


class TestLinearProbe:
    def test_fit_predict(self, synthetic_data):
        X, y = synthetic_data
        probe = LinearProbe()
        probe.fit(X[:700], y[:700])

        preds = probe.predict(X[700:])
        assert preds.shape == (300,)
        assert set(preds).issubset({0, 1})

    def test_auroc_above_chance(self, synthetic_data):
        X, y = synthetic_data
        probe = LinearProbe()
        probe.fit(X[:700], y[:700])
        auroc = probe.auroc(X[700:], y[700:])
        assert auroc > 0.70, f"AUROC {auroc} too low for separable data"

    def test_margin(self, synthetic_data):
        X, y = synthetic_data
        probe = LinearProbe(architecture="linear")
        probe.fit(X[:700], y[:700])

        margins = probe.margin(X[700:])
        assert margins.shape == (300,)
        assert (margins >= 0).all(), "Margins should be non-negative (absolute value)"

    def test_weight_vector(self, synthetic_data):
        X, y = synthetic_data
        probe = LinearProbe()
        probe.fit(X[:700], y[:700])
        w = probe.weight_vector()
        assert w.shape == (128,), f"Expected (128,), got {w.shape}"

    def test_l2_normalization_applied(self, synthetic_data):
        X, y = synthetic_data
        probe = LinearProbe(normalization="l2")
        probe.fit(X[:700], y[:700])
        # Should run without error on un-normalized data too
        auroc = probe.auroc(X[700:] * 10, y[700:])  # Scale up
        auroc_norm = probe.auroc(X[700:], y[700:])
        # AUROC should be similar (normalization makes it scale-invariant)
        assert abs(auroc - auroc_norm) < 0.05

    def test_save_load(self, synthetic_data, tmp_path):
        X, y = synthetic_data
        probe = LinearProbe()
        probe.fit(X[:700], y[:700])
        original_auroc = probe.auroc(X[700:], y[700:])

        # Save and reload
        save_path = tmp_path / "probe.pkl"
        probe.save(save_path)
        loaded_probe = LinearProbe.load(save_path)

        loaded_auroc = loaded_probe.auroc(X[700:], y[700:])
        assert abs(original_auroc - loaded_auroc) < 1e-6

    def test_unfitted_raises(self, synthetic_data):
        X, y = synthetic_data
        probe = LinearProbe()
        with pytest.raises(RuntimeError, match="not fitted"):
            probe.predict(X)

    def test_mlp_probe(self, synthetic_data):
        X, y = synthetic_data
        probe = LinearProbe(architecture="mlp")
        probe.fit(X[:700], y[:700])
        auroc = probe.auroc(X[700:], y[700:])
        assert auroc > 0.65

    def test_knn_probe(self, synthetic_data):
        X, y = synthetic_data
        probe = LinearProbe(architecture="knn", knn_k=15)
        probe.fit(X[:700], y[:700])
        auroc = probe.auroc(X[700:], y[700:])
        assert auroc > 0.60

    def test_margin_mlp_raises(self, synthetic_data):
        X, y = synthetic_data
        probe = LinearProbe(architecture="mlp")
        probe.fit(X[:700], y[:700])
        with pytest.raises(NotImplementedError):
            probe.margin(X[700:])


class TestLayerWiseProbeTrainer:
    def test_fit_all_layers(self, synthetic_data):
        X, y = synthetic_data
        # Simulate 3 layers
        hidden_states = {
            10: X + np.random.randn(*X.shape) * 0.5,  # Noisier
            15: X + np.random.randn(*X.shape) * 0.2,  # Less noisy
            20: X,  # Clean
        }

        trainer = LayerWiseProbeTrainer(n_seeds=3, train_ratio=0.70)
        results = trainer.fit_all_layers(hidden_states, y, total_layers=32)

        # Layer 20 (cleanest) should have highest AUROC
        best_layer, best_auroc = results.best_layer()
        assert best_layer == 20
        assert best_auroc > 0.70

    def test_summary_table(self, synthetic_data):
        X, y = synthetic_data
        hidden_states = {10: X, 15: X}
        trainer = LayerWiseProbeTrainer(n_seeds=2)
        results = trainer.fit_all_layers(hidden_states, y, total_layers=32)
        summary = results.summary_table()
        assert "AUROC" in summary
        assert "Best layer" in summary


class TestAblations:
    def test_auroc_vs_training_size(self, synthetic_data):
        X, y = synthetic_data
        sizes = [100, 500, len(X)]
        results = compute_auroc_vs_training_size(X, y, sizes=sizes, n_seeds=2)
        assert len(results) == len(sizes)
        # AUROC should generally increase with more data
        aurocs = [results[s][0] for s in sorted(results.keys())]
        assert aurocs[0] < aurocs[-1]  # More data → better performance

    def test_compare_architectures(self, synthetic_data):
        X, y = synthetic_data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
        results = compare_probes_architectures(X_train, y_train, X_test, y_test)
        assert "linear" in results
        assert "svm" in results
        assert all(0 < v < 1 for v in results.values())
