"""
Unit tests for kbp.extractor

Split into two groups:
  - torch-free tests: ExtractionConfig defaults, optimal layer ranges (always run)
  - torch-required tests: HiddenStateOutput, _normalize, token aggregation
    → skipped automatically when PyTorch is not installed
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

requires_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")


# ── torch-free tests ──────────────────────────────────────────────────────────

class TestExtractionConfig:
    def test_defaults(self):
        from kbp.extractor import ExtractionConfig
        cfg = ExtractionConfig()
        assert cfg.layer_start == 16
        assert cfg.layer_end == 32
        assert cfg.token_position == "last"
        assert cfg.normalization == "l2"
        assert cfg.batch_size == 32

    def test_custom_values(self):
        from kbp.extractor import ExtractionConfig
        cfg = ExtractionConfig(layer_start=10, layer_end=20, token_position="mean")
        assert cfg.layer_start == 10
        assert cfg.layer_end == 20
        assert cfg.token_position == "mean"


class TestOptimalLayerRange:
    def _make_stub(self, model_name: str, n_layers: int):
        from kbp.extractor import HiddenStateExtractor, ExtractionConfig
        ext = object.__new__(HiddenStateExtractor)
        ext.model_name = model_name
        ext.config = ExtractionConfig()
        ext.device = "cpu"
        ext.n_layers = n_layers
        return ext

    def test_llama3_8b(self):
        ext = self._make_stub("meta-llama/Meta-Llama-3-8B", 32)
        start, end = ext.get_optimal_layer_range("meta-llama/Meta-Llama-3-8B")
        assert (start, end) == (16, 32)

    def test_qwen3_8b(self):
        ext = self._make_stub("Qwen/Qwen3-8B", 36)
        start, end = ext.get_optimal_layer_range("Qwen/Qwen3-8B")
        assert (start, end) == (16, 36)

    def test_best_probe_layer_llama(self):
        ext = self._make_stub("meta-llama/Meta-Llama-3-8B", 32)
        assert ext.get_best_probe_layer("meta-llama/Meta-Llama-3-8B") == 23

    def test_best_probe_layer_qwen(self):
        ext = self._make_stub("Qwen/Qwen3-8B", 36)
        assert ext.get_best_probe_layer("Qwen/Qwen3-8B") == 23

    def test_unknown_model_returns_ints(self):
        ext = self._make_stub("some/unknown-7b", 28)
        start, end = ext.get_optimal_layer_range("some/unknown-7b")
        assert isinstance(start, int) and isinstance(end, int)
        assert start <= end


# ── torch-required tests ──────────────────────────────────────────────────────

@requires_torch
class TestHiddenStateOutput:
    def test_get_layer(self):
        from kbp.extractor import HiddenStateOutput
        hs = {5: torch.randn(10, 64), 10: torch.randn(10, 64)}
        out = HiddenStateOutput(hidden_states=hs, queries=["q"]*10, layer_indices=[5, 10])
        assert out.get_layer(5).shape == (10, 64)

    def test_get_layer_missing_raises(self):
        from kbp.extractor import HiddenStateOutput
        hs = {5: torch.randn(10, 64)}
        out = HiddenStateOutput(hidden_states=hs, queries=["q"]*10, layer_indices=[5])
        with pytest.raises(KeyError):
            out.get_layer(99)

    def test_normalized_l2(self):
        from kbp.extractor import HiddenStateOutput
        hs = {5: torch.randn(10, 64) * 100}
        out = HiddenStateOutput(hidden_states=hs, queries=["q"]*10, layer_indices=[5])
        out_l2 = out.normalized("l2")
        norms = out_l2.get_layer(5).norm(dim=-1)
        assert torch.allclose(norms, torch.ones(10), atol=1e-5)

    def test_save_load(self, tmp_path):
        from kbp.extractor import HiddenStateOutput
        hs = {5: torch.randn(8, 64), 10: torch.randn(8, 64)}
        queries = [f"q{i}" for i in range(8)]
        out = HiddenStateOutput(hidden_states=hs, queries=queries,
                                layer_indices=[5, 10], model_name="test-model")
        path = str(tmp_path / "hs.pt")
        out.save(path)
        loaded = HiddenStateOutput.load(path)
        assert loaded.layer_indices == [5, 10]
        assert loaded.model_name == "test-model"
        assert torch.allclose(loaded.get_layer(5), out.get_layer(5))

    def test_to_cpu(self):
        from kbp.extractor import HiddenStateOutput
        hs = {5: torch.randn(8, 64)}
        out = HiddenStateOutput(hidden_states=hs, queries=["q"]*8, layer_indices=[5])
        assert out.to("cpu").get_layer(5).device.type == "cpu"


@requires_torch
class TestNormalize:
    def test_l2_unit_norm(self):
        from kbp.extractor import _normalize
        x = torch.randn(20, 128)
        assert torch.allclose(_normalize(x, "l2").norm(dim=-1), torch.ones(20), atol=1e-5)

    def test_l1_unit_norm(self):
        from kbp.extractor import _normalize
        x = torch.abs(torch.randn(10, 64))
        assert torch.allclose(_normalize(x, "l1").abs().sum(dim=-1), torch.ones(10), atol=1e-5)

    def test_none_passthrough(self):
        from kbp.extractor import _normalize
        x = torch.randn(10, 64) * 5.0
        assert torch.allclose(_normalize(x, "none"), x)

    def test_invalid_raises(self):
        from kbp.extractor import _normalize
        with pytest.raises(ValueError, match="Unknown normalization"):
            _normalize(torch.randn(5, 32), "bad_norm")


@requires_torch
class TestTokenAggregation:
    def _stub(self, position: str, last_k: int = 3):
        from kbp.extractor import ExtractionConfig, HiddenStateExtractor
        cfg = ExtractionConfig(token_position=position, last_k=last_k)
        ext = object.__new__(HiddenStateExtractor)
        ext.config = cfg
        return ext

    def test_last_token(self):
        from kbp.extractor import HiddenStateExtractor
        ext = self._stub("last")
        h = torch.zeros(2, 5, 4)
        h[0, 2] = 1.0  # seq0: length=3, last real @ index 2
        h[1, 4] = 2.0  # seq1: length=5, last real @ index 4
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        result = HiddenStateExtractor._aggregate_token_position(ext, h, mask)
        assert result.shape == (2, 4)
        assert torch.allclose(result[0], torch.ones(4))
        assert torch.allclose(result[1], torch.full((4,), 2.0))

    def test_first_token(self):
        from kbp.extractor import HiddenStateExtractor
        ext = self._stub("first")
        h = torch.zeros(2, 5, 4)
        h[:, 0] = 99.0
        mask = torch.ones(2, 5)
        result = HiddenStateExtractor._aggregate_token_position(ext, h, mask)
        assert torch.allclose(result, torch.full((2, 4), 99.0))

    def test_mean_pool(self):
        from kbp.extractor import HiddenStateExtractor
        ext = self._stub("mean")
        h = torch.ones(1, 4, 8)
        h[0, 3] = 0.0
        mask = torch.tensor([[1, 1, 1, 0]])
        result = HiddenStateExtractor._aggregate_token_position(ext, h, mask)
        assert torch.allclose(result[0], torch.ones(8))

    def test_last_k(self):
        from kbp.extractor import HiddenStateExtractor
        ext = self._stub("last_k", last_k=2)
        h = torch.zeros(1, 6, 4)
        h[0, 3] = 1.0
        h[0, 4] = 3.0  # last 2 real tokens: idx 3 & 4, mean = 2.0
        mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
        result = HiddenStateExtractor._aggregate_token_position(ext, h, mask)
        assert torch.allclose(result[0], torch.full((4,), 2.0))

    def test_invalid_position_raises(self):
        from kbp.extractor import HiddenStateExtractor
        ext = self._stub("bad_pos")
        with pytest.raises(ValueError, match="Unknown token_position"):
            HiddenStateExtractor._aggregate_token_position(ext, torch.zeros(2, 5, 4), torch.ones(2, 5))
