"""
Hidden State Extractor
======================
Extracts intermediate-layer hidden states from transformer LLMs using
forward hooks. Supports batch extraction for efficiency.

Paper Section 5.1:
  "We extract hidden states from layers 16–32 for Llama-3-8B and
   layers 16–36 for Qwen3-8B."

Paper Appendix N (Token Position Ablation):
  "The last-token representation consistently yields the highest AUROC."
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

import functools

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
    _inference_mode = torch.inference_mode
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None     # type: ignore
    def _inference_mode():
        def decorator(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                raise ImportError("PyTorch is required. Install with: pip install torch")
            return wrapper
        return decorator


@dataclass
class ExtractionConfig:
    """Configuration for hidden state extraction."""

    layer_start: int = 16
    """First layer to extract (inclusive). Paper: 16 for both models."""

    layer_end: int = 32
    """Last layer to extract (inclusive). Paper: 32 for Llama, 36 for Qwen."""

    token_position: str = "last"
    """
    Which token position to extract.
    Options:
      - "last"   : last input token (paper default, best AUROC per Appendix N)
      - "mean"   : mean-pool over all tokens
      - "first"  : first token (CLS-like)
      - "last_k" : mean of last k tokens
    """

    last_k: int = 3
    """Used when token_position='last_k'."""

    normalization: str = "l2"
    """
    Normalization applied to extracted vectors.
    Options: "l2" (paper default), "l1", "none"
    """

    batch_size: int = 32
    """Batch size for extraction."""

    max_length: int = 256
    """Maximum tokenization length."""

    dtype: object = None
    """Model dtype (torch.float16 by default). Set at runtime."""

    def __post_init__(self):
        if self.dtype is None and _TORCH_AVAILABLE:
            self.dtype = torch.float16


@dataclass
class HiddenStateOutput:
    """Output of hidden state extraction."""

    hidden_states: Dict[int, object]  # int → torch.Tensor (N, d)
    """Map from layer index → tensor of shape (N, d)."""

    queries: List[str]
    """Original queries in order."""

    layer_indices: List[int]
    """Sorted list of extracted layer indices."""

    model_name: str = ""
    """Model name for bookkeeping."""

    def to(self, device: Union[str, torch.device]) -> "HiddenStateOutput":
        self.hidden_states = {k: v.to(device) for k, v in self.hidden_states.items()}
        return self

    def get_layer(self, layer: int) -> torch.Tensor:
        """Return hidden states for a specific layer, shape (N, d)."""
        if layer not in self.hidden_states:
            raise KeyError(
                f"Layer {layer} not extracted. Available: {self.layer_indices}"
            )
        return self.hidden_states[layer]

    def normalized(self, norm: str = "l2") -> "HiddenStateOutput":
        """Return a copy with normalized hidden states."""
        new_states = {}
        for layer, h in self.hidden_states.items():
            new_states[layer] = _normalize(h, norm)
        return HiddenStateOutput(
            hidden_states=new_states,
            queries=self.queries,
            layer_indices=self.layer_indices,
            model_name=self.model_name,
        )

    def save(self, path: str) -> None:
        torch.save(
            {
                "hidden_states": self.hidden_states,
                "queries": self.queries,
                "layer_indices": self.layer_indices,
                "model_name": self.model_name,
            },
            path,
        )
        logger.info(f"Saved hidden states to {path}")

    @classmethod
    def load(cls, path: str) -> "HiddenStateOutput":
        data = torch.load(path, map_location="cpu")
        return cls(**data)


def _normalize(h: "torch.Tensor", norm: str) -> "torch.Tensor":
    """Normalize a batch of vectors (N, d)."""
    if norm == "l2":
        return nn.functional.normalize(h, p=2, dim=-1)
    elif norm == "l1":
        return nn.functional.normalize(h, p=1, dim=-1)
    elif norm == "none":
        return h
    else:
        raise ValueError(f"Unknown normalization: {norm}. Choose 'l2', 'l1', or 'none'.")


class HiddenStateExtractor:
    """
    Extracts intermediate-layer hidden states from a causal LM.

    Uses PyTorch forward hooks to capture the output of each transformer
    layer at the position(s) specified by the extraction config.

    Example
    -------
    >>> extractor = HiddenStateExtractor("meta-llama/Meta-Llama-3-8B")
    >>> output = extractor.extract(queries, layers=[22, 23, 24])
    >>> h = output.get_layer(23)  # shape (N, 4096)
    """

    def __init__(
        self,
        model_name_or_path: str,
        config: Optional[ExtractionConfig] = None,
        device: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for HiddenStateExtractor. "
                "Install with: pip install torch transformers"
            )
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name_or_path
        self.config = config or ExtractionConfig()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info(f"Loading model: {model_name_or_path} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            padding_side="right",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=self.config.dtype,
            device_map=device if device != "cpu" else None,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

        # Determine total number of layers
        self.n_layers = self._count_layers()
        logger.info(f"Model has {self.n_layers} transformer layers")

    def _count_layers(self) -> int:
        """Count the number of transformer decoder layers."""
        # Works for LlamaForCausalLM, Qwen2ForCausalLM, etc.
        for attr in ("model", "transformer"):
            base = getattr(self.model, attr, None)
            if base is not None:
                for layers_attr in ("layers", "h", "blocks"):
                    layers = getattr(base, layers_attr, None)
                    if layers is not None:
                        return len(layers)
        raise RuntimeError("Cannot determine number of layers for this model architecture.")

    def _get_layer_module(self, layer_idx: int) -> nn.Module:
        """Return the nn.Module for a given layer index."""
        for attr in ("model", "transformer"):
            base = getattr(self.model, attr, None)
            if base is not None:
                for layers_attr in ("layers", "h", "blocks"):
                    layers = getattr(base, layers_attr, None)
                    if layers is not None:
                        return layers[layer_idx]
        raise RuntimeError(f"Cannot get layer module at index {layer_idx}.")

    @contextmanager
    def _hook_layers(self, target_layers: List[int]):
        """Context manager that registers forward hooks on target layers."""
        captured: Dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(layer_idx: int):
            def hook(module, input, output):
                # output is typically a tuple; first element is hidden state
                h = output[0] if isinstance(output, tuple) else output
                captured[layer_idx] = h.detach().cpu()

            return hook

        for idx in target_layers:
            module = self._get_layer_module(idx)
            handle = module.register_forward_hook(make_hook(idx))
            handles.append(handle)

        try:
            yield captured
        finally:
            for handle in handles:
                handle.remove()

    def _aggregate_token_position(
        self, h: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate hidden states across token positions.

        h: (batch, seq_len, d)
        attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding

        Returns: (batch, d)
        """
        pos = self.config.token_position
        if pos == "last":
            # Index of last real token for each sample
            lengths = attention_mask.sum(dim=1) - 1  # (batch,)
            batch_size = h.shape[0]
            return h[torch.arange(batch_size), lengths]
        elif pos == "first":
            return h[:, 0, :]
        elif pos == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            return (h * mask).sum(dim=1) / mask.sum(dim=1)
        elif pos == "last_k":
            k = self.config.last_k
            lengths = attention_mask.sum(dim=1)  # (batch,)
            results = []
            for i in range(h.shape[0]):
                start = max(0, lengths[i] - k)
                results.append(h[i, start : lengths[i]].mean(dim=0))
            return torch.stack(results)
        else:
            raise ValueError(f"Unknown token_position: {pos}")

    @_inference_mode()
    def extract(
        self,
        queries: List[str],
        layers: Optional[List[int]] = None,
    ) -> HiddenStateOutput:
        """
        Extract hidden states for a list of queries.

        Parameters
        ----------
        queries : list of str
        layers : list of int, optional
            Layer indices to extract. Defaults to range(layer_start, layer_end+1).

        Returns
        -------
        HiddenStateOutput
        """
        if layers is None:
            layers = list(range(self.config.layer_start, self.config.layer_end + 1))

        # Validate layer indices
        for layer in layers:
            if layer < 0 or layer >= self.n_layers:
                raise ValueError(
                    f"Layer {layer} out of range [0, {self.n_layers - 1}]"
                )

        all_hidden: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}

        # Process in batches
        bs = self.config.batch_size
        n_batches = (len(queries) + bs - 1) // bs

        logger.info(
            f"Extracting hidden states: {len(queries)} queries, "
            f"{len(layers)} layers, {n_batches} batches"
        )

        for batch_idx in range(n_batches):
            batch_queries = queries[batch_idx * bs : (batch_idx + 1) * bs]

            enc = self.tokenizer(
                batch_queries,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
            ).to(self.device)

            with self._hook_layers(layers) as captured:
                _ = self.model(**enc)

            attention_mask = enc["attention_mask"].cpu()
            for layer in layers:
                h_raw = captured[layer]  # (batch, seq_len, d)
                h_agg = self._aggregate_token_position(h_raw, attention_mask)
                h_norm = _normalize(h_agg, self.config.normalization)
                all_hidden[layer].append(h_norm)

            if (batch_idx + 1) % 10 == 0:
                logger.debug(f"  Processed {batch_idx + 1}/{n_batches} batches")

        # Concatenate all batches
        final_hidden = {
            layer: torch.cat(tensors, dim=0)
            for layer, tensors in all_hidden.items()
        }

        return HiddenStateOutput(
            hidden_states=final_hidden,
            queries=queries,
            layer_indices=sorted(layers),
            model_name=self.model_name,
        )

    def get_optimal_layer_range(self, model_name: str) -> Tuple[int, int]:
        """
        Return the paper-recommended layer range for a given model.

        Per Section 5.1:
          Llama-3-8B: layers 16–32 (out of 32 total)
          Qwen3-8B:   layers 16–36 (out of 36 total)
        """
        name_lower = model_name.lower()
        if "llama" in name_lower and "8b" in name_lower:
            return (16, 32)
        elif "qwen" in name_lower and "8b" in name_lower:
            return (16, 36)
        else:
            logger.warning(
                f"Unknown model '{model_name}'. Using default layer range 16–32. "
                "Override with layer_start/layer_end in ExtractionConfig."
            )
            return (16, 32)

    def get_best_probe_layer(self, model_name: str) -> int:
        """
        Return the paper-reported best probe layer (ℓ*).

        Per Section 6.1:
          Both models peak at layers 22–24 (≈65–75% depth).
        """
        name_lower = model_name.lower()
        if "llama" in name_lower:
            return 23  # Layer 23/32 = 71.9% depth, AUROC 88.4
        elif "qwen" in name_lower:
            return 23  # Layer 23/36 = 63.9% depth, AUROC 86.3
        else:
            logger.warning(
                f"Unknown model '{model_name}'. Defaulting to layer 23 as ℓ*."
            )
            return 23
