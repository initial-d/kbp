"""
Knowledge Boundary Probe (KBP)
================================
Detecting LLM knowledge boundaries via internal representation geometry.

Paper: "Does the Model Know What It Knows? Detecting Knowledge Boundaries
        via Internal Representation Geometry" (ACL 2025)

Quick Start
-----------
>>> from kbp import KBP
>>> kbp = KBP.from_pretrained("meta-llama/Meta-Llama-3-8B")
>>> kbp.fit(train_queries, train_labels)
>>> result = kbp.predict("Who wrote 1984?")
>>> print(result.label, result.score)
"""

from kbp.probe import LinearProbe
from kbp.effective_rank import EffectiveRankEstimator
def __getattr__(name):
    """Lazy-load torch-dependent modules to allow partial use without PyTorch."""
    if name in ("KBP", "KBPResult"):
        from kbp.kbp import KBP, KBPResult
        return {"KBP": KBP, "KBPResult": KBPResult}[name]
    if name == "HiddenStateExtractor":
        from kbp.extractor import HiddenStateExtractor
        return HiddenStateExtractor
    if name == "KBPRouter":
        from kbp.routing import KBPRouter
        return KBPRouter
    raise AttributeError(f"module 'kbp' has no attribute {name!r}")

__version__ = "0.1.0"
__all__ = [
    "KBP",
    "KBPResult",
    "LinearProbe",
    "EffectiveRankEstimator",
    "HiddenStateExtractor",
    "KBPRouter",
]
