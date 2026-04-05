"""
Dynamic RAG Router
==================
Uses KBP to route queries to retrieval only when the model lacks knowledge.

Paper Section 7.2 / Table 4:
  "KBP matches Self-RAG† accuracy (66.9% vs. 66.3%) while retrieving
   only 51% of queries — a 33% latency reduction."

  Routing strategy results on PopQA (low+mid frequency):
  ┌──────────────────────────┬──────┬─────────┬─────────┐
  │ Strategy                 │ Acc. │ % Retr. │ Latency │
  ├──────────────────────────┼──────┼─────────┼─────────┤
  │ Never retrieve           │ 41.3 │  0%     │  1.00×  │
  │ Always retrieve          │ 67.4 │ 100%    │  2.73×  │
  │ KBP (unsupervised)       │ 64.2 │  57%    │  1.94×  │
  │ KBP (supervised)†        │ 66.9 │  51%    │  1.83×  │
  └──────────────────────────┴──────┴─────────┴─────────┘

Threshold Motivation (Appendix K):
  AUROC ≥ 0.85 → non-negative expected value for RAG routing
  under cost ratio cr/Δa = 0.5.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

from kbp.kbp import KBP, KBPResult

logger = logging.getLogger(__name__)


@dataclass
class RouterConfig:
    """Configuration for the KBP-based RAG router."""

    # KBP inference
    mode: str = "supervised"
    """KBP mode: 'supervised' or 'unsupervised'."""

    # Retrieval control
    retrieve_on_uncertain: bool = True
    """Whether to retrieve when KBP returns UNCERTAIN (unsupervised mode)."""

    retrieve_on_deficient: bool = True
    """Whether to retrieve when KBP returns KNOWLEDGE_DEFICIENT."""

    # Expected value threshold (Appendix K)
    cost_ratio: float = 0.5
    """Retrieval cost ratio cr/Δa. Higher → retrieve less aggressively."""

    # Logging
    log_decisions: bool = False
    """Log each routing decision for debugging."""


@dataclass
class RoutingDecision:
    """Result from the KBP router for a single query."""

    query: str
    """Original query."""

    kbp_result: KBPResult
    """KBP detection result."""

    retrieve: bool
    """Whether to retrieve."""

    answer: Optional[str] = None
    """Final answer (if generation was performed)."""

    retrieved_docs: Optional[List[str]] = None
    """Retrieved documents (if retrieval was performed)."""

    latency_kbp_ms: float = 0.0
    """KBP inference latency in milliseconds."""

    latency_retrieval_ms: float = 0.0
    """Retrieval latency in milliseconds."""

    latency_generation_ms: float = 0.0
    """Generation latency in milliseconds."""

    @property
    def total_latency_ms(self) -> float:
        return self.latency_kbp_ms + self.latency_retrieval_ms + self.latency_generation_ms


@dataclass
class RouterStats:
    """Aggregate statistics across multiple routing decisions."""

    n_queries: int = 0
    n_retrieved: int = 0
    n_correct: int = 0
    total_latency_ms: float = 0.0
    kbp_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0

    @property
    def retrieval_rate(self) -> float:
        return self.n_retrieved / max(1, self.n_queries)

    @property
    def accuracy(self) -> float:
        return self.n_correct / max(1, self.n_queries)

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(1, self.n_queries)

    def relative_latency(self, always_retrieve_latency_ms: float) -> float:
        """Compute latency relative to always-retrieve baseline."""
        return self.avg_latency_ms / max(1e-6, always_retrieve_latency_ms)

    def summary(self) -> str:
        return (
            f"Queries:        {self.n_queries}\n"
            f"Retrieved:      {self.n_retrieved} ({self.retrieval_rate:.1%})\n"
            f"Accuracy:       {self.accuracy:.1%}\n"
            f"Avg latency:    {self.avg_latency_ms:.1f} ms\n"
            f"KBP overhead:   {self.kbp_latency_ms / max(1, self.n_queries):.2f} ms/query"
        )


class KBPRouter:
    """
    RAG router using KBP to dynamically decide when to retrieve.

    Example
    -------
    >>> router = KBPRouter(
    ...     kbp=kbp,
    ...     retriever=my_retriever,
    ...     generator=my_generator,
    ... )
    >>> decision = router.route("What is the GDP of Laos?")
    >>> print(decision.retrieve, decision.answer)

    Batch processing
    ----------------
    >>> decisions = router.route_batch(queries)
    >>> stats = router.get_stats()
    >>> print(f"Retrieved {stats.retrieval_rate:.1%} of queries")
    """

    def __init__(
        self,
        kbp: KBP,
        retriever: Optional[Callable[[str], List[str]]] = None,
        generator: Optional[Callable[[str, Optional[List[str]]], str]] = None,
        config: Optional[RouterConfig] = None,
    ):
        """
        Parameters
        ----------
        kbp : KBP instance (fitted)
        retriever : callable(query) → list of document strings
        generator : callable(query, docs) → answer string
            If None, the router only makes routing decisions without generating.
        config : RouterConfig
        """
        self.kbp = kbp
        self.retriever = retriever
        self.generator = generator
        self.config = config or RouterConfig()
        self._stats = RouterStats()

    def route(self, query: str) -> RoutingDecision:
        """
        Route a single query.

        Returns a RoutingDecision with the KBP label and whether to retrieve.
        If a generator is provided, also generates the answer.
        """
        # Step 1: KBP inference
        t0 = time.perf_counter()
        kbp_result = self.kbp.predict(query)
        kbp_latency = (time.perf_counter() - t0) * 1000  # ms

        # Step 2: Routing decision
        retrieve = self._should_retrieve(kbp_result)

        decision = RoutingDecision(
            query=query,
            kbp_result=kbp_result,
            retrieve=retrieve,
            latency_kbp_ms=kbp_latency,
        )

        if self.config.log_decisions:
            logger.debug(
                f"Query: '{query[:60]}...' | "
                f"KBP: {kbp_result.label} | "
                f"Retrieve: {retrieve}"
            )

        # Step 3: Retrieval (if needed)
        docs = None
        if retrieve and self.retriever is not None:
            t0 = time.perf_counter()
            docs = self.retriever(query)
            decision.retrieved_docs = docs
            decision.latency_retrieval_ms = (time.perf_counter() - t0) * 1000

        # Step 4: Generation
        if self.generator is not None:
            t0 = time.perf_counter()
            decision.answer = self.generator(query, docs)
            decision.latency_generation_ms = (time.perf_counter() - t0) * 1000

        # Update stats
        self._update_stats(decision)
        return decision

    def route_batch(
        self,
        queries: List[str],
        ground_truth: Optional[List[str]] = None,
    ) -> List[RoutingDecision]:
        """
        Route a batch of queries.

        If ground_truth is provided, computes accuracy statistics.
        """
        decisions = []
        for i, query in enumerate(queries):
            decision = self.route(query)
            decisions.append(decision)

            if ground_truth is not None and decision.answer is not None:
                # Exact match evaluation
                correct = ground_truth[i].lower() in decision.answer.lower()
                self._stats.n_correct += int(correct)

            if (i + 1) % 100 == 0:
                logger.info(
                    f"Processed {i + 1}/{len(queries)} queries | "
                    f"Retrieval rate so far: {self._stats.retrieval_rate:.1%}"
                )

        return decisions

    def get_stats(self) -> RouterStats:
        return self._stats

    def reset_stats(self) -> None:
        self._stats = RouterStats()

    def _should_retrieve(self, kbp_result: KBPResult) -> bool:
        """Determine whether to retrieve based on KBP result."""
        if kbp_result.label == "KNOWLEDGE_DEFICIENT":
            return self.config.retrieve_on_deficient
        elif kbp_result.label == "UNCERTAIN":
            return self.config.retrieve_on_uncertain
        else:  # KNOWLEDGE_SUFFICIENT
            return False

    def _update_stats(self, decision: RoutingDecision) -> None:
        self._stats.n_queries += 1
        self._stats.n_retrieved += int(decision.retrieve)
        self._stats.total_latency_ms += decision.total_latency_ms
        self._stats.kbp_latency_ms += decision.latency_kbp_ms
        self._stats.retrieval_latency_ms += decision.latency_retrieval_ms

    def evaluate_routing(
        self,
        queries: List[str],
        ground_truth: List[str],
        retriever: Optional[Callable] = None,
        generator: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """
        Evaluate routing strategy (reproduces Table 4 metrics).

        Parameters
        ----------
        queries : list of str
        ground_truth : list of str (expected answers)
        retriever : override retriever for this evaluation
        generator : override generator for this evaluation

        Returns
        -------
        dict with 'accuracy', 'retrieval_rate', 'relative_latency'
        """
        old_retriever = self.retriever
        old_generator = self.generator
        if retriever is not None:
            self.retriever = retriever
        if generator is not None:
            self.generator = generator

        self.reset_stats()
        decisions = self.route_batch(queries, ground_truth)
        stats = self.get_stats()

        # Compute always-retrieve latency as reference
        always_retrieve_latency = self._estimate_always_retrieve_latency(
            decisions
        )

        metrics = {
            "accuracy": stats.accuracy,
            "retrieval_rate": stats.retrieval_rate,
            "relative_latency": stats.relative_latency(always_retrieve_latency),
            "avg_kbp_latency_ms": stats.kbp_latency_ms / max(1, stats.n_queries),
        }

        logger.info("Routing evaluation results:")
        logger.info(f"  Accuracy:       {metrics['accuracy']:.1%}")
        logger.info(f"  Retrieval rate: {metrics['retrieval_rate']:.1%}")
        logger.info(f"  Rel. latency:   {metrics['relative_latency']:.2f}×")

        self.retriever = old_retriever
        self.generator = old_generator
        return metrics

    def _estimate_always_retrieve_latency(
        self, decisions: List[RoutingDecision]
    ) -> float:
        """
        Estimate the average latency of the always-retrieve baseline.
        Uses per-query retrieval and generation latencies observed.
        """
        if not decisions:
            return 1.0
        avg_retrieval = np.mean([d.latency_retrieval_ms for d in decisions if d.retrieve] or [0])
        avg_generation = np.mean([d.latency_generation_ms for d in decisions] or [0])
        return avg_retrieval + avg_generation


# ------------------------------------------------------------------
# Convenience function for quick evaluation
# ------------------------------------------------------------------

def evaluate_routing_strategies(
    kbp_supervised: KBP,
    kbp_unsupervised: KBP,
    queries: List[str],
    ground_truth: List[str],
    retriever: Callable,
    generator: Callable,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate multiple routing strategies and compare (Table 4).

    Returns
    -------
    dict mapping strategy name → metrics dict
    """
    import numpy as np

    results = {}

    strategies = [
        ("Never retrieve", None, False),
        ("Always retrieve", None, True),
        ("KBP (supervised)", kbp_supervised, None),  # dynamic
        ("KBP (unsupervised)", kbp_unsupervised, None),
    ]

    for name, kbp, always_retrieve in strategies:
        if always_retrieve is True:
            # Always retrieve
            n_correct = 0
            for q, gt in zip(queries, ground_truth):
                docs = retriever(q)
                answer = generator(q, docs)
                if gt.lower() in answer.lower():
                    n_correct += 1
            results[name] = {
                "accuracy": n_correct / len(queries),
                "retrieval_rate": 1.0,
            }
        elif always_retrieve is False:
            # Never retrieve
            n_correct = 0
            for q, gt in zip(queries, ground_truth):
                answer = generator(q, None)
                if gt.lower() in answer.lower():
                    n_correct += 1
            results[name] = {
                "accuracy": n_correct / len(queries),
                "retrieval_rate": 0.0,
            }
        else:
            # KBP dynamic routing
            router = KBPRouter(kbp, retriever=retriever, generator=generator)
            metrics = router.evaluate_routing(queries, ground_truth)
            results[name] = metrics

    # Print comparison table
    print(f"\n{'Strategy':<28} {'Acc.':>8} {'% Retr.':>10}")
    print("-" * 50)
    for name, m in results.items():
        print(
            f"{name:<28} "
            f"{m['accuracy']:>7.1%} "
            f"{m['retrieval_rate']:>9.1%}"
        )

    return results
