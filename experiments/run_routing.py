"""
Dynamic RAG Routing Evaluation (Table 4)
=========================================
Reproduces the routing experiment from Section 7.2:

  Strategy                Acc.    % Retrieved   Latency
  Never retrieve          41.3%        0%         1.00×
  Always retrieve         67.4%      100%         2.73×
  KBP (unsupervised)      64.2%       57%         1.94×
  KBP (supervised)†       66.9%       51%         1.83×
  Self-RAG†               66.3%       61%         1.98×

† Statistically tied (paired t-test, p > 0.05)

Usage
-----
# Quick evaluation with pre-fitted KBP and a simple BM25 retriever:
python experiments/run_routing.py \
    --probe-checkpoint checkpoints/kbp_llama3_layer23.pkl \
    --model meta-llama/Meta-Llama-3-8B \
    --best-layer 23 \
    --dataset popqa \
    --split low_mid \
    --n-queries 1000 \
    --retriever bm25 \
    --retriever-corpus data/wiki_100w.jsonl

# Sweep retrieval thresholds (Appendix K, Figure 9):
python experiments/run_routing.py \
    --probe-checkpoint checkpoints/kbp_llama3_layer23.pkl \
    --model meta-llama/Meta-Llama-3-8B \
    --best-layer 23 \
    --sweep-thresholds \
    --output-dir results/routing
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Dummy retriever/generator for evaluation without real infrastructure ──────

class BM25Retriever:
    """Thin wrapper around rank_bm25. Falls back to random if not installed."""

    def __init__(self, corpus_path: Optional[str] = None, top_k: int = 5):
        self.top_k = top_k
        self.corpus = []
        self._bm25 = None

        if corpus_path and Path(corpus_path).exists():
            self._load_corpus(corpus_path)

    def _load_corpus(self, path: str):
        import json
        logger.info(f"Loading corpus from {path}...")
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                self.corpus.append(item.get("text", item.get("passage", "")))
        logger.info(f"Corpus loaded: {len(self.corpus)} passages")
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [doc.lower().split() for doc in self.corpus]
            self._bm25 = BM25Okapi(tokenized)
        except ImportError:
            logger.warning("rank_bm25 not installed. Using random retrieval.")

    def __call__(self, query: str) -> List[str]:
        if self._bm25 is None or not self.corpus:
            # Fallback: return placeholder passages
            return [f"[Retrieved passage {i} for: {query[:40]}]" for i in range(self.top_k)]
        scores = self._bm25.get_scores(query.lower().split())
        top_idx = np.argsort(scores)[::-1][: self.top_k]
        return [self.corpus[i] for i in top_idx]


class GreedyGenerator:
    """Wraps a HuggingFace causal LM for greedy generation."""

    def __init__(self, model, tokenizer, max_new_tokens: int = 20):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def __call__(self, query: str, docs: Optional[List[str]] = None) -> str:
        import torch

        if docs:
            context = "\n\n".join(docs[:3])
            prompt = (
                f"Answer the question based on the context.\n\n"
                f"Context: {context}\n\nQuestion: {query}\nAnswer:"
            )
        else:
            prompt = f"Question: {query}\nAnswer:"

        enc = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(next(self.model.parameters()).device)

        with torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        gen_ids = out[0][enc["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


# ── Dataset helpers ───────────────────────────────────────────────────────────

def load_popqa_low_mid(
    max_samples: int = 1000,
    freq_lo: int = 1000,
    freq_hi: int = 100_000,
) -> Tuple[List[str], List[str], List[int]]:
    """
    Load PopQA low+mid frequency split (the target partition in Table 4).

    Returns (queries, answers, ks_labels).
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("akariasai/PopQA", split="test")
    except Exception as e:
        logger.warning(f"Could not load PopQA: {e}. Using synthetic data.")
        queries = [f"Who is person {i}?" for i in range(max_samples)]
        answers = [f"Person {i}" for i in range(max_samples)]
        labels = [0] * max_samples
        return queries, answers, labels

    queries, answers, labels = [], [], []
    for item in ds:
        pv = item.get("s_pop", 0)
        if not (freq_lo <= pv < freq_hi):
            continue
        q = item.get("question", "")
        a = item.get("obj", item.get("answer", ""))
        if not q:
            continue
        label = 1 if pv >= 10_000 else 0
        queries.append(q)
        answers.append(str(a))
        labels.append(label)
        if len(queries) >= max_samples:
            break

    logger.info(
        f"PopQA low+mid: {len(queries)} queries "
        f"(KS={sum(labels)}, KD={len(labels)-sum(labels)})"
    )
    return queries, answers, labels


# ── Core evaluation ───────────────────────────────────────────────────────────

def evaluate_strategy(
    strategy: str,
    queries: List[str],
    ground_truth: List[str],
    retriever: Callable,
    generator: Callable,
    kbp=None,
) -> Dict[str, float]:
    """
    Evaluate a routing strategy and return accuracy + retrieval rate.

    strategy: "never" | "always" | "kbp_supervised" | "kbp_unsupervised"
    """
    n_correct = 0
    n_retrieved = 0
    latencies = []

    for i, (query, gt) in enumerate(zip(queries, ground_truth)):
        t0 = time.perf_counter()

        if strategy == "never":
            retrieve = False
        elif strategy == "always":
            retrieve = True
        elif strategy in ("kbp_supervised", "kbp_unsupervised"):
            result = kbp.predict(query)
            retrieve = result.should_retrieve
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        docs = retriever(query) if retrieve else None
        answer = generator(query, docs)
        latencies.append((time.perf_counter() - t0) * 1000)

        n_retrieved += int(retrieve)
        # Exact-match: ground truth occurs in generated answer
        if gt.lower() in answer.lower():
            n_correct += 1

        if (i + 1) % 200 == 0:
            logger.info(
                f"  [{strategy}] {i+1}/{len(queries)} | "
                f"Acc so far: {n_correct/(i+1):.2%} | "
                f"Retr: {n_retrieved/(i+1):.2%}"
            )

    return {
        "accuracy": n_correct / max(1, len(queries)),
        "retrieval_rate": n_retrieved / max(1, len(queries)),
        "avg_latency_ms": float(np.mean(latencies)),
        "n_queries": len(queries),
    }


def compute_relative_latency(
    results: Dict[str, Dict],
    baseline_strategy: str = "always",
) -> Dict[str, float]:
    """Compute latency relative to always-retrieve baseline."""
    baseline_lat = results[baseline_strategy]["avg_latency_ms"]
    return {
        strat: res["avg_latency_ms"] / max(1e-6, baseline_lat)
        for strat, res in results.items()
    }


def sweep_retrieval_thresholds(
    queries: List[str],
    ground_truth: List[str],
    retriever: Callable,
    generator: Callable,
    kbp_supervised,
    thresholds: Optional[List[float]] = None,
) -> Dict[float, Dict]:
    """
    Reproduce Figure 9 / Appendix K: accuracy and retrieval rate
    as a function of probe score threshold (default: 0.5).

    Lower threshold → retrieve more aggressively → higher accuracy, higher cost.
    """
    if thresholds is None:
        thresholds = np.linspace(0.2, 0.9, 15).tolist()

    results = {}
    for thresh in thresholds:
        n_correct = 0
        n_retrieved = 0

        for query, gt in zip(queries, ground_truth):
            result = kbp_supervised.predict(query)
            # Override: retrieve if score < threshold (more aggressive at lower thresh)
            retrieve = result.score is not None and result.score < thresh
            docs = retriever(query) if retrieve else None
            answer = generator(query, docs)

            n_retrieved += int(retrieve)
            if gt.lower() in answer.lower():
                n_correct += 1

        results[thresh] = {
            "accuracy": n_correct / len(queries),
            "retrieval_rate": n_retrieved / len(queries),
        }
        logger.info(
            f"  thresh={thresh:.2f}: acc={results[thresh]['accuracy']:.3f}, "
            f"retr={results[thresh]['retrieval_rate']:.3f}"
        )

    return results


def parse_args():
    p = argparse.ArgumentParser(description="KBP Dynamic RAG Routing Evaluation")
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    p.add_argument("--best-layer", type=int, default=23)
    p.add_argument("--probe-checkpoint", default=None)
    p.add_argument("--unsupervised", action="store_true",
                   help="Also evaluate unsupervised KBP")
    p.add_argument("--dataset", default="popqa", choices=["popqa"])
    p.add_argument("--split", default="low_mid",
                   help="Dataset split: low_mid, low, mid, all")
    p.add_argument("--n-queries", type=int, default=1000)
    p.add_argument("--retriever", default="bm25", choices=["bm25", "dummy"])
    p.add_argument("--retriever-corpus", default=None,
                   help="Path to corpus JSONL for BM25")
    p.add_argument("--sweep-thresholds", action="store_true",
                   help="Sweep probe score threshold (Appendix K)")
    p.add_argument("--output-dir", default="results/routing")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────────
    logger.info("Loading dataset...")
    queries, ground_truth, ks_labels = load_popqa_low_mid(args.n_queries)

    # ── Set up retriever & generator ──────────────────────────────────────────
    retriever = BM25Retriever(corpus_path=args.retriever_corpus)

    logger.info(f"Loading model: {args.model}")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map=device
        )
        generator = GreedyGenerator(model, tokenizer)
    except ImportError:
        logger.warning("transformers/torch not installed. Using dummy generator.")
        generator = lambda q, docs: f"answer_{hash(q) % 100}"  # noqa

    # ── Set up KBP ────────────────────────────────────────────────────────────
    kbp_sup = None
    kbp_unsup = None

    if args.probe_checkpoint and Path(args.probe_checkpoint).exists():
        from kbp.probe import LinearProbe
        from kbp.extractor import HiddenStateExtractor, ExtractionConfig
        from kbp.kbp import KBP

        kbp_sup = KBP(args.model, best_layer=args.best_layer, mode="supervised")
        kbp_sup._probe = LinearProbe.load(args.probe_checkpoint)
        kbp_sup._is_fitted = True
        logger.info(f"Supervised KBP loaded (layer {args.best_layer})")

        if args.unsupervised:
            kbp_unsup = KBP(args.model, best_layer=args.best_layer, mode="unsupervised")
            kbp_unsup.calibrate(queries[:256])
            logger.info("Unsupervised KBP calibrated")
    else:
        logger.warning(
            "No probe checkpoint provided. Skipping KBP routing strategies.\n"
            "Run scripts/train_probe.py first to obtain a checkpoint."
        )

    # ── Evaluate strategies ───────────────────────────────────────────────────
    strategies = ["never", "always"]
    if kbp_sup:
        strategies.append("kbp_supervised")
    if kbp_unsup:
        strategies.append("kbp_unsupervised")

    all_results = {}
    for strategy in strategies:
        logger.info(f"\n── Strategy: {strategy} ──")
        kbp = kbp_sup if "supervised" in strategy else kbp_unsup
        res = evaluate_strategy(
            strategy, queries, ground_truth, retriever, generator, kbp=kbp
        )
        all_results[strategy] = res
        logger.info(
            f"  Accuracy:       {res['accuracy']:.2%}\n"
            f"  Retrieval rate: {res['retrieval_rate']:.2%}\n"
            f"  Avg latency:    {res['avg_latency_ms']:.1f} ms"
        )

    # Relative latency
    if "always" in all_results:
        rel_lats = compute_relative_latency(all_results)
        for strat, rel in rel_lats.items():
            all_results[strat]["relative_latency"] = rel

    # ── Print comparison table ────────────────────────────────────────────────
    print(f"\n{'═' * 65}")
    print(f"  Table 4: RAG Routing Results — {args.dataset} low+mid frequency")
    print(f"{'═' * 65}")
    header = f"{'Strategy':<26} {'Acc.':>8} {'Retr.':>8} {'Latency':>10}"
    print(header)
    print("─" * 65)

    display_names = {
        "never": "Never retrieve",
        "always": "Always retrieve",
        "kbp_supervised": "KBP (supervised)†",
        "kbp_unsupervised": "KBP (unsupervised)",
    }
    paper_results = {
        "never":           (0.413, 0.00, 1.00),
        "always":          (0.674, 1.00, 2.73),
        "kbp_supervised":  (0.669, 0.51, 1.83),
        "kbp_unsupervised":(0.642, 0.57, 1.94),
    }

    for strat in strategies:
        res = all_results[strat]
        name = display_names.get(strat, strat)
        lat = res.get("relative_latency", 0.0)
        print(
            f"{name:<26} {res['accuracy']:>7.1%} {res['retrieval_rate']:>7.1%} "
            f"{lat:>9.2f}×"
        )

    print("\n  Paper targets (Llama-3-8B, PopQA low+mid):")
    for strat, (acc, retr, lat) in paper_results.items():
        name = display_names.get(strat, strat)
        print(f"  {name:<26} {acc:>7.1%} {retr:>7.1%} {lat:>9.2f}×")

    # ── Threshold sweep ───────────────────────────────────────────────────────
    if args.sweep_thresholds and kbp_sup:
        logger.info("\nSweeping retrieval thresholds (Figure 9)...")
        sweep = sweep_retrieval_thresholds(
            queries, ground_truth, retriever, generator, kbp_sup
        )
        all_results["threshold_sweep"] = {
            str(t): v for t, v in sweep.items()
        }

    # Save
    out_path = output_dir / "routing_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
