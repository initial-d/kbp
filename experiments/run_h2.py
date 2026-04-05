"""
H2: Activation Dispersion as Label-Free Proxy for Knowledge Deficiency
======================================================================
Reproduces Table 3 (effective rank by task) and Figure 2 (erank vs. SNR).

Usage
-----
python experiments/run_h2.py \
    --model meta-llama/Meta-Llama-3-8B \
    --best-layer 23 \
    --tasks popqa_high popqa_mid popqa_low medbench laobench mkqa_arabic \
    --data-dir data/ \
    --n-queries 256
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

# Task metadata (effective rank regime per Table 3)
TASK_METADATA = {
    "popqa_high":    {"name": "PopQA High-freq",    "expected_regime": "KS"},
    "mmlu_general":  {"name": "MMLU General",        "expected_regime": "KS"},
    "popqa_mid":     {"name": "PopQA Mid-freq",      "expected_regime": "Mixed"},
    "mmlu_medgen":   {"name": "MMLU-MedGen",         "expected_regime": "Mixed"},
    "mmlu_colmed":   {"name": "MMLU-ColMed",         "expected_regime": "Mixed"},
    "mkqa_arabic":   {"name": "MKQA-Arabic",         "expected_regime": "Mixed"},
    "mkqa_thai":     {"name": "MKQA-Thai",           "expected_regime": "KD"},
    "mkqa_swahili":  {"name": "MKQA-Swahili",        "expected_regime": "KD"},
    "popqa_low":     {"name": "PopQA Low-freq",      "expected_regime": "KD"},
    "laobench_k12":  {"name": "LaoBench K12",        "expected_regime": "KD"},
    "medbench":      {"name": "MedBench",            "expected_regime": "KD"},
    "laobench_appl": {"name": "LaoBench Appl.",      "expected_regime": "KD"},
}

# Paper Table 3 reference values for Llama-3-8B
PAPER_ERANKS_LLAMA = {
    "popqa_high":    (18.3, 1.1),
    "mmlu_general":  (19.1, 1.3),
    "popqa_mid":     (26.4, 2.1),
    "mmlu_medgen":   (28.2, 2.4),
    "mmlu_colmed":   (31.4, 2.6),
    "mkqa_arabic":   (34.8, 2.8),
    "mkqa_thai":     (35.9, 2.9),
    "mkqa_swahili":  (37.2, 3.1),
    "popqa_low":     (38.7, 3.2),
    "laobench_k12":  (41.2, 3.6),
    "medbench":      (41.3, 3.6),
    "laobench_appl": (48.3, 4.6),
}


def parse_args():
    p = argparse.ArgumentParser(description="H2: Effective rank analysis")
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    p.add_argument("--best-layer", type=int, default=23, help="ℓ* from H1 experiment")
    p.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASK_METADATA.keys()),
        choices=list(TASK_METADATA.keys()),
    )
    p.add_argument("--data-dir", default="data/", help="Directory with task query files")
    p.add_argument("--hidden-states-dir", default=None, help="Pre-extracted hidden states")
    p.add_argument("--n-queries", type=int, default=256, help="Queries per task")
    p.add_argument("--n-bootstrap", type=int, default=20, help="Bootstrap resamples")
    p.add_argument("--compute-snr", action="store_true", help="Also compute gradient SNR")
    p.add_argument("--snr-n-steps", type=int, default=5)
    p.add_argument("--snr-n-examples", type=int, default=32)
    p.add_argument("--output-dir", default="results/h2")
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def load_task_queries(task_id: str, data_dir: str, n_queries: int) -> list:
    """Load queries for a given task from data directory."""
    data_path = Path(data_dir) / f"{task_id}_queries.txt"
    if not data_path.exists():
        logger.warning(f"Query file not found: {data_path}. Using synthetic placeholder.")
        # Return synthetic placeholder for demonstration
        return [f"[{task_id}] sample query {i}" for i in range(n_queries)]

    with open(data_path) as f:
        queries = [line.strip() for line in f if line.strip()]
    return queries[:n_queries]


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------
    # Load extractor (lazy — only loads if hidden states not pre-extracted)
    # -----------------------------------------------------------
    extractor = None
    if args.hidden_states_dir is None:
        from kbp.extractor import HiddenStateExtractor, ExtractionConfig
        config = ExtractionConfig(
            layer_start=args.best_layer,
            layer_end=args.best_layer,
        )
        extractor = HiddenStateExtractor(args.model, config=config)
        logger.info(f"Extractor loaded: {args.model}")

    # -----------------------------------------------------------
    # Compute effective rank for each task
    # -----------------------------------------------------------
    from kbp.effective_rank import EffectiveRankEstimator, classify_sft_viability

    estimator = EffectiveRankEstimator(
        n_queries=args.n_queries,
        n_bootstrap=args.n_bootstrap,
    )

    task_results = {}
    eranks_for_correlation = []
    snrs_for_correlation = []

    for task_id in args.tasks:
        meta = TASK_METADATA.get(task_id, {"name": task_id, "expected_regime": "?"})
        logger.info(f"\nProcessing task: {meta['name']}")

        # Load or extract hidden states
        if args.hidden_states_dir is not None:
            hs_path = Path(args.hidden_states_dir) / f"{task_id}_layer{args.best_layer}.npy"
            if hs_path.exists():
                H = np.load(hs_path)[: args.n_queries]
            else:
                logger.warning(f"Hidden states not found: {hs_path}. Skipping.")
                continue
        else:
            queries = load_task_queries(task_id, args.data_dir, args.n_queries)
            import torch
            from kbp.extractor import HiddenStateOutput
            output = extractor.extract(queries, layers=[args.best_layer])
            H = output.get_layer(args.best_layer).numpy()

        # Estimate effective rank
        erank_result = estimator.estimate(H)

        # Paper reference comparison
        paper_ref = PAPER_ERANKS_LLAMA.get(task_id)
        paper_str = f" (paper: {paper_ref[0]:.1f}±{paper_ref[1]:.1f})" if paper_ref else ""

        logger.info(
            f"  erank = {erank_result.erank:.2f} ± {erank_result.erank_std:.2f}"
            f"{paper_str}"
        )
        logger.info(f"  Regime: {erank_result.regime}")

        # SFT viability
        sft_info = classify_sft_viability(erank_result.erank)
        logger.info(f"  SFT: {sft_info['recommendation']}")

        task_results[task_id] = {
            "name": meta["name"],
            "erank": erank_result.erank,
            "erank_std": erank_result.erank_std,
            "regime": erank_result.regime,
            "expected_regime": meta["expected_regime"],
            "sft_recommendation": sft_info["recommendation"],
        }
        eranks_for_correlation.append(erank_result.erank)

        # Gradient SNR (optional)
        if args.compute_snr:
            import torch
            import torch.nn as nn
            from kbp.effective_rank import compute_gradient_snr

            # Simple linear probe as the fine-tuning model
            d = H.shape[1]
            probe_model = nn.Linear(d, 2)
            H_tensor = torch.tensor(H, dtype=torch.float32)
            # Placeholder labels for SNR computation
            y_tensor = torch.randint(0, 2, (len(H),))

            snr = compute_gradient_snr(
                probe_model, H_tensor, y_tensor,
                n_steps=args.snr_n_steps,
                batch_size=8,
            )
            task_results[task_id]["gradient_snr"] = snr
            snrs_for_correlation.append(snr)
            logger.info(f"  Gradient SNR: {snr:.4f}")

    # -----------------------------------------------------------
    # Print Table 3
    # -----------------------------------------------------------
    print("\n" + "═" * 60)
    print("  Table 3: Activation Effective Rank by Task")
    print("═" * 60)
    print(f"{'Task':<25} {'erank':>8} {'±std':>7} {'Regime':>8}")
    print("─" * 60)

    regime_order = {"KS": 0, "Mixed": 1, "KD": 2}
    sorted_tasks = sorted(
        task_results.items(),
        key=lambda x: (regime_order.get(x[1]["expected_regime"], 3), x[1]["erank"])
    )

    for task_id, res in sorted_tasks:
        match = "✓" if res["regime"] == res["expected_regime"] else "✗"
        print(
            f"{res['name']:<25} {res['erank']:>8.2f} {res['erank_std']:>6.2f} "
            f"  {res['regime']:>8} {match}"
        )

    # -----------------------------------------------------------
    # Spearman correlation (if SNR computed)
    # -----------------------------------------------------------
    if args.compute_snr and len(snrs_for_correlation) >= 4:
        from kbp.effective_rank import spearman_erank_vs_snr

        corr_results = spearman_erank_vs_snr(
            eranks_for_correlation, snrs_for_correlation
        )

        print("\n" + "═" * 60)
        print("  Spearman Correlation: erank vs. Gradient SNR")
        print("═" * 60)
        print(f"  ρs = {corr_results['rho']:.3f}  (p={corr_results['pval']:.4f})")
        print(f"  95% CI: [{corr_results['ci_low']:.3f}, {corr_results['ci_high']:.3f}]")
        print(f"  N = {len(eranks_for_correlation)} tasks")
        print("  Paper target: ρs ≈ −0.82, CI [−0.91, −0.71]")

        task_results["_spearman"] = {
            k: float(v) for k, v in corr_results.items() if k != "boot_rhos"
        }

    # -----------------------------------------------------------
    # Plot (Figure 2)
    # -----------------------------------------------------------
    if args.plot and args.compute_snr and len(snrs_for_correlation) >= 4:
        from kbp.metrics import plot_erank_vs_snr
        task_names = [task_results[t]["name"] for t in args.tasks if t in task_results]
        plot_erank_vs_snr(
            task_names=task_names,
            eranks=eranks_for_correlation,
            snrs=snrs_for_correlation,
            save_path=str(output_dir / "figure2_erank_vs_snr.pdf"),
        )

    # Save results
    results_path = output_dir / "h2_results.json"
    with open(results_path, "w") as f:
        json.dump(task_results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
