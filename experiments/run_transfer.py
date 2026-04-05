"""
Cross-Domain Transfer and Cross-Model Alignment
================================================
Reproduces Table 2 (cross-domain transfer) and Appendix O (Procrustes).

Usage
-----
# Cross-domain transfer
python experiments/run_transfer.py \
    --probe-checkpoint checkpoints/kbp_llama3_popqa.pkl \
    --source-hidden-states data/hidden_states/llama3_popqa.pt \
    --eval-hidden-states data/hidden_states/ \
    --datasets medbench laobench_k12 mkqa_arabic

# Cross-model Procrustes alignment (Appendix O)
python experiments/run_transfer.py \
    --cross-model \
    --src-model-hs data/hidden_states/llama3_popqa.pt \
    --tgt-model-hs data/hidden_states/qwen3_popqa.pt \
    --alignment-queries data/alignment_queries.txt
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--probe-checkpoint", default=None)
    p.add_argument("--source-hidden-states", default=None)
    p.add_argument("--eval-hidden-states", default="data/hidden_states/")
    p.add_argument("--datasets", nargs="+", default=["medbench", "laobench_k12", "mmlu_general"])
    p.add_argument("--best-layer", type=int, default=23)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--output-dir", default="results/transfer")

    # Cross-model alignment
    p.add_argument("--cross-model", action="store_true")
    p.add_argument("--src-model-hs", default=None)
    p.add_argument("--tgt-model-hs", default=None)
    p.add_argument("--alignment-sizes", nargs="+", type=int, default=[50, 100, 150, 200, 300])

    return p.parse_args()


def cross_domain_transfer(args, probe, source_layer: int):
    """Evaluate cross-domain transfer (Table 2)."""
    from sklearn.metrics import roc_auc_score

    results = {}

    for dataset in args.datasets:
        hs_path = Path(args.eval_hidden_states) / f"{dataset}_layer{source_layer}.npy"
        label_path = Path(args.eval_hidden_states) / f"{dataset}_labels.npy"

        if not hs_path.exists() or not label_path.exists():
            logger.warning(f"Files not found for {dataset}. Skipping.")
            continue

        H = np.load(hs_path)
        labels = np.load(label_path)

        # Evaluate probe (trained on PopQA) on this dataset
        auroc = probe.auroc(H, labels)
        logger.info(f"  {dataset:20s}: AUROC={auroc:.4f}")
        results[dataset] = float(auroc)

    return results


def cross_model_procrustes(args):
    """
    Cross-model probe transfer via Procrustes alignment (Appendix O).

    Paper Table 18:
      Direct transfer:  AUROC 53-55 (chance)
      Aligned transfer: AUROC 77.8-79.4
    """
    from kbp.extractor import HiddenStateOutput
    from kbp.probe import LinearProbe
    from scipy.linalg import orthogonal_procrustes
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    logger.info("Loading source model hidden states (Llama-3-8B)...")
    src_output = HiddenStateOutput.load(args.src_model_hs)
    H_src_all = src_output.get_layer(args.best_layer).numpy()

    logger.info("Loading target model hidden states (Qwen3-8B)...")
    tgt_output = HiddenStateOutput.load(args.tgt_model_hs)
    H_tgt_all = tgt_output.get_layer(args.best_layer).numpy()

    # Load labels
    src_label_path = Path(args.src_model_hs).with_suffix(".labels.npy")
    labels = np.load(src_label_path)

    N = min(len(H_src_all), len(H_tgt_all), len(labels))
    H_src_all = H_src_all[:N]
    H_tgt_all = H_tgt_all[:N]
    labels = labels[:N]

    # Train probe on source model (Llama)
    rng = np.random.RandomState(42)
    train_idx = rng.choice(N, size=int(0.7 * N), replace=False)
    test_idx = np.setdiff1d(np.arange(N), train_idx)

    probe_src = LinearProbe()
    probe_src.fit(H_src_all[train_idx], labels[train_idx])

    # In-domain AUROC (source model)
    auroc_in_domain = probe_src.auroc(H_src_all[test_idx], labels[test_idx])
    logger.info(f"In-domain AUROC (Llama-3-8B): {auroc_in_domain:.4f}")

    # Direct transfer (no alignment) → should be ~53-55
    auroc_direct = probe_src.auroc(H_tgt_all[test_idx], labels[test_idx])
    logger.info(f"Direct transfer AUROC: {auroc_direct:.4f} (paper: ~53-55)")

    # Procrustes alignment at different sizes
    alignment_results = {}
    best_output_baseline = 75.2  # SelfCheckGPT from Table 1

    print(f"\n{'Align N':>10} {'AUROC':>8} {'> Baseline':>12}")
    print("─" * 35)

    for n_align in args.alignment_sizes:
        aurocs_per_seed = []
        for seed in range(args.seeds):
            rng_s = np.random.RandomState(seed)
            # Disjoint alignment and test sets
            pool = np.setdiff1d(np.arange(N), test_idx)
            if n_align > len(pool):
                continue
            align_idx = rng_s.choice(pool, size=n_align, replace=False)

            H_align_src = H_src_all[align_idx]
            H_align_tgt = H_tgt_all[align_idx]

            # Procrustes: R* = argmin_{RᵀR=I} Σ‖H_tgt_i - R H_src_i‖²
            R, _ = orthogonal_procrustes(H_align_src, H_align_tgt)

            # Apply aligned probe to target model test set
            H_test_tgt = H_tgt_all[test_idx]
            H_test_aligned = H_test_tgt @ R  # Rotate target to source space

            auroc = probe_src.auroc(H_test_aligned, labels[test_idx])
            aurocs_per_seed.append(auroc)

        mean_auroc = np.mean(aurocs_per_seed)
        above_baseline = "✓" if mean_auroc > best_output_baseline else "✗"
        print(f"{n_align:>10} {mean_auroc:>8.4f} {above_baseline:>12}")

        alignment_results[n_align] = {
            "mean_auroc": float(mean_auroc),
            "std_auroc": float(np.std(aurocs_per_seed)),
            "above_baseline": bool(mean_auroc > best_output_baseline),
        }

    return {
        "in_domain_auroc": float(auroc_in_domain),
        "direct_transfer_auroc": float(auroc_direct),
        "aligned_results": {str(k): v for k, v in alignment_results.items()},
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    if args.cross_model:
        # Appendix O: cross-model Procrustes alignment
        logger.info("=== Cross-Model Procrustes Alignment (Appendix O) ===")
        all_results["cross_model"] = cross_model_procrustes(args)

    else:
        # Table 2: cross-domain transfer
        from kbp.probe import LinearProbe
        from kbp.extractor import HiddenStateOutput

        logger.info("=== Cross-Domain Transfer (Table 2) ===")

        # Load or train probe
        if args.probe_checkpoint and Path(args.probe_checkpoint).exists():
            probe = LinearProbe.load(args.probe_checkpoint)
            logger.info(f"Probe loaded from {args.probe_checkpoint}")
        elif args.source_hidden_states:
            logger.info("Training probe on PopQA source data...")
            hs_output = HiddenStateOutput.load(args.source_hidden_states)
            H_src = hs_output.get_layer(args.best_layer).numpy()
            label_path = Path(args.source_hidden_states).with_suffix(".labels.npy")
            labels = np.load(label_path)

            probe = LinearProbe()
            probe.fit(H_src, labels)
            # Save for reuse
            Path("checkpoints").mkdir(exist_ok=True)
            probe.save("checkpoints/kbp_probe.pkl")
        else:
            raise ValueError("Provide either --probe-checkpoint or --source-hidden-states")

        results = cross_domain_transfer(args, probe, args.best_layer)
        all_results["cross_domain"] = results

        # Print table
        print("\n" + "═" * 50)
        print("  Cross-Domain Transfer AUROC (Table 2)")
        print("═" * 50)
        for dataset, auroc in results.items():
            print(f"  {dataset:<25}: {auroc:.4f}")
        print("═" * 50)

    # Save results
    results_path = output_dir / "transfer_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
