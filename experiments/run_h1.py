"""
H1: Linear Separability of Knowledge Sufficiency
==================================================
Reproduces the main H1 result from Table 1 and Figure 1.

Usage
-----
# Full sweep (finds ℓ*):
python experiments/run_h1.py \
    --hidden-states data/hidden_states/llama3_popqa.pt \
    --model-name "Llama-3-8B" \
    --n-layers 32 \
    --seeds 5

# Quick run at paper-reported best layer:
python experiments/run_h1.py \
    --hidden-states data/hidden_states/llama3_popqa.pt \
    --best-layer 23 \
    --quick
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
    p = argparse.ArgumentParser(description="H1: Linear separability experiment")
    p.add_argument("--hidden-states", required=True, help="Path to .pt file with hidden states")
    p.add_argument("--model-name", default="Llama-3-8B", help="Model name for display")
    p.add_argument("--n-layers", type=int, default=32, help="Total number of model layers")
    p.add_argument("--best-layer", type=int, default=None, help="Fixed layer (skips sweep)")
    p.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--quick", action="store_true", help="Only run at best-layer, skip sweep")
    p.add_argument("--output-dir", default="results/h1")
    p.add_argument("--plot", action="store_true", help="Generate layer-wise AUROC plot")
    p.add_argument(
        "--include-ablations",
        action="store_true",
        help="Run probe architecture and training size ablations",
    )
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------
    # Load hidden states
    # -----------------------------------------------------------
    import torch
    from kbp.extractor import HiddenStateOutput

    logger.info(f"Loading hidden states from {args.hidden_states}")
    hs_output = HiddenStateOutput.load(args.hidden_states)
    layers = hs_output.layer_indices
    logger.info(f"Layers available: {layers}")
    logger.info(f"N queries: {len(hs_output.queries)}")

    # -----------------------------------------------------------
    # Load labels
    # -----------------------------------------------------------
    # Labels should be stored alongside the hidden states.
    # Expected format: .pt file with {"labels": tensor, "queries": list}
    label_path = Path(args.hidden_states).with_suffix(".labels.npy")
    if label_path.exists():
        labels = np.load(label_path)
        logger.info(f"Labels loaded from {label_path}: {labels.sum()} KS, {(1-labels).sum()} KD")
    else:
        logger.error(
            f"Label file not found: {label_path}\n"
            "Expected a .labels.npy file alongside the hidden states.\n"
            "Create it with scripts/extract_hidden_states.py or provide PopQA frequency labels."
        )
        sys.exit(1)

    # -----------------------------------------------------------
    # Layer-wise probe training
    # -----------------------------------------------------------
    from kbp.probe import LayerWiseProbeTrainer, LinearProbe
    from sklearn.model_selection import train_test_split

    hidden_states = {l: hs_output.get_layer(l).numpy() for l in layers}

    if args.quick and args.best_layer is not None:
        # Single layer evaluation
        logger.info(f"Quick mode: evaluating layer {args.best_layer} only")
        X = hidden_states[args.best_layer]
        aurocs = []
        for seed in range(args.seeds):
            X_train, X_test, y_train, y_test = train_test_split(
                X, labels,
                train_size=args.train_ratio,
                stratify=labels,
                random_state=seed,
            )
            probe = LinearProbe(random_state=seed)
            probe.fit(X_train, y_train)
            aurocs.append(probe.auroc(X_test, y_test))

        mean_auroc = np.mean(aurocs)
        std_auroc = np.std(aurocs)
        print(f"\n{'═' * 50}")
        print(f"  H1 Result ({args.model_name})")
        print(f"{'═' * 50}")
        print(f"  Layer {args.best_layer} ({args.best_layer / args.n_layers:.1%} depth)")
        print(f"  AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")
        print(f"{'═' * 50}\n")

        results = {
            "layer": args.best_layer,
            "normalized_depth": args.best_layer / args.n_layers,
            "auroc_mean": float(mean_auroc),
            "auroc_std": float(std_auroc),
            "per_seed": [float(a) for a in aurocs],
            "model": args.model_name,
        }

    else:
        # Full layer-wise sweep
        trainer = LayerWiseProbeTrainer(
            n_seeds=args.seeds,
            train_ratio=args.train_ratio,
        )
        lw_results = trainer.fit_all_layers(
            hidden_states,
            labels,
            total_layers=args.n_layers,
            model_name=args.model_name,
        )

        best_layer, best_auroc = lw_results.best_layer()
        print("\n" + lw_results.summary_table())

        results = {
            "best_layer": best_layer,
            "best_auroc": float(best_auroc),
            "model": args.model_name,
            "layer_results": {
                str(l): {
                    "mean": lw_results.mean_auroc(l),
                    "std": lw_results.std_auroc(l),
                    "depth": l / args.n_layers,
                }
                for l in sorted(lw_results.results.keys())
            },
        }

        # -----------------------------------------------------------
        # Probe architecture ablation (Appendix C.1 / Table 9)
        # -----------------------------------------------------------
        if args.include_ablations:
            logger.info("\nRunning probe architecture ablation (Table 9)...")
            from kbp.probe import compare_probes_architectures

            X = hidden_states[best_layer]
            X_train, X_test, y_train, y_test = train_test_split(
                X, labels, train_size=args.train_ratio, stratify=labels, random_state=0
            )
            arch_results = compare_probes_architectures(X_train, y_train, X_test, y_test)
            results["architecture_ablation"] = arch_results

            print("\nProbe Architecture Ablation (Table 9):")
            for arch, auroc in sorted(arch_results.items(), key=lambda x: -x[1]):
                print(f"  {arch:15s}: {auroc:.4f}")

            # Training size ablation (Appendix C.2 / Figure 6)
            logger.info("\nRunning training size ablation (Figure 6)...")
            from kbp.probe import compute_auroc_vs_training_size

            size_results = compute_auroc_vs_training_size(X, labels, n_seeds=args.seeds)
            results["training_size_ablation"] = {
                str(k): {"mean": v[0], "std": v[1]}
                for k, v in size_results.items()
            }

            print("\nProbe AUROC vs. Training Size (Figure 6):")
            for size, (mean, std) in sorted(size_results.items()):
                print(f"  n={size:6d}: {mean:.4f} ± {std:.4f}")

        # -----------------------------------------------------------
        # Plot (Figure 1)
        # -----------------------------------------------------------
        if args.plot:
            from kbp.metrics import plot_layerwise_auroc

            layer_plot_data = {
                args.model_name: {
                    l: (lw_results.mean_auroc(l), lw_results.std_auroc(l))
                    for l in sorted(lw_results.results.keys())
                }
            }
            plot_layerwise_auroc(
                layer_plot_data,
                total_layers=args.n_layers,
                save_path=str(output_dir / "figure1_layerwise_auroc.pdf"),
            )

    # Save results
    results_path = output_dir / f"h1_results_{args.model_name.lower().replace('-', '_')}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
