"""
Train and save a KBP probe.

Usage
-----
python scripts/train_probe.py \
    --hidden-states data/hidden_states/llama3_popqa.pt \
    --layer 23 \
    --output checkpoints/kbp_llama3_layer23.pkl \
    --seeds 5

# Or sweep all layers to find ℓ*:
python scripts/train_probe.py \
    --hidden-states data/hidden_states/llama3_popqa.pt \
    --sweep \
    --n-layers 32 \
    --output-dir checkpoints/
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
    p = argparse.ArgumentParser(description="Train KBP probe")
    p.add_argument("--hidden-states", required=True, help="Path to .pt hidden states file")
    p.add_argument("--layer", type=int, default=23, help="Layer to train probe on")
    p.add_argument("--output", default=None, help="Output .pkl for single-layer probe")
    p.add_argument("--sweep", action="store_true", help="Sweep all layers to find ℓ*")
    p.add_argument("--n-layers", type=int, default=32, help="Total model layers (for sweep)")
    p.add_argument("--output-dir", default="checkpoints", help="Output dir for sweep results")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--C", type=float, default=1.0, help="Logistic regression regularization")
    p.add_argument("--save-best", action="store_true",
                   help="After sweep, save probe at best layer")
    return p.parse_args()


def main():
    args = parse_args()

    from kbp.extractor import HiddenStateOutput
    from kbp.probe import LayerWiseProbeTrainer, LinearProbe
    from sklearn.model_selection import train_test_split

    # Load hidden states
    logger.info(f"Loading hidden states from {args.hidden_states}")
    hs_output = HiddenStateOutput.load(args.hidden_states)

    # Load labels
    label_path = Path(args.hidden_states).with_suffix(".labels.npy")
    if not label_path.exists():
        logger.error(f"Labels not found at {label_path}")
        sys.exit(1)
    labels = np.load(label_path)
    logger.info(f"Labels: {labels.sum()} KS, {(1-labels).sum()} KD")

    probe_kwargs = {"C": args.C}
    trainer = LayerWiseProbeTrainer(
        probe_kwargs=probe_kwargs,
        train_ratio=args.train_ratio,
        n_seeds=args.seeds,
    )

    if args.sweep:
        # Full layer sweep
        layers = hs_output.layer_indices
        hidden_states = {l: hs_output.get_layer(l).numpy() for l in layers}

        lw_results = trainer.fit_all_layers(
            hidden_states,
            labels,
            total_layers=args.n_layers,
        )
        print("\n" + lw_results.summary_table())

        # Save summary
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            str(l): {
                "mean": lw_results.mean_auroc(l),
                "std": lw_results.std_auroc(l),
            }
            for l in sorted(lw_results.results.keys())
        }
        with open(output_dir / "layer_sweep_results.json", "w") as f:
            json.dump(summary, f, indent=2)

        if args.save_best:
            best_layer, best_auroc = lw_results.best_layer()
            probe, _ = trainer.fit_best_probe(hidden_states, labels, best_layer)
            save_path = output_dir / f"kbp_probe_layer{best_layer}.pkl"
            probe.save(save_path)
            logger.info(f"Best probe saved: layer={best_layer}, AUROC={best_auroc:.4f}")

    else:
        # Single layer
        if args.layer not in hs_output.layer_indices:
            available = hs_output.layer_indices
            logger.error(f"Layer {args.layer} not available. Have: {available}")
            sys.exit(1)

        X = hs_output.get_layer(args.layer).numpy()
        aurocs = []

        for seed in range(args.seeds):
            X_train, X_test, y_train, y_test = train_test_split(
                X, labels,
                train_size=args.train_ratio,
                stratify=labels,
                random_state=seed,
            )
            probe = LinearProbe(C=args.C, random_state=seed)
            probe.fit(X_train, y_train)
            auroc = probe.auroc(X_test, y_test)
            aurocs.append(auroc)
            logger.info(f"  Seed {seed}: AUROC={auroc:.4f}")

        mean_auroc = np.mean(aurocs)
        std_auroc = np.std(aurocs)
        logger.info(f"\nFinal: AUROC={mean_auroc:.4f}±{std_auroc:.4f} (layer {args.layer})")

        # Save best-seed probe
        best_seed = int(np.argmax(aurocs))
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, train_size=args.train_ratio,
            stratify=labels, random_state=best_seed,
        )
        final_probe = LinearProbe(C=args.C, random_state=best_seed)
        final_probe.fit(X_train, y_train)

        if args.output:
            final_probe.save(args.output)
            logger.info(f"Probe saved to {args.output}")
        else:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / f"kbp_probe_layer{args.layer}.pkl"
            final_probe.save(save_path)
            logger.info(f"Probe saved to {save_path}")


if __name__ == "__main__":
    main()
