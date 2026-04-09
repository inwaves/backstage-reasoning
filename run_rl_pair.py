"""Launch script for the RL-pair experiment."""

import argparse
import logging

from performativity.experiments.rl_pair import ExperimentConfig, run_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test", action="store_true",
        help="Small test run (50 items, 512 max tokens)",
    )
    args = parser.parse_args()

    if args.test:
        config = ExperimentConfig(
            mmlu_n=40,
            gpqa_n=10,
            max_new_tokens=2048,
            post_trained_max_new_tokens=4096,
            n_bootstrap=1000,
            output_dir="results/rl_pair_test",
        )
    else:
        config = ExperimentConfig()

    results = run_experiment(config)
    if "error" not in results:
        print(f"\nDelta: {results['performativity']['delta']:.4f}")
        print(
            f"95% CI: [{results['performativity']['delta_ci_lower']:.4f}, "
            f"{results['performativity']['delta_ci_upper']:.4f}]"
        )
