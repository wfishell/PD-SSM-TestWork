import argparse
import json

from train import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="Name of a JSON config in experiment_configs/ (without .json)")
    args = parser.parse_args()

    with open(f"experiment_configs/{args.config}.json") as f:
        config = json.load(f)

    run_experiment(config)


if __name__ == "__main__":
    main()
