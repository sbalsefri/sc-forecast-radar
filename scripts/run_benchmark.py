import argparse
import yaml
from src.benchmark.runner import run_benchmark


def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    run_benchmark(cfg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    main(args.config)
