#!/usr/bin/env python
"""Run experiment from YAML config."""

import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    from task_space.experiments import ExperimentConfig, run_experiment

    config = ExperimentConfig.from_yaml(args.config)

    if args.dry_run:
        print(f"Config: {config.name}")
        print(f"  Similarity: {config.similarity}")
        print(f"  Shock: {config.shock or 'None'}")
        return

    run_experiment(config)


if __name__ == '__main__':
    main()
