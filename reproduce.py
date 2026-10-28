"""Run the public-dataset experiments from one entry point."""
import argparse
import importlib
from pathlib import Path
import sys
import types

TASKS = {
    "chronos": "chronos",
    "ec4": "ec4_prophet_featuresets",
    "sensitivity": "ec4_spike_sensitivity",
    "compare": "comparisons",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=TASKS, help="experiment to run")
    args = parser.parse_args(sys.argv[1:2])
    remaining = sys.argv[2:]
    if args.task in {"ec4", "sensitivity"} and remaining:
        parser.error(f"{args.task} takes no additional arguments")

    # Load local modules without shadowing Python's standard-library `code`.
    package_name = "_retail_forecasting"
    package = types.ModuleType(package_name)
    package.__path__ = [str(Path(__file__).resolve().parent / "code")]
    sys.modules[package_name] = package
    module = importlib.import_module(f"{package_name}.{TASKS[args.task]}")
    sys.argv = [f"{parser.prog} {args.task}", *remaining]
    module.main()


if __name__ == "__main__":
    main()
