"""End-to-end pipeline driver: EDA -> preprocessing -> FLAML training.

Run with `python main.py` (after activating the `fraud-ml` conda env).
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))

from utils.logger import get_logger  # noqa: E402

LOGGER = get_logger("pipeline")


def run_eda() -> None:
    from data_acquisition_understanding import eda

    eda.main()


def run_preprocess() -> None:
    from data_preparation import preprocess

    preprocess.main()


def run_train() -> None:
    from modeling import train

    train.main()


def run_explain() -> None:
    from modeling import explain

    explain.main()


STAGES = {
    "eda": run_eda,
    "preprocess": run_preprocess,
    "train": run_train,
    "explain": run_explain,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fraud detection pipeline")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["eda", "preprocess", "train"],
        choices=list(STAGES.keys()),
        help="Subset of stages to run (default: full pipeline)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    LOGGER.info("Pipeline starting | stages=%s", args.stages)
    for stage in args.stages:
        LOGGER.info(">>> Stage start: %s", stage)
        try:
            STAGES[stage]()
        except Exception:
            LOGGER.error("Stage %s failed:\n%s", stage, traceback.format_exc())
            return 1
        LOGGER.info("<<< Stage finished: %s", stage)
    LOGGER.info("Pipeline complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
