"""T6 word algebra problems: the official 40-epoch, five-run evaluator."""

import argparse

from examples.wap import wap_evaluate

from ...utils.stats import use_sample_std
from ...utils.tee import run_log


def build_parser():
    # Upstream's evaluator already runs the paper's protocol and takes no arguments.
    return argparse.ArgumentParser(description=__doc__)


def main():
    build_parser().parse_args()
    use_sample_std()
    with run_log("t6_wap"):
        print("=== wap | 40 epochs, batch 32, 5 runs", flush=True)
        wap_evaluate.main()


if __name__ == "__main__":
    main()
