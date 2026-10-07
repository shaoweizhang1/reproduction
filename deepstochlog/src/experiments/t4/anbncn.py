"""T4 a^n b^n c^n: the official evaluator with the paper's one-epoch protocol."""

import argparse

from examples.anbncn import anbncn_evaluate

from ...utils.stats import use_sample_std
from ...utils.tee import run_log


def build_parser():
    parser = argparse.ArgumentParser(
        description="T4 under the paper's protocol.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--max-length", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    return parser


def main():
    args = build_parser().parse_args()
    use_sample_std()
    with run_log("t4_anbncn_l%d" % args.max_length):
        print("=== max_length=%d epochs=%d runs=%d"
              % (args.max_length, args.epochs, args.runs), flush=True)
        anbncn_evaluate.main(max_length=args.max_length, epochs=args.epochs,
                             runs=args.runs)


if __name__ == "__main__":
    main()
