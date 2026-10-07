"""T3 parentheses: the official Table 3 protocol, with a log per setting."""

import argparse

from examples.bracket import bracket_evaluate

from ...utils.stats import use_sample_std
from ...utils.tee import run_log


def build_parser():
    parser = argparse.ArgumentParser(
        description="T3 under the paper's protocol.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--max-length", type=int, default=10)
    return parser


def main():
    args = build_parser().parse_args()
    use_sample_std()
    with run_log("t3_bracket_l%d" % args.max_length):
        print("=== max_length=%d" % args.max_length, flush=True)
        bracket_evaluate.main(max_length=args.max_length)


if __name__ == "__main__":
    main()
