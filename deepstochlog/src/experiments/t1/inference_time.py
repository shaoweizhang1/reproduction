"""Table 7: wraps examples/addition/addition_timing.py
(github.com/ML-KULeuven/deepstochlog), 100 training queries per run."""
import argparse
from statistics import mean, stdev

import torch
from examples.addition import addition_timing

from ...utils.tee import run_log


def build_parser():
    parser = argparse.ArgumentParser(
        description="Table 7 under the paper's protocol.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lengths", type=int, nargs="+", choices=[1, 2, 3, 4],
                        default=[1, 2, 3, 4])
    parser.add_argument("--runs", type=int, default=5)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")

    with run_log("t1_table7"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Table 7: device=%s threads=%d torch=%s queries=100 runs=%d"
              % (device, torch.get_num_threads(), torch.__version__, args.runs),
              flush=True)
        results = []
        for length in args.lengths:
            run_means = []
            for seed in range(args.runs):
                print("=== digits=%d seed=%d ===" % (length, seed), flush=True)
                _, avg, query_sd = addition_timing.run(
                    digit_length=length, seed=seed, verbose=True
                )
                run_means.append(1000 * avg)
                print("Run: digits=%d seed=%d mean_ms=%.6f query_sd_ms=%.6f"
                      % (length, seed, 1000 * avg, 1000 * query_sd), flush=True)
            results.append((length, mean(run_means),
                            stdev(run_means) if args.runs > 1 else 0.0))
        print("\nTable 7: mean and sample SD of run means (milliseconds)")
        print("| Digits per number | Mean ± SD |")
        print("|---|---:|")
        for length, avg, sd in results:
            print("| %d | %.3f ± %.3f |" % (length, avg, sd), flush=True)


if __name__ == "__main__":
    main()
