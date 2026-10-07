"""T5 citation classification with the paper's depth-2 program and Linear(50).

The official Citeseer runner is used; Cora's base runner is adapted locally.
Validation selects the model. Each run prints its selected test accuracy;
multiple runs print their mean and sample standard deviation.
"""
import argparse
import importlib
import sys
import types
from statistics import mean, stdev

# DGL 2.1 has no GraphBolt binary for the pinned PyTorch version.
# Citation datasets do not use GraphBolt; bypass its eager import for T5.
_graphbolt = types.ModuleType("dgl.graphbolt")
_graphbolt.__path__ = []
sys.modules.setdefault("dgl.graphbolt", _graphbolt)

from examples.citeseer import citeseer_utils

from ...utils.paths import PROGRAMS_DIR
from ...utils.tee import run_log
from .accuracy import ParallelAccuracyCalculator


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=["citeseer", "cora"], default="citeseer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--log-freq", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")

    module = importlib.import_module(
        "examples.citeseer.base.citeseer" if args.dataset == "citeseer"
        else "src.experiments.t5.cora"
    )
    module.root_path = PROGRAMS_DIR / "depth2"
    num_outputs = 6 if args.dataset == "citeseer" else 7

    def classifier(*a, **kw):
        kw["num_outputs"] = num_outputs
        return citeseer_utils.Classifier(*a, **kw)

    module.Classifier = classifier
    calculator = None
    calculator_type = (ParallelAccuracyCalculator if args.workers > 1
                       else module.AccuracyCalculator)

    def make_accuracy(*a, **kw):
        nonlocal calculator
        if args.workers > 1:
            kw["workers"] = args.workers
        calculator = calculator_type(*a, **kw)
        return calculator

    module.AccuracyCalculator = make_accuracy
    batch_size = args.batch_size or len(module.train_dataset)
    name = "t5_%s_seed%d_runs%d_w%d" % (
        args.dataset, args.seed, args.runs, args.workers
    )
    with run_log(name):
        scores = []
        for seed in range(args.seed, args.seed + args.runs):
            print("=== dataset=%s seed=%d epochs=%d batch=%d workers=%d device=%s"
                  % (args.dataset, seed, args.epochs, batch_size,
                     args.workers, args.device), flush=True)
            module.run(epochs=args.epochs, batch_size=batch_size, seed=seed,
                       lr=args.lr, log_freq=args.log_freq, verbose=True,
                       device_str=args.device)
            scores.append(100 * calculator.current_test)
            print("T5 seed=%d: best validation=%.2f%% selected test=%.2f%%"
                  % (seed, 100 * calculator.max_val, scores[-1]), flush=True)
        print("T5 %s: runs=%d test accuracy=%.2f +- %.2f%% (sample SD)"
              % (args.dataset, len(scores), mean(scores),
                 stdev(scores) if len(scores) > 1 else 0.0), flush=True)


if __name__ == "__main__":
    main()
