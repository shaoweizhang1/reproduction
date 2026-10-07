"""Tiny CPU checks of every entry point; no saved experiment logs are required."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

JOBS = {
    "T1": "from examples.addition.addition import run; run(digit_length=1, epochs=1, batch_size=2, train_size=2, val_size=2, test_size=2, log_freq=1, seed=0)",
    "T2": "from examples.mathexpression.mathexpression import run; run(expression_length=1, epochs=1, batch_size=2, train_size=2, test_size=2, log_freq=1, seed=0, device_str=\"cpu\")",
    "T3": "from examples.bracket.bracket import run; run(min_length=2, max_length=2, epochs=1, batch_size=2, train_size=2, val_size=2, test_size=2, log_freq=1, seed=0)",
    "T4": "from examples.anbncn.anbncn import run; run(min_length=3, max_length=3, epochs=1, batch_size=2, train_size=2, val_size=2, test_size=2, log_freq=1, seed=0)",
    "T6": "from examples.wap.wap import run; run(epochs=1, batch_size=2, train_size=2, val_size=2, test_size=2, log_freq=1, seed=0)",
}
TASKS = ["T1", "T2", "T3", "T4", "T5", "T6", "Table6"]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    return parser


def run(cmd, timeout):
    subprocess.run(cmd, check=True, timeout=timeout)


def main():
    args = build_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for task in args.tasks:
        print("CHECK: " + task, flush=True)
        if task in JOBS:
            run([sys.executable, "-c", JOBS[task]], 180)
            if task == "T1":
                fast = ("import deepstochlog.utils as u; "
                        "from src.experiments.t1.fast_accuracy import calculate_accuracy; "
                        "u.calculate_accuracy = calculate_accuracy; " + JOBS[task])
                run([sys.executable, "-c", fast], 180)
        elif task == "T5":
            for dataset in ("citeseer", "cora"):
                run([sys.executable, "-m", "src.experiments.t5.citation",
                     "--dataset", dataset, "--seed", "0", "--epochs", "1",
                     "--workers", "2", "--device", "cpu"], 300)
        else:
            run_dir = Path(os.environ["DEEPSTOCHLOG_LOG_DIR"])
            run([sys.executable, "-m", "src.experiments.t2.table6_unified", "validate",
                 "--release", str(run_dir / "release"),
                 "--out", str(run_dir / "table6-validation")], 180)
    print("ALL REQUESTED CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
