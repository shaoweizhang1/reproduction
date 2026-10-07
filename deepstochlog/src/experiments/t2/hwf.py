"""T2 handwritten expressions under the paper's protocol.

Training and greedy evaluation are provided by the official
examples/mathexpression/mathexpression.py; lengths and seeds are selected here.
--model paper replaces the released model with the one in the paper's appendix:
separate digit and operator networks, and neural switches over the term and
expression rules (src/programs/hwf_paper/mathexpression.pl) instead of fixed
0.34/0.33 rule probabilities.
"""

import argparse
import re
from pathlib import Path
from statistics import mean, stdev

import torch
import torch.nn.functional as F
from deepstochlog.network import Network, NetworkStore
from deepstochlog.term import Term
from examples.mathexpression import mathexpression
from examples.mathexpression.mathexpression_data import operator_word_list
from examples.models import SymbolClassifier, SymbolEncoder

from ...utils.paths import PROGRAMS_DIR
from ...utils.tee import run_log


def build_parser():
    parser = argparse.ArgumentParser(
        description="T2 under the paper's protocol.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--length", type=int, default=3,
                        help="expression length; the paper's columns are 1, 3, 5, 7")
    parser.add_argument("--model", choices=["released", "paper"], default="released")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--report", type=Path, default=None,
                        help="print mean and sample SD over seed*/ logs in this directory")
    return parser


class Switch(torch.nn.Module):
    """A learned distribution over rules; the input is empty."""

    def __init__(self, n=3):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(n))

    def forward(self, x):
        return F.softmax(self.logits, dim=-1).expand(len(x), -1)


def paper_networks():
    switch_terms = [Term(str(i)) for i in range(3)]
    return NetworkStore(
        Network("number", SymbolClassifier(SymbolEncoder(), N=10),
                index_list=[Term(str(i)) for i in range(10)]),
        Network("operator", SymbolClassifier(SymbolEncoder(), N=4),
                index_list=[Term(op) for op in operator_word_list]),
        Network("term_switch", Switch(), index_list=switch_terms),
        Network("expression_switch", Switch(), index_list=switch_terms),
    )


def report(root, model):
    suffix = "_paper" if model == "paper" else ""
    for length in (1, 3, 5, 7):
        scores = []
        for log in sorted(root.glob("seed[0-9]*/t2_hwf_l%d%s.log" % (length, suffix))):
            if not log.parent.name[4:].isdigit():
                continue
            found = re.search(r"^Test accuracy:([0-9.]+)", log.read_text(), re.M)
            if found:
                scores.append(100 * float(found[1]))
        if scores:
            print("T2 length=%d model=%s: runs=%d test accuracy=%.2f +- %.2f%% (sample SD)"
                  % (length, model, len(scores), mean(scores),
                     stdev(scores) if len(scores) > 1 else 0.0), flush=True)


def main():
    args = build_parser().parse_args()
    if args.report is not None:
        return report(args.report, args.model)
    name = "t2_hwf_l%d" % args.length
    if args.model == "paper":
        mathexpression.load_expression_networks = paper_networks
        mathexpression.root_path = PROGRAMS_DIR / "hwf_paper"
        name += "_paper"
    with run_log(name):
        scores = []
        for i in range(args.runs):
            print(
                "=== length=%d model=%s epochs=%d batch=%d lr=%g seed=%d device=%s run=%d/%d"
                % (args.length, args.model, args.epochs, args.batch_size, args.lr,
                   args.seed + i, args.device, i + 1, args.runs),
                flush=True,
            )
            score = mathexpression.run(
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                expression_length=args.length,
                expression_max_length=args.length,
                device_str=args.device,
                train_size=args.train_size,
                test_size=args.test_size,
                log_freq=args.log_freq,
                seed=args.seed + i,
                verbose=True,
            )
            scores.append(100 * score)
        print("T2 length=%d model=%s: runs=%d test accuracy=%.2f +- %.2f%% (sample SD)"
              % (args.length, args.model, len(scores), mean(scores),
                 stdev(scores) if len(scores) > 1 else 0.0), flush=True)


if __name__ == "__main__":
    main()
