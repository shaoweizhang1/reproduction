"""T1 addition: upstream training with a selectable accuracy pass.

Adapted from examples/addition/addition_evaluate.py
(github.com/ML-KULeuven/deepstochlog). The accuracy pass is either spread over
CPU processes or replaced by the exact evaluator in fast_accuracy.py.
"""

import argparse

import deepstochlog.trainer
import deepstochlog.utils
from deepstochlog.utils import calculate_zipped_probabilities
from examples.addition import addition_evaluate

from ...utils.parallel import run_sharded
from ...utils.stats import use_sample_std
from ...utils.tee import run_log
from .fast_accuracy import calculate_accuracy as sum_dp_accuracy


def build_parser():
    parser = argparse.ArgumentParser(
        description="T1 under the paper's protocol.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--digit-length", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--accuracy-mode", choices=["engine", "sum_dp"], default="engine")
    return parser


def score_batch(model, batch):
    other_possibilities = {
        elem: model.get_direct_contextualized_proof_possibilities(elem)
        for elem in batch
    }
    all_to_evaluate = [
        possibility
        for possibilities in other_possibilities.values()
        for possibility in possibilities
    ]
    probabilities = dict(calculate_zipped_probabilities(model, all_to_evaluate))
    return [
        (
            not any(
                probabilities[possibility] > probabilities[elem]
                for possibility in other_possibilities[elem]
            ),
            float(probabilities[elem]),
        )
        for elem in batch
    ]


def calculate_accuracy(
    model, test_data, workers=1, most_probable_parse_accuracy=False, **kwargs
):
    """Adapted from deepstochlog.utils.calculate_accuracy; batches are scored over processes."""
    if most_probable_parse_accuracy:
        raise NotImplementedError("--workers covers generation_output_accuracy only")
    batches = list(test_data)
    num_instances = sum(len(batch) for batch in batches)
    if num_instances == 0:
        return 0, 0, 0

    def work(model, indices):
        return [(i, score_batch(model, batches[i])) for i in indices]

    scores = [None] * len(batches)
    for payload in run_sharded(model, len(batches), workers, work):
        for i, batch_scores in payload:
            scores[i] = batch_scores
    scores = [score for batch_scores in scores for score in batch_scores]
    return (
        sum(1 for is_highest, _ in scores if is_highest) / num_instances,
        sum(own_prob for _, own_prob in scores) / num_instances,
        0,
    )


def patch_workers(workers):
    def with_workers(model, test_data, **kwargs):
        return calculate_accuracy(model, test_data, workers=workers, **kwargs)

    deepstochlog.utils.calculate_accuracy = with_workers


def patch_progress():
    original = deepstochlog.trainer.PandasLogger.log

    def with_progress(self, *args, **kwargs):
        original(self, *args, **kwargs)
        print(
            "  " + "  ".join("%s=%g" % item for item in self.df.iloc[-1].items()),
            flush=True,
        )

    deepstochlog.trainer.PandasLogger.log = with_progress


def main():
    args = build_parser().parse_args()
    if args.accuracy_mode == "sum_dp":
        if args.workers != 1:
            raise ValueError("sum_dp accuracy does not use worker processes")
        deepstochlog.utils.calculate_accuracy = sum_dp_accuracy
    elif args.workers > 1:
        patch_workers(args.workers)
    patch_progress()
    use_sample_std()
    name = "t1_addition_d%d_e%d" % (args.digit_length, args.epochs)
    if args.workers > 1:
        name += "_w%d" % args.workers
    if args.accuracy_mode == "sum_dp":
        name += "_sumdp"
    with run_log(name):
        print(
            "=== digit_length=%d epochs=%d runs=%d workers=%d accuracy_mode=%s"
            % (args.digit_length, args.epochs, args.runs, args.workers,
               args.accuracy_mode),
            flush=True,
        )
        addition_evaluate.main(
            digit_length=args.digit_length, epochs=args.epochs, runs=args.runs
        )


if __name__ == "__main__":
    main()
