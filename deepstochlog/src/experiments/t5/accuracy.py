"""Adapted from examples/citeseer/citeseer_utils.py
(github.com/ML-KULeuven/deepstochlog). Documents are scored over processes and
summed in dataset order, the order the upstream loop sums in.
"""

from time import time

import torch

from deepstochlog.context import ContextualizedTerm
from deepstochlog.utils import calculate_zipped_probabilities

from ...utils.parallel import run_sharded


def _score_document(model, elem, query_with_variable):
    other_possibilities = [
        ContextualizedTerm(context=elem.context, term=t)
        for t in model.get_direct_proof_possibilities(query_with_variable)
    ]
    probabilities = dict(calculate_zipped_probabilities(model, other_possibilities))
    own_prob = probabilities[elem]
    is_highest = not any(
        pos for pos in other_possibilities if probabilities[pos] > own_prob
    )
    return bool(is_highest), float(own_prob)


def score_all(model, data, workers):
    def work(model, indices):
        return [
            (i, _score_document(model, data[i], data.queries_for_model[i]))
            for i in indices
        ]

    scores = [None] * len(data)
    for payload in run_sharded(model, len(data), workers, work):
        for i, score in payload:
            scores[i] = score
    return scores


def calculate_accuracy(model, test_data, workers=1):
    if len(test_data) == 0:
        return 0, 0
    scores = score_all(model, test_data, workers)
    test_acc = sum(1 for is_highest, _ in scores if is_highest)
    label_probability = 0.0
    for _, own_prob in scores:
        label_probability += own_prob
    return test_acc / len(test_data), label_probability / len(test_data)


class ParallelAccuracyCalculator:
    header = "Valid acc\tTest acc\tP(correct)\ttime"

    def __init__(self, model, valid, test, start_time, after_epoch=0, workers=1):
        self.model = model
        self.valid_set = valid
        self.test_set = test
        self.max_val = 0.0
        self.current_test = 0.0
        self.start_time = start_time
        self.after_epoch = after_epoch
        self.workers = workers
        self.epoch = 0

    def __call__(self):
        self.epoch += 1
        if self.after_epoch < self.epoch:
            for network in self.model.neural_networks.networks.values():
                network.neural_model.eval()
            acc, average_right_prob = calculate_accuracy(
                self.model, self.valid_set, self.workers
            )
            if acc > self.max_val:
                self.max_val = acc
                self.current_test, _ = calculate_accuracy(
                    self.model, self.test_set, self.workers
                )
            for network in self.model.neural_networks.networks.values():
                network.neural_model.train()
        else:
            acc = 0
            average_right_prob = 0
        return "{:.3f}\t\t{:.3f}\t\t{:.3f}\t\t{:.3f}".format(
            acc, self.current_test, average_right_prob, time() - self.start_time
        )
