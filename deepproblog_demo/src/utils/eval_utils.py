"""Evaluation helpers.

get_confusion_matrix is adapted from deepproblog.evaluate (https://github.com/ML-KULeuven/deepproblog); the only
change is the tqdm wrapper. macro_f1 is ours -- the library's
ConfusionMatrix exposes accuracy() and nothing else.
"""

from typing import Optional

from tqdm import tqdm

from deepproblog.dataset import Dataset
from deepproblog.model import Model
from deepproblog.utils.confusion_matrix import ConfusionMatrix

def get_confusion_matrix(
    model: Model, dataset: Dataset, verbose: int = 0, eps: Optional[float] = None
) -> ConfusionMatrix:
    confusion_matrix = ConfusionMatrix()
    model.eval()
    queries = tqdm(dataset.to_queries(), total=len(dataset), desc="evaluating", leave=False)
    for i, gt_query in enumerate(queries):
        test_query = gt_query.variable_output()
        answer = model.solve([test_query])[0]
        actual = str(gt_query.output_values()[0])
        if len(answer.result) == 0:
            predicted = "no_answer"
            if verbose > 1:
                print("no answer for query {}".format(gt_query))
        else:
            max_ans = max(answer.result, key=lambda x: answer.result[x])
            p = answer.result[max_ans]
            if eps is None:
                predicted = str(max_ans.args[gt_query.output_ind[0]])
            else:
                predicted = float(max_ans.args[gt_query.output_ind[0]])
                actual = float(gt_query.output_values()[0])
                if abs(actual - predicted) < eps:
                    predicted = actual
            if verbose > 1 and actual != predicted:
                print(
                    "{} {} vs {}::{} for query {}".format(
                        i, actual, p, predicted, test_query
                    )
                )
        confusion_matrix.add_item(predicted, actual)

    if verbose > 0:
        print(confusion_matrix)
        print("Accuracy", confusion_matrix.accuracy())

    return confusion_matrix

def macro_f1(confusion_matrix: ConfusionMatrix) -> float:
    matrix = confusion_matrix.matrix
    f1s = []
    for c in range(confusion_matrix.n):
        support = matrix[:, c].sum()
        if support == 0:
            continue
        true_positives = matrix[c, c]
        predicted = matrix[c, :].sum()
        precision = true_positives / predicted if predicted else 0.0
        recall = true_positives / support
        f1s.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return sum(f1s) / len(f1s) if f1s else 0.0

def scores(confusion_matrix: ConfusionMatrix):
    return [
        ("Accuracy", confusion_matrix.accuracy()),
        ("F1", macro_f1(confusion_matrix)),
    ]
