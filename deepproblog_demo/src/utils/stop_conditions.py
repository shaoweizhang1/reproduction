"""IterationStop -- ours, not from the library, which stops only on whole
epochs. Lets runs with different batch sizes share one iteration budget.
"""

from deepproblog.utils.stop_condition import StopCondition


class IterationStop(StopCondition):
    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations

    def __str__(self):
        return "for {} iterations".format(self.max_iterations)

    def is_stop(self, train_object):
        return train_object.i >= self.max_iterations
