"""SGD on the logic parameters, with the paper's warm-up.

deepproblog.model.Model defaults to the base Optimizer, whose
add_parameter_gradient is a no-op, so the probabilistic parameters are
never updated; deepproblog.optimizer.SGD (https://github.com/ML-KULeuven/deepproblog) is what updates them. The
library has no warm-up, so get_lr implements the paper's: 0.0001 rising
linearly to 0.01 over four epochs.
"""

from deepproblog.optimizer import SGD


class WarmUpSGD(SGD):
    def __init__(self, model, start_lr=1e-4, end_lr=1e-2, warm_up_epochs=4):
        super().__init__(model, start_lr)
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.warm_up_epochs = warm_up_epochs

    def get_lr(self):
        if self.warm_up_epochs <= 0 or self.epoch >= self.warm_up_epochs:
            return self.end_lr
        progress = self.epoch / self.warm_up_epochs
        return self.start_lr + progress * (self.end_lr - self.start_lr)
