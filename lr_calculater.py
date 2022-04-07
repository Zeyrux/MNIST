from math import e as euler
from math import pow
from numpy import log


class LRLinear:
    def __init__(self,
                 max_lr: float,
                 min_lr: float,
                 cnt_batches: int,
                 epochs: int):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.step_lr = (max_lr - min_lr) / (cnt_batches * epochs)

    def calculate_lr(self, batch_cnt: int) -> float:
        return self.max_lr - self.step_lr * batch_cnt

    def __str__(self):
        return f"LRLinear(max_lr={self.max_lr}, min_lr={self.min_lr})"


# f(x)=0.51 (((0.5)/(0.51)))^(x)
# f(x)=0.503 (((0.5)/(0.503)))^(x)
# min = b * e ^ ((ln(z) - ln(b)) / x - 1)
class LRExponential:
    def __init__(self,
                 max_lr: float,
                 min_lr: float,
                 cnt_batches: int,
                 epochs: int):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.all_batches = cnt_batches * epochs
        self.calcu_min = self._calculate_min_f(max_lr,
                                               min_lr,
                                               self.all_batches)

    def _calculate_min_f(self, max_lr, min_lr, all_batches):
        return max_lr * ((max_lr / min_lr) ** (1/(all_batches - 1)))

    def calculate_lr(self, batch_cnt: int):
        return self.calcu_min * ((self.max_lr/self.calcu_min) ** (batch_cnt*2))

    def __str__(self):
        return f"LRExponential(max_lr={self.max_lr}, min_lr={self.min_lr})"
