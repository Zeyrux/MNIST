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


class LRExponential:
    def __init__(self,
                 max_lr: float,
                 min_lr: float,
                 cnt_batches: int,
                 epochs: int):
        pass

    def calculate_lr(self, batch_cnt: int):
        pass
