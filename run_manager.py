import time

from torch.utils.tensorboard import SummaryWriter


class RunManager:
    def __init__(self,
                 run,
                 dataloader,
                 dataset,
                 tensorboard=True):
        self.tensorboard = tensorboard
        self.lr_calculater = run.lr_calculater(run.max_lr, run.min_lr,
                                               len(dataloader), run.epochs)

        self.dataloader = dataloader
        self.dataset = dataset

        self.epochs = run.epochs
        self.epoch_cnt = 0

        self.str = f"{str(self.lr_calculater)}, epochs={self.epochs}"
        if tensorboard:
            self.tb = SummaryWriter(comment=f"  {self.str}")

        self.start_time = None

        self.batch_cnt = 0

        self.correct = 0
        self.total_loss = 0

    def _calculate_lr(self):
        return self.lr_calculater.calculate_lr(self.batch_cnt)

    def get_learning_rate(self):
        lr = self._calculate_lr()
        if self.tensorboard:
            self.tb.add_scalar("lr per batch", lr, self.batch_cnt)
        return lr

    def end_batch(self, correct=0, loss=0):
        self.correct += correct
        self.total_loss += loss
        self.batch_cnt += 1

    def start_epoch(self):
        self.epoch_cnt += 1
        self.correct = 0
        self.total_loss = 0

    def end_epoch(self):
        print((self.correct / len(self.dataset)) * 100)
        if self.tensorboard:
            self.tb.add_scalar("duration",
                               time.time() - self.start_time,
                               self.epoch_cnt)
            self.tb.add_scalar("lr per epoch",
                               self._calculate_lr(),
                               self.epoch_cnt)
            self.tb.add_scalar("correct",
                               self.correct,
                               self.epoch_cnt)
            self.tb.add_scalar("correct in percent",
                               (self.correct / len(self.dataset)) * 100,
                               self.epoch_cnt)
            self.tb.add_scalar("total loss per epoch",
                               self.total_loss,
                               self.epoch_cnt)
            self.tb.add_scalar("average loss",
                               self.total_loss / (self.batch_cnt * len(
                                   self.dataloader)),
                               self.epoch_cnt)

    def start_run(self, network=None):
        self.start_time = time.time()
        if network is not None and self.tensorboard:
            imgs, _ = next(iter(self.dataloader))
            self.tb.add_graph(network, imgs)

    def end_run(self):
        if self.tensorboard:
            self.tb.close()

    def __str__(self):
        return self.str
