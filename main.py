import time
from collections import namedtuple, OrderedDict
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms

data_train_set = torchvision.datasets.MNIST(
    root="data\\train",
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
)
data_train_loader = torch.utils.data.DataLoader(
    data_train_set,
    batch_size=64,
    shuffle=True
)

data_test_set = torchvision.datasets.MNIST(
    root="data\\test",
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
)
data_test_loader = torch.utils.data.DataLoader(
    data_test_set,
    batch_size=64,
    shuffle=True
)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=4)
        # self.conv_dropout = nn.Dropout2d()

        self.fc1 = nn.Linear(in_features=32 * 4 * 4, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=10)

    def forward(self, t: torch.Tensor):
        t = self.conv1(t)
        t = F.max_pool2d(t, kernel_size=2)
        t = F.relu(t)

        t = self.conv2(t)
        # t = self.conv_dropout(t)
        t = F.max_pool2d(t, kernel_size=2)
        t = F.relu(t)

        t = t.reshape(-1, 32 * 4 * 4)

        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        return F.relu(self.out(t))


class RunBuilder:
    @staticmethod
    def get_runs(params: dict) -> list:
        Run = namedtuple("Run", params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs


class RunManager:
    def __init__(self,
                 max_lr,
                 min_lr,
                 epochs,
                 dataloader,
                 dataset,
                 comment=None,
                 tensorboard=True):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.tensorboard = tensorboard

        self.dataloader = dataloader
        self.dataset = dataset

        if tensorboard:
            if comment is not None:
                self.tb = SummaryWriter(comment=comment)
            else:
                self.tb = SummaryWriter()

        self.epochs = epochs
        self.epoch_cnt = 0

        self.start_time = None

        self.batch_cnt = 0

        self.step_lr = (max_lr - min_lr) / (len(dataloader) * self.epochs)

        self.correct = 0
        self.total_loss = 0

    def _calculate_lr(self):
        return self.max_lr - self.step_lr * self.batch_cnt

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


net = Network()
optimizer = optim.SGD(net.parameters(), lr=0.03)


def get_num_correct(predicts: torch.Tensor, labels: torch.Tensor):
    return predicts.argmax(dim=1).eq(labels).sum().item()


def train(params, tensorboard=True):
    for run in RunBuilder.get_runs(params):

        print(f"start run: {str(run)}")

        manager = RunManager(
            run.max_lr,
            run.min_lr,
            run.epochs,
            data_train_loader,
            data_train_set,
            comment=str(run),
            tensorboard=tensorboard
        )
        manager.start_run(network=net)

        for _ in range(manager.epochs):

            manager.start_epoch()

            for images, labels in manager.dataloader:
                optimizer.param_groups[0]["lr"] = manager.get_learning_rate()

                predicts = net(images)
                loss = F.cross_entropy(predicts, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                manager.end_batch(correct=get_num_correct(predicts, labels),
                                  loss=loss.item())

            manager.end_epoch()

        manager.end_run()


def test(epochs):
    for epoch in range(epochs):

        start_time = time.time()
        correct = 0
        total_loss = 0

        for images, labels in data_test_loader:
            predicts = net(images)
            total_loss += F.cross_entropy(predicts, labels).item()
            correct += get_num_correct(predicts, labels)

        print("\nEpoch:", epoch + 1)
        print("Correct:", correct, "/", len(data_test_set))
        print("Percent:", str((correct / len(data_test_set)) * 100) + "%")
        print("Total loss:", total_loss)
        print("Average loss:", total_loss / len(data_test_set))
        print("Duration:", time.time() - start_time)


params = OrderedDict(
    epochs=[5],
    max_lr=[0.1],
    min_lr=[0.001]
)

# f(x)=0.51 (((0.5)/(0.51)))^(x)
train(params)
# print("\n\nTEST")
# test(2)
