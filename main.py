import time
from collections import namedtuple, OrderedDict
from itertools import product

from lr_calculater import LRLinear, LRExponential
from run_manager import RunManager

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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


def get_num_correct(predicts: torch.Tensor, labels: torch.Tensor):
    return predicts.argmax(dim=1).eq(labels).sum().item()


def train(params, tensorboard=True):
    net = None

    for run in RunBuilder.get_runs(params):

        net = Network()
        optimizer = optim.SGD(net.parameters(), lr=0.03)

        manager = RunManager(
            run,
            data_train_loader,
            data_train_set,
            tensorboard=tensorboard
        )
        print(f"start run: {str(manager)}")
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
    return net


def test(epochs, net):
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


if __name__ == "__main__":
    params = OrderedDict(
        epochs=[3],
        max_lr=[0.05],
        min_lr=[0.01, 0.005, 0.001],
        lr_calculater=[LRExponential, LRLinear]
    )

    net = train(params)
