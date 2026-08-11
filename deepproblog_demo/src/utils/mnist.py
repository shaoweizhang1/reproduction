"""Machinery shared by the T1 and T2 MNIST-addition experiments.

Sources, all https://github.com/ML-KULeuven/deepproblog:
  - Separate_Baseline / Separate_Baseline_Multi are copied verbatim from
    examples/MNIST/neural_baseline/baseline_models.py, which the pip
    package does not ship, so they cannot simply be imported.
  - run_baseline follows neural_baseline/{baseline,multidigit_baseline}.py.
  - run_deepproblog follows examples/MNIST/addition.py.
Concat_Baseline has no counterpart upstream. Deviations from the originals
are in the t{1,2}/ headers.
"""

import time
from json import dumps

import torch
import torch.nn as nn
from tqdm import tqdm

from .device import GpuNetwork, move, resolve
from .eval_utils import get_confusion_matrix, macro_f1, scores
from .paths import LOG_DIR, example_dir
from .stop_conditions import IterationStop

from deepproblog.dataset import DataLoader
from deepproblog.engines import ExactEngine
from deepproblog.examples.MNIST.data import MNIST_train, MNIST_test, addition
from deepproblog.examples.MNIST.network import MNIST_Net
from deepproblog.model import Model
from deepproblog.train import train_model
from deepproblog.utils.confusion_matrix import ConfusionMatrix
from deepproblog.utils.logger import Logger

class Concat_Baseline(nn.Module):
    """The CNN of the paper's Figure 3a, not released. Rebuilt from the
    architecture in the appendix, with 19 outputs and the two images
    stacked as channels -- that keeps the 16x4x4 feature map the
    description implies, where side-by-side images would give 16x4x11 and
    a first linear layer of 84600 parameters. The appendix's 44k is the
    digit network (1 channel, 10 outputs, 44426 here); this one has 45341.
    """

    def __init__(self):
        super(Concat_Baseline, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 19),
        )

    def forward(self, x, y):
        x = self.encoder(torch.cat([x, y], dim=1))
        return self.classifier(x.view(-1, 16 * 4 * 4))

class Separate_Baseline(nn.Module):
    def __init__(self, batched=False, probabilities=True):
        super(Separate_Baseline, self).__init__()
        self.batched = batched
        self.probabilities = probabilities
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 8 * 2, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 19),
        )
        self.activation = nn.Softmax(dim=-1)

    def forward(self, x, y):
        if not self.batched:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        x = self.encoder(x)
        y = self.encoder(y)
        x = x + y
        x = x.view(-1, 16 * 8 * 2)
        x = self.classifier(x)
        if self.probabilities:
            x = self.activation(x)
        if not self.batched:
            x = x.squeeze(0)

        return x

class Separate_Baseline_Multi(nn.Module):
    def __init__(self, n=4):
        super(Separate_Baseline_Multi, self).__init__()
        self.n = n
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(16 * 4 * 4 * self.n // 2, 100),
        )
        self.classifier2 = nn.Sequential(
            nn.ReLU(), nn.Linear(100 * 2, 128), nn.ReLU(), nn.Linear(128, 199)
        )

    def forward(self, imgs1, imgs2):
        imgs1 = [self.encoder(x) for x in imgs1]
        imgs2 = [self.encoder(x) for x in imgs2]
        x1, x2 = torch.cat(imgs1, 2), torch.cat(imgs2, 2)
        x1, x2 = (
            x1.view(-1, 16 * 4 * 4 * self.n // 2),
            x2.view(-1, 16 * 4 * 4 * self.n // 2),
        )
        x1, x2 = self.classifier(x1), self.classifier(x2)
        x = torch.cat([x1, x2], 1)
        x = self.classifier2(x)
        return x

BASELINES = {"concat": Concat_Baseline, "separate": Separate_Baseline}

def run_deepproblog(n_digits, program, args, name):
    device = resolve(args.device)
    train_set, test_set = addition(n_digits, "train"), addition(n_digits, "test")

    network = MNIST_Net()
    net = GpuNetwork(network, "mnist_net", batching=True)
    net.optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    if device == "cuda":
        net.cuda()

    model = Model(str(example_dir("MNIST", "models", program)), [net])
    model.set_engine(ExactEngine(model), cache=True)
    model.add_tensor_source("train", MNIST_train)
    model.add_tensor_source("test", MNIST_test)

    LOG_DIR.mkdir(exist_ok=True)
    train = train_model(
        model,
        DataLoader(train_set, args.batch_size, False),
        IterationStop(args.total_iterations),
        log_iter=args.log_period,
        test_iter=args.test_period,
        test=lambda m: scores(get_confusion_matrix(m, test_set.subset(args.test_subset))),
        profile=0,
    )
    train.logger.comment(dumps(model.get_hyperparameters()))
    final = get_confusion_matrix(model, test_set, verbose=1)
    train.logger.comment("F1\t{}".format(macro_f1(final)))
    train.logger.comment("Accuracy\t{}".format(final.accuracy()))
    train.logger.write_to_file(str(LOG_DIR / name))

def evaluate(net, dataset, unwrap, device="cpu"):
    confusion_matrix = ConfusionMatrix()
    for x, y, label in tqdm(dataset, desc="testing", leave=False):
        inputs = (move(unwrap(x), device), move(unwrap(y), device))
        predicted = torch.max(net(*inputs).data, 1)[1]
        confusion_matrix.add_item(int(predicted.squeeze()), int(label))
    return confusion_matrix

def run_baseline(net, n_digits, args, name, train_unwrap, test_unwrap):
    device = resolve(args.device)
    net = net.to(device)
    train_set, test_set = addition(n_digits, "train"), addition(n_digits, "test")
    loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True
    )
    optimizer = torch.optim.Adam(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    log = Logger()

    LOG_DIR.mkdir(exist_ok=True)
    started = time.time()
    running_loss, i, epoch = 0.0, 1, 1
    while i <= args.total_iterations:
        for x, y, label in tqdm(loader, desc=f"epoch {epoch}", leave=False):
            if i > args.total_iterations:
                break
            optimizer.zero_grad()
            inputs = (move(train_unwrap(x), device), move(train_unwrap(y), device))
            loss = criterion(net(*inputs), label.to(device))
            loss.backward()
            optimizer.step()
            running_loss += float(loss)
            if i % args.log_period == 0:
                log.log("time", i, time.time() - started)
                log.log("loss", i, running_loss / args.log_period)
                running_loss = 0.0
            if i % args.test_period == 0:
                matrix = evaluate(
                    net, test_set.subset(args.test_subset), test_unwrap, device
                )
                for series, value in scores(matrix):
                    log.log(series, i, value)
            i += 1
        epoch += 1

    final = evaluate(net, test_set, test_unwrap, device)
    log.comment("F1\t{}".format(macro_f1(final)))
    log.comment("Accuracy\t{}".format(final.accuracy()))
    log.write_to_file(str(LOG_DIR / name))
