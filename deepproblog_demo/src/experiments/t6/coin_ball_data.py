"""Dataset for T6, built from the paper's description -- it ships none.

    "For the Coin-Urn experiment, we generate the RGB pairs by adding
    Gaussian noise (sigma = 0.03) to the base colours in the HSV domain.
    The coins are MNIST images, where we use even numbers as heads, and
    odd for tails."

What the paper never states is the distribution the balls are drawn from;
the 0.5 / 0.333 in Listing 6 are the initial values of learnable
parameters, not the truth. URN_RATIOS and COIN_BIAS below are therefore
ours, and recovering them is what success means here.
"""

import colorsys
import random

import torch

from ...utils.paths import DATA_DIR

from deepproblog.dataset import Dataset
from deepproblog.query import Query
from problog.logic import Constant, Term

URN_RATIOS = {1: {"red": 0.3, "blue": 0.7}, 2: {"red": 0.2, "green": 0.5, "blue": 0.3}}
COIN_BIAS = 0.6  # P(heads)

BASE_HUES = {"red": 0.0, "green": 1 / 3, "blue": 2 / 3}
HSV_NOISE = 0.03

def sample_colour(rng, name):
    hue = (BASE_HUES[name] + rng.gauss(0, HSV_NOISE)) % 1.0
    saturation = min(max(1.0 + rng.gauss(0, HSV_NOISE), 0.0), 1.0)
    value = min(max(1.0 + rng.gauss(0, HSV_NOISE), 0.0), 1.0)
    return colorsys.hsv_to_rgb(hue, saturation, value)

def sample_from(rng, ratios):
    draw = rng.random()
    for name, probability in ratios.items():
        draw -= probability
        if draw < 0:
            return name
    return name

def outcome(side, colour1, colour2):
    if side == "heads" and (colour1 == "red" or colour2 == "red"):
        return "win"
    if colour1 == colour2:
        return "win"
    return "loss"

class CoinBall(Dataset):

    def __init__(self, subset, size, seed, mnist_offset=0):
        self.subset = subset

        import torchvision
        import torchvision.transforms as transforms

        mnist = torchvision.datasets.MNIST(
            root=str(DATA_DIR / "MNIST"),
            train=(subset != "test"),
            download=True,
            transform=transforms.ToTensor(),
        )

        rng = random.Random(seed)
        order = list(range(len(mnist)))
        random.Random(12345).shuffle(order)
        order = order[mnist_offset:]

        self.coins = []
        self.balls = []
        self.labels = []
        self.coin_sides = []
        self.ball_colours = []
        for i in range(size):
            image, digit = mnist[order[i]]
            side = "heads" if digit % 2 == 0 else "tails"
            colour1 = sample_from(rng, URN_RATIOS[1])
            colour2 = sample_from(rng, URN_RATIOS[2])
            self.coins.append(image)
            self.balls.append(
                (
                    torch.tensor(sample_colour(rng, colour1)),
                    torch.tensor(sample_colour(rng, colour2)),
                )
            )
            self.labels.append(outcome(side, colour1, colour2))
            self.coin_sides.append(side)
            self.ball_colours.append((colour1, colour2))

    def __len__(self):
        return len(self.labels)

    def to_query(self, i):
        arguments = [
            Term("tensor", Term(f"{self.subset}_coin", Constant(i))),
            Term("tensor", Term(f"{self.subset}_ball1", Constant(i))),
            Term("tensor", Term(f"{self.subset}_ball2", Constant(i))),
        ]
        return Query(Term("game", *arguments, Term(self.labels[i])))

class TensorSource:

    def __init__(self, dataset, which):
        self.dataset = dataset
        self.which = which

    def __getitem__(self, item):
        index = int(item[0])
        if self.which == "coin":
            return self.dataset.coins[index]
        return self.dataset.balls[index][0 if self.which == "ball1" else 1]
