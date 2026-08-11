"""T6: the coin-ball problem, reconstructed from the paper's Listing 6.

Nothing here is adapted from the library: its Coins example is a different
task (two coins in one image, no learnable parameters). Program, data and
runner are ours -- see coin_ball.pl and coin_ball_data.py.

Run: python3 -m src.experiments.t6.coin_ball [flags]
"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn

from .coin_ball_data import COIN_BIAS, URN_RATIOS, CoinBall, TensorSource
from .optimizer import WarmUpSGD
from ...utils.eval_utils import get_confusion_matrix, macro_f1
from ...utils.paths import LOG_DIR

from deepproblog.dataset import DataLoader
from deepproblog.engines import ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.utils.stop_condition import EpochStop

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_size", type=int, default=256)
    parser.add_argument("--test_size", type=int, default=64)
    parser.add_argument("--dev_size", type=int, default=64)
    # The paper gives the two networks different rates: "The learning rate is
    # 0.001 for the MNIST network, and 1 for the colour network."
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--colour_lr", type=float, default=1.0)
    parser.add_argument("--param_lr", type=float, default=1e-2)
    parser.add_argument("--param_warm_up_lr", type=float, default=1e-4)
    parser.add_argument("--param_warm_up_epochs", type=int, default=4)
    parser.add_argument("--log_period", type=int, default=20)
    parser.add_argument("--test_period", type=int, default=52)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--program",
        default="coin_ball_ad.pl",
        choices=["coin_ball.pl", "coin_ball_ad.pl", "coin_ball_nobias.pl"],
    )
    return parser

class CoinNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5), nn.MaxPool2d(2, 2), nn.ReLU(True),
            nn.Conv2d(6, 16, 5), nn.MaxPool2d(2, 2), nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120), nn.ReLU(), nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, 2), nn.Softmax(dim=-1),
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x.view(-1, 16 * 4 * 4))

class ColourNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(3, 3), nn.Softmax(dim=-1))

    def forward(self, x):
        return self.classifier(x.view(-1, 3))

if __name__ == "__main__":
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)

    train_set = CoinBall("train", args.train_size, seed=args.seed)
    dev_set = CoinBall("dev", args.dev_size, seed=args.seed + 1, mnist_offset=args.train_size)
    test_set = CoinBall("test", args.test_size, seed=args.seed + 2)

    coin_net, colour_net = CoinNet(), ColourNet()
    networks = [
        Network(coin_net, "m_coin", torch.optim.Adam(coin_net.parameters(), lr=args.lr), batching=True),
        Network(colour_net, "m_colour", torch.optim.Adam(colour_net.parameters(), lr=args.colour_lr), batching=True),
    ]

    model = Model(str(Path(__file__).parent / args.program), networks)
    model.set_engine(ExactEngine(model), cache=True)
    for split, dataset in (("train", train_set), ("dev", dev_set), ("test", test_set)):
        for which in ("coin", "ball1", "ball2"):
            model.add_tensor_source(f"{split}_{which}", TensorSource(dataset, which))
    model.optimizer = WarmUpSGD(
        model,
        start_lr=args.param_warm_up_lr,
        end_lr=args.param_lr,
        warm_up_epochs=args.param_warm_up_epochs,
    )

    LOG_DIR.mkdir(exist_ok=True)
    with TemporaryDirectory() as tmp:
        checkpoint = f"{tmp}/best.pth"
        best = {"dev": -1.0}  # a dict, since this runs at module scope

        def score(m):
            dev_accuracy = get_confusion_matrix(m, dev_set).accuracy()
            if dev_accuracy > best["dev"]:
                best["dev"] = dev_accuracy
                m.save_state(checkpoint)
            return [("Accuracy", dev_accuracy)]

        train = train_model(
            model,
            DataLoader(train_set, args.batch_size),
            EpochStop(args.epochs),
            log_iter=args.log_period,
            test_iter=args.test_period,
            test=score,
        )
        final_test = get_confusion_matrix(model, test_set).accuracy()
        if best["dev"] >= 0:
            model.load_state(checkpoint)

    selected = get_confusion_matrix(model, test_set, verbose=1)
    accuracy = selected.accuracy()
    train.logger.comment("Best dev accuracy\t{}".format(best["dev"]))
    train.logger.comment("Final-model test accuracy\t{}".format(final_test))

    with torch.no_grad():
        sides = ["heads", "tails"]
        colours = ["red", "green", "blue"]
        coin_hits = sum(
            sides[int(coin_net(image.unsqueeze(0)).argmax())] == truth
            for image, truth in zip(test_set.coins, test_set.coin_sides)
        )
        ball_hits, ball_total = 0, 0
        for (ball1, ball2), (truth1, truth2) in zip(test_set.balls, test_set.ball_colours):
            for ball, truth in ((ball1, truth1), (ball2, truth2)):
                ball_hits += colours[int(colour_net(ball.unsqueeze(0)).argmax())] == truth
                ball_total += 1
    coin_accuracy = coin_hits / len(test_set.coins)
    colour_accuracy = ball_hits / ball_total
    print(f"\ncoin network accuracy  : {coin_accuracy:.3f}")
    print(f"colour network accuracy: {colour_accuracy:.3f}")
    train.logger.comment("Coin network accuracy\t{}".format(coin_accuracy))
    train.logger.comment("Colour network accuracy\t{}".format(colour_accuracy))
    print("\nlearned parameters vs ground truth")
    print("  ground truth urns:", URN_RATIOS, " coin P(heads):", COIN_BIAS)
    print("  learned          :", model.parameters)
    train.logger.comment("Ground truth\t{} coin={}".format(URN_RATIOS, COIN_BIAS))
    train.logger.comment("Learned parameters\t{}".format(model.parameters))
    train.logger.comment("F1\t{}".format(macro_f1(selected)))
    train.logger.comment("Accuracy\t{}".format(accuracy))
    train.logger.comment("Program\t{}".format(args.program))
    train.logger.write_to_file(str(LOG_DIR / "t6_coin_ball"))
