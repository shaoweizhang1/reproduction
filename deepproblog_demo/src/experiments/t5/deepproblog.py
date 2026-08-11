"""T5: Forth WAP, word algebra problems.

Adapted from examples/Forth/WAP/wap.py (https://github.com/ML-KULeuven/deepproblog); the loop lives in
src/utils/forth.py. Only the vocabulary had to be copied into data/ (see
network.py). Deviation: the periodic accuracy is measured on dev, keeping
the test set for the final number.

Run: python3 -m src.experiments.t5.deepproblog [flags]
"""

from .network import get_networks
from ...utils.args import epoch_parser
from ...utils.forth import run_forth
from ...utils.paths import example_dir

from deepproblog.dataset import QueryDataset
from deepproblog.engines import ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network

if __name__ == "__main__":
    parser = epoch_parser(batch_size=10, log_period=10, test_period=30, lr=0.005)
    parser.add_argument("--p_drop", type=float, default=0.5)
    args = parser.parse_args()

    data_dir = example_dir("Forth", "WAP", "data")
    train_queries, dev_queries, test_queries = (
        QueryDataset(str(data_dir / f"{name}.pl")) for name in ("train", "dev", "test")
    )

    networks = get_networks(args.lr, args.p_drop)
    program = str(example_dir("Forth", "WAP", "wap.pl"))

    model = Model(program, [Network(net, name, optimizer) for net, name, optimizer in networks])
    model.set_engine(ExactEngine(model), cache=True)

    # The RNN is shared verbatim; only the four classifier heads get k=1.
    test_model = Model(
        program,
        [Network(networks[0][0], networks[0][1])]
        + [Network(net, name, k=1) for net, name, _ in networks[1:]],
    )
    test_model.set_engine(ExactEngine(test_model), cache=False)

    run_forth(model, test_model, train_queries, dev_queries, test_queries, args, "t5_deepproblog")
