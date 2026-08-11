"""T3: Forth "Add" sketch.

Adapted from examples/Forth/Add/add.py (https://github.com/ML-KULeuven/deepproblog); the loop lives in
src/utils/forth.py. Deviations: choose.pl and the query splits are read
from the installed package and this repo's data/, and
--train_length/--test_length select one cell of the paper's Table 1a
instead of the single train2_test8 split the script hardcodes.

Run: python3 -m src.experiments.t3.deepproblog [flags]
"""

import torch

from ...utils.args import epoch_parser
from ...utils.forth import load_queries, run_forth
from ...utils.paths import example_dir

from deepproblog.engines import ExactEngine
from deepproblog.examples.Forth import EncodeModule
from deepproblog.model import Model
from deepproblog.network import Network

if __name__ == "__main__":
    args = epoch_parser(
        lengths=True, batch_size=50, log_period=20, test_period=100, lr=0.02
    ).parse_args()
    train_queries, dev_queries, test_queries = load_queries(
        "Add", args.train_length, args.test_length
    )

    result_net = EncodeModule(30, 50, 10, "tanh")
    carry_net = EncodeModule(22, 10, 2, "tanh")
    nets = [
        Network(result_net, "neural1", torch.optim.Adam(result_net.parameters(), lr=args.lr)),
        Network(carry_net, "neural2", torch.optim.Adam(carry_net.parameters(), lr=args.lr)),
    ]

    program = str(example_dir("Forth", "Add", "choose.pl"))
    model = Model(program, nets)
    model.set_engine(ExactEngine(model), cache=True)

    test_model = Model(
        program,
        [Network(result_net, "neural1", k=1), Network(carry_net, "neural2", k=1)],
    )
    test_model.set_engine(ExactEngine(test_model), cache=False)

    run_forth(
        model,
        test_model,
        train_queries,
        dev_queries,
        test_queries,
        args,
        f"t3_train{args.train_length}_test{args.test_length}",
    )
