"""T4: Forth "Sort" sketch, bubble sort.

Adapted from examples/Forth/Sort/sort.py (https://github.com/ML-KULeuven/deepproblog); the loop lives in
src/utils/forth.py. Same deviations as t3/deepproblog.py.

Run: python3 -m src.experiments.t4.deepproblog [flags]
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
        lengths=True, batch_size=16, log_period=50, test_period=50, lr=1.0
    ).parse_args()
    train_queries, dev_queries, test_queries = load_queries(
        "Sort", args.train_length, args.test_length
    )

    swap_net = EncodeModule(20, 20, 2)

    program = str(example_dir("Forth", "Sort", "compare.pl"))
    model = Model(
        program,
        [Network(swap_net, "swap_net", torch.optim.Adam(swap_net.parameters(), args.lr))],
    )
    model.set_engine(ExactEngine(model), cache=True)

    test_model = Model(program, [Network(swap_net, "swap_net", k=1)])
    test_model.set_engine(ExactEngine(test_model), cache=False)

    run_forth(
        model,
        test_model,
        train_queries,
        dev_queries,
        test_queries,
        args,
        f"t4_train{args.train_length}_test{args.test_length}",
    )
