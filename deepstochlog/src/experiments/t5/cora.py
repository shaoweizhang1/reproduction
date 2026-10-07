"""T5 on Cora with the paper's base program (Appendix B.5).

Adapted from examples/citeseer/base/citeseer.py
(github.com/ML-KULeuven/deepstochlog). Upstream ships no base entry point for
Cora -- examples/cora/ holds the rule-weight variant only -- so this is that
runner with the Cora data module, the Cora class terms and no rule-weight
network. It loads src/programs/depth2/cora.pl directly, which is where the depth-2
program lives for both datasets.
"""

from time import time

import torch
from torch.optim import Adam

from deepstochlog.dataloader import DataLoader
from deepstochlog.model import DeepStochLogModel
from deepstochlog.network import Network, NetworkStore
from deepstochlog.term import Term
from deepstochlog.trainer import DeepStochLogTrainer, print_logger
from deepstochlog.utils import set_fixed_seed
from examples.citeseer.citeseer_utils import AccuracyCalculator
from examples.cora.with_rule_weights.cora_data_withrules import (
    citations,
    queries_for_model,
    test_dataset,
    train_dataset,
    valid_dataset,
)
from examples.models import Classifier

from ...utils.paths import PROGRAMS_DIR

root_path = PROGRAMS_DIR / "depth2"


def run(
    epochs=100,
    batch_size=32,
    lr=0.01,
    log_freq=50,
    logger=print_logger,
    seed=None,
    verbose=True,
    device_str=None,
):
    set_fixed_seed(seed)

    classifier = Classifier(input_size=len(train_dataset.documents[0]))
    networks = NetworkStore(
        Network(
            "classifier",
            classifier,
            index_list=[Term("class" + str(i)) for i in range(7)],
        )
    )

    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    proving_start = time()
    model = DeepStochLogModel.from_file(
        file_location=str((root_path / "cora.pl").absolute()),
        query=queries_for_model,
        networks=networks,
        device=device,
        prolog_facts=citations,
        verbose=verbose,
    )
    optimizer = Adam(model.get_all_net_parameters(), lr=lr)
    optimizer.zero_grad()

    if verbose:
        logger.print("\nProving the program took {:.2f} seconds".format(time() - proving_start))

    accuracy = AccuracyCalculator(
        model=model, valid=valid_dataset, test=test_dataset, start_time=time()
    )
    trainer = DeepStochLogTrainer(
        log_freq=log_freq,
        accuracy_tester=(accuracy.header, accuracy),
        logger=logger,
        print_time=verbose,
    )
    trainer.train(
        model=model,
        optimizer=optimizer,
        dataloader=DataLoader(train_dataset, batch_size=batch_size),
        epochs=epochs,
    )

    return None
