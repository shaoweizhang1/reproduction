"""Shared training loop for the T3-T5 Forth experiments.

Follows examples/Forth/{Add/add,Sort/sort,WAP/wap}.py (https://github.com/ML-KULeuven/deepproblog). Two
deviations: training stops once dev accuracy reaches --stop_at_accuracy
instead of running a fixed 40 epochs, and the best-on-dev checkpoint is
restored before the test set is scored.
"""

import time
from tempfile import TemporaryDirectory

from .eval_utils import get_confusion_matrix, macro_f1
from .paths import DATA_DIR, LOG_DIR

from deepproblog.dataset import DataLoader, QueryDataset
from deepproblog.train import train_model
from deepproblog.utils.stop_condition import EpochStop, Threshold

def load_queries(task, train_length, test_length):
    directory = DATA_DIR / "Forth" / task
    split = "train{}_test{}".format(train_length, test_length)
    return tuple(
        QueryDataset(str(directory / f"{split}_{name}.txt"))
        for name in ("train", "dev", "test")
    )

def run_forth(model, test_model, train_queries, dev_queries, test_queries, args, name):
    LOG_DIR.mkdir(exist_ok=True)
    stop = EpochStop(args.epochs)
    if args.stop_at_accuracy > 0:
        stop = stop | Threshold("Accuracy", args.stop_at_accuracy, duration=1)

    with TemporaryDirectory() as tmp:
        best_checkpoint = f"{tmp}/best.pth"
        best = {"dev": -1.0, "time_to_perfect": None}
        started = time.time()

        def score(_):
            accuracy = get_confusion_matrix(test_model, dev_queries).accuracy()
            if accuracy > best["dev"]:
                best["dev"] = accuracy
                model.save_state(best_checkpoint)
            logged = [("Accuracy", accuracy)]
            if args.track_test:
                on_test = get_confusion_matrix(test_model, test_queries).accuracy()
                if on_test >= 1.0 and best["time_to_perfect"] is None:
                    best["time_to_perfect"] = time.time() - started
                logged.append(("Test accuracy", on_test))
            return logged

        train_obj = train_model(
            model,
            DataLoader(train_queries, args.batch_size),
            stop,
            log_iter=args.log_period,
            test_iter=args.test_period,
            test=score,
        )

        final_test = get_confusion_matrix(test_model, test_queries).accuracy()
        if best["dev"] >= 0:
            model.load_state(best_checkpoint)
        selected = get_confusion_matrix(test_model, test_queries)

    logger = train_obj.logger
    logger.comment("Best dev accuracy\t{}".format(best["dev"]))
    logger.comment("Final-model test accuracy\t{}".format(final_test))
    if args.track_test:
        logger.comment("Time to 100%\t{}".format(best["time_to_perfect"]))
    logger.comment("F1\t{}".format(macro_f1(selected)))
    logger.comment("Accuracy\t{}".format(selected.accuracy()))
    logger.write_to_file(str(LOG_DIR / name))
