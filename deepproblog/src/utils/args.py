"""Shared CLI parsers, so a run is changed by editing commands/*.sh."""

import argparse

def iteration_parser(**defaults):
    """Fixed-iteration-budget training with periodic eval (T1, T2)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--total_iterations", type=int, default=30000)
    parser.add_argument("--log_period", type=int, default=100)
    parser.add_argument("--test_period", type=int, default=1000)
    parser.add_argument("--test_subset", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.set_defaults(**defaults)
    return parser

def epoch_parser(lengths=False, **defaults):
    """Epoch-based training on the Forth query datasets (T3, T4, T5)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--log_period", type=int, default=20)
    parser.add_argument("--test_period", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--stop_at_accuracy", type=float, default=1.0)
    parser.add_argument("--track_test", action="store_true")
    if lengths:
        parser.add_argument("--train_length", type=int, default=2)
        parser.add_argument("--test_length", type=int, default=8)
    parser.set_defaults(**defaults)
    return parser
