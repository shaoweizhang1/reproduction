"""Tees stdout of an experiment into log/<name>.log; never overwrites a log."""

import contextlib
import sys

from .paths import LOG_DIR


class _Tee:
    def __init__(self, handle):
        self._handle = handle

    def write(self, text):
        sys.__stdout__.write(text)
        self._handle.write(text)
        self._handle.flush()
        return len(text)

    def flush(self):
        sys.__stdout__.flush()
        self._handle.flush()


@contextlib.contextmanager
def run_log(name):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / (name + ".log")).open("x") as handle:
        with contextlib.redirect_stdout(_Tee(handle)):
            yield
