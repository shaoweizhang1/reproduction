"""T1: single-digit MNIST addition.

Adapted from examples/MNIST/addition.py (https://github.com/ML-KULeuven/deepproblog); the loop lives in
src/utils/mnist.py. Deviations: paths resolved against the installed
package, a fixed iteration budget instead of one epoch, and accuracy
measured periodically rather than only at the end.

Run: python3 -m src.experiments.t1.deepproblog [flags]
"""

from ...utils.args import iteration_parser
from ...utils.mnist import run_deepproblog

if __name__ == "__main__":
    args = iteration_parser().parse_args()
    run_deepproblog(1, "addition.pl", args, "t1_deepproblog")
