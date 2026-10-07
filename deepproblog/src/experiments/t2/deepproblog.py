"""T2: multi-digit MNIST addition.

Same as t1/deepproblog.py with n_digits=2. addition.pl defines both
addition/3 and multi_addition/3, and the library picks between them by
digit count. models/all_digit_addition.pl is a different task despite the
name -- there the result is an image too.

Run: python3 -m src.experiments.t2.deepproblog [flags]
"""

from ...utils.args import iteration_parser
from ...utils.mnist import run_deepproblog

if __name__ == "__main__":
    args = iteration_parser().parse_args()
    run_deepproblog(2, "addition.pl", args, "t2_deepproblog")
