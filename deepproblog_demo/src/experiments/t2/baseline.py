"""T2: multi-digit MNIST addition, plain-NN baseline.

Adapted from examples/MNIST/neural_baseline/multidigit_baseline.py (https://github.com/ML-KULeuven/deepproblog);
architecture and loop in src/utils/mnist.py. Deviations: the full training
set, a fixed iteration budget instead of early stopping, and n=2*digits
where the original hardcodes n=4.

Run: python3 -m src.experiments.t2.baseline [flags]
"""

from ...utils.args import iteration_parser
from ...utils.mnist import Separate_Baseline_Multi, run_baseline

if __name__ == "__main__":
    args = iteration_parser(lr=0.001).parse_args()
    run_baseline(
        Separate_Baseline_Multi(n=4),
        2,
        args,
        "t2_baseline",
        train_unwrap=lambda imgs: imgs,
        test_unwrap=lambda imgs: [img.unsqueeze(0) for img in imgs],
    )
