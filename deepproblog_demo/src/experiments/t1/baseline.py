"""T1: single-digit MNIST addition, plain-NN baseline.

Adapted from examples/MNIST/neural_baseline/baseline.py (https://github.com/ML-KULeuven/deepproblog); the
architectures and the loop live in src/utils/mnist.py. Deviation: the full
training set on a fixed iteration budget instead of an N=500 subset with
early stopping, matching t1/deepproblog.py.

--model picks between the paper's two baselines. "concat" is the CNN of
Figure 3a, not released and rebuilt from the implementation details;
"separate" is the shared-weight encoder the repository ships, and the
fairer comparison -- it factorises per image the way DeepProbLog does, and
matches T2's baseline.

Run: python3 -m src.experiments.t1.baseline [flags]
"""

from ...utils.args import iteration_parser
from ...utils.mnist import BASELINES, Separate_Baseline, run_baseline

if __name__ == "__main__":
    parser = iteration_parser(lr=0.001, weight_decay=1e-2)
    parser.add_argument("--model", default="concat", choices=sorted(BASELINES))
    args = parser.parse_args()

    net = BASELINES[args.model]
    net = net(batched=True, probabilities=False) if net is Separate_Baseline else net()

    run_baseline(
        net,
        1,
        args,
        f"t1_baseline_{args.model}",
        train_unwrap=lambda imgs: imgs[0],
        test_unwrap=lambda imgs: imgs[0].unsqueeze(0),
    )
