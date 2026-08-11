# data/

Everything needed to run is either committed here or downloaded on first
use. Nothing has to be fetched by hand.

| What | Where it comes from | Used by |
|---|---|---|
| `Forth/Add/train{2,4,8}_test{8,64}_{train,dev,test}.txt` | committed, copied from the official repo | T3 |
| `Forth/Sort/train{2,3,4,5,6}_test{8,64}_{train,dev,test}.txt` | committed, copied from the official repo | T4 |
| `Forth/WAP/vocab_746.txt` | committed, copied from the official repo | T5 |
| `MNIST/` | downloaded by torchvision on first run | T6 |
| MNIST inside the installed package | downloaded by `deepproblog.examples.MNIST.data` on first run | T1, T2 |
| T6 coins and balls | generated in process, every run | T6 |

## Why some files are committed

The `deepproblog` pip package only packages `.py` and `.pl` files, so the
plain-text data under `examples/Forth/*/data/` is missing from an install.
Those files are copied in here. To refresh them from upstream:

```bash
git clone --depth 1 https://github.com/ML-KULeuven/deepproblog.git /tmp/dpl
S=/tmp/dpl/src/deepproblog/examples/Forth

for L in 2 4 8; do for M in 8 64; do for P in train dev test; do
  cp "$S/Add/data/train${L}_test${M}_${P}.txt" Forth/Add/
done; done; done

for L in 2 3 4 5 6; do for M in 8 64; do for P in train dev test; do
  cp "$S/Sort/data/train${L}_test${M}_${P}.txt" Forth/Sort/
done; done; done

cp "$S/WAP/data/vocab_746.txt" Forth/WAP/
```

`train<L>_test<M>` is one cell of the paper's Table 1a: trained on lists
or numbers of length `L`, tested on length `M`. Upstream ships more cells
than the paper reports (sorting up to training length 8, an Add
`train24_test128`); only the reported ones are here.

T5's query splits are `.pl` files, so they *are* in the pip package and
are read from there. Only its vocabulary had to be copied.

## MNIST

T1 and T2 read MNIST through `deepproblog.examples.MNIST.data`, which
downloads into the installed package's own directory. T6 downloads its own
copy into `data/MNIST/` via torchvision. Both are automatic, both need
network access on the first run, and both are gitignored.

## T6 generates its data

The coin-ball dataset is not published anywhere. The official repo's
nearest example is a different task whose images come from a Blender
scene, and the paper describes its own data in two sentences instead of
releasing it. So `src/experiments/t6/coin_ball_data.py` builds the dataset
in process on every run, from that description: an MNIST digit for the
coin (even = heads), and RGB triples for the balls obtained by adding
Gaussian noise (sigma = 0.03) to the base colours in the HSV domain.

Dependencies are `torch` and `torchvision`, both already in
`requirements.txt`. No Blender, no downloads beyond MNIST.

The urn ratios and the coin bias it draws from are constants at the top of
that file — the paper never states them, and recovering them is what the
experiment is supposed to demonstrate.
