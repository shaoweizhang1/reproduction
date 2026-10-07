# DeepStochLog Reproduction

This project reproduces the experiments of Winters et al., *DeepStochLog: Neural
Stochastic Logic Programming*, AAAI 2022
([paper](https://cdn.aaai.org/ojs/21248/21248-13-25261-1-2-20220628.pdf)). It
uses the official [ML-KULeuven/deepstochlog](https://github.com/ML-KULeuven/deepstochlog)
code for training and proving, with a small patch.
Blog post: [DeepStochLog, Neural Stochastic Logic Programming](https://shaoweizhang1.github.io/reading-notes/deepstochlog-neural-stochastic-logic-programming).

## Results

Accuracy (%) is the mean ± the sample standard deviation over five runs. The
tasks are described under [Layout](#layout).

| Task | Setting | Reproduced | Paper |
|---|---|---:|---:|
| T1 addition | 1 digit | 98.0 ± 0.1 | 97.9 ± 0.1 |
| | 2 digits | 96.1 ± 0.3 | 96.4 ± 0.1 |
| | 3 digits | 94.7 ± 0.4 | 94.5 ± 1.1 |
| | 4 digits | 92.8 ± 0.5 | 92.7 ± 0.6 |
| T2 formulas | length 1 | 89.1 ± 1.0 | 90.8 ± 1.0 |
| | length 3 | 85.5 ± 1.7 | 86.3 ± 1.9 |
| | length 5 | 90.7 ± 1.1 | 92.1 ± 1.4 |
| | length 7 | 94.5 ± 0.6 | 94.8 ± 0.9 |
| T3 parentheses | max 10 | 100.0 ± 0.0 | 100.0 ± 0.0 |
| | max 14 | 99.6 ± 0.5 | 100.0 ± 0.0 |
| | max 18 | 99.6 ± 0.5 | 100.0 ± 0.0 |
| T4 a^n b^n c^n | 3–12 | 99.3 ± 0.3 | 99.4 ± 0.5 |
| | 3–15 | 99.2 ± 0.6 | 99.2 ± 0.4 |
| | 3–18 | 99.1 ± 0.2 | 98.8 ± 0.2 |
| T5 citation | Citeseer | 64.5 ± 1.9 | 65.0 |
| | Cora | 72.4 ± 0.4 | 69.4 |
| T6 word problems | | 94.4 ± 0.5 | 94.8 ± 1.1 |

T2 and T5 report the test accuracy at the best validation score, and the other
tasks use the upstream selection rule. T2 ran on CPU; earlier GPU runs of the
same code gave 90.4 ± 1.5, 85.9 ± 2.5, 91.8 ± 0.6 and 95.8 ± 0.6. Cora is about
3 points above the paper; the cause was not investigated.

### The paper's Table 6: parsing with and without tabling

Tabling saves the answers of a subgoal the first time it is proved and reuses
them. The experiment lists every way to read a T2 formula of a given length,
once without tabling (SLD) and once with it (SLG). Times are in seconds, as the
mean of three runs:

| Length | Answers | SLD | SLD (paper) | SLG | SLG (paper) |
|---|---:|---:|---:|---:|---:|
| 1 | 10 | 0.191 | 0.067 | 0.179 | 0.060 |
| 3 | 95 | 0.223 | 0.081 | 0.212 | 0.096 |
| 5 | 1,066 | 2.110 | 3.78 | 0.646 | 0.95 |
| 7 | 10,386 | 79.853 | 30.42 | 5.624 | 10.95 |
| 9 | 68,298 | 3,306.359 | 1,494.23 | 61.487 | 132.26 |
| 11 | 416,517 | timeout | timeout | 573.647 | 1,996.09 |

Every finished run gives the expected answers, and SLD at length 11 timed out
(one hour) all three times. Tabling helps as much as the paper says, or more,
but the absolute times differ: these runs were made on an AMD EPYC 7452
server, and the paper's on a 2020 MacBook Pro.

### The paper's Table 7: time per query

The time to compute the probability of one T1 query after its proofs are built,
in milliseconds, over five runs of 100 queries on one CPU thread:

| Digits per number | Reproduced | Paper |
|---|---:|---:|
| 1 | 1.209 ± 0.184 | 1.3 ± 0.9 |
| 2 | 2.130 ± 0.282 | 2.3 ± 0.4 |
| 3 | 3.582 ± 0.453 | 4.0 ± 0.4 |
| 4 | 9.259 ± 2.663 | 5.7 ± 1.8 |

Lengths 1–3 are close to the paper. Length 4 is slower and less stable, which
looks like a few very slow queries in four of the five runs.

## Observations

**The released tabling switch does nothing.** In release 0.0.1, which Table 6
is based on, `tabling=False` leaves the `:- table solve/1.` declaration in the
program, so both modes run with tabling and the SLD column cannot be measured
with the released code. Here the SLD mode is rebuilt by removing that
declaration and the branch that only fills tables, and SWI-Prolog checks before each run
that `solve/1` is (or is not) tabled.

**The paper's T2 model is not the released one.** The paper's appendix uses
separate networks for digits and operators and lets neural networks choose
between the grammar rules. The released code shares one encoder between digits
and operators, passes its parameters to Adam twice (so they get two updates per
step), and fixes the rule probabilities at 0.34/0.33/0.33. The paper's version
was also run (`bash commands/t2_hwf.sh paper`), on CPU with seeds 0–4:

| Length | Released model | Paper's model |
|---|---:|---:|
| 1 | 89.1 ± 1.0 | 91.2 ± 1.2 |
| 3 | 85.5 ± 1.7 | 83.6 ± 3.2 |
| 5 | 90.7 ± 1.1 | 91.9 ± 0.8 (four seeds); seed 2: 1.25 |
| 7 | 94.5 ± 0.6 | 1.3 ± 0.1 |

At length 7 the paper's model did not learn at all: the loss stayed near 8 in
all five runs, while the released model (seed 0) reached 90% within three
epochs. A second run of the failed seed at length 5 reached 41.8%. Which of the two
changes causes this was not isolated.

**T2 is faster on CPU than on GPU.** Almost all of T2's time goes to the proof
trees, not the networks. The engine computes every node as a separate operation
on a single number, and the trees are large. Training one seed took:

| Length | GPU (s) | CPU (s) | Speedup |
|---|---:|---:|---:|
| 1 | 40 | 40 | 1.0× |
| 3 | 154 | 121 | 1.3× |
| 5 | 1,441 | 739 | 2.0× |
| 7 | 40,395 | 16,527 | 2.4× |

With the five seeds running side by side on CPU, length 7 went from about 56 h
to under 5 h.

**Reruns are not exact.** Running T2 again with the same seed gives a different
result: at length 1, seed 0 gave between 87.0% and 90.0% over six runs, with or
without a fixed `PYTHONHASHSEED`.

## Speedups and fixes

Two evaluation steps were sped up without touching training or proving. For T1
this was necessary, because the original evaluator could not finish lengths 3
and 4:

- **T1**: the original evaluator walks through the proof trees for every
  possible sum, which is 19,999 sums for four-digit numbers. The new evaluator
  combines the digit probabilities and carries directly and gets exactly the same
  distribution.
- **T5**: each test document is scored on its own, so the documents are spread
  over 16 CPU processes.

Time for one epoch:

| Task | Setting | Original (s) | Faster (s) | Speedup |
|---|---|---:|---:|---:|
| T1 | 1 digit | 269 | 54 | 5.0× |
| | 2 digits | 7,081 | 77 | 92× |
| | 3 digits | > 21,600 | 94 | > 220× |
| | 4 digits | > 21,600 | 104 | > 200× |
| T5 | Citeseer | 23.93 | 4.91 | 4.9× |
| | Cora | 59.84 | 8.90 | 6.7× |

The accuracy does not change: at T1 length 1, five full runs with the original
evaluator reach 98.14 ± 0.27% against 97.98 ± 0.15% with the new one, and T5
gives the same validation and test accuracy either way. The original and faster
runs did not share a machine or a time slot, so the speedups are rough.

**Fixes.**

- The patch makes the code run on current Python and pandas, puts the T2
  evaluation inputs on the model's device, and fixes the predicate name and
  query count (99 instead of 100) in the T1 timing script.
- Table 6 also fixes three problems in release 0.0.1: a missing two-argument
  `nn` fact, a grammar included many times, and answer printing that kept
  backtracking state alive.
- T5 skips DGL's unused GraphBolt import, which does not work with the pinned
  PyTorch version. Its classifiers have 6 outputs for Citeseer and 7 for Cora, as in the
  paper (upstream uses 10), and the paper does not give the depth of the
  citation program, so 2 is used.
- The appendix gives T1's two `multi_addition` rules probability 0.5 and the
  code gives them 1. This scales every derivation by the same constant and
  changes nothing.

## Layout

```
requirements.txt     project dependencies
commands/            setup, experiment and check scripts
src/experiments/     t1 … t6
src/programs/        depth-2 citation programs, paper's T2 grammar
src/utils/           shared helpers
patches/             fixes to the official code
upstream/            official code, created by setup (not tracked)
log/                 run logs (not tracked)
```

| Task | What it is | Command |
|---|---|---|
| T1 | Add two numbers written as 1–4 handwritten digits each; only the sum is labelled | `t1_addition_sumdp.sh` |
| T2 | Compute a handwritten formula like `7 * 2 - 3`; only the value is labelled | `t2_hwf.sh` |
| T3 | Parse well-formed parentheses written as MNIST 0s and 1s | `t3_bracket.sh` |
| T4 | Decide if MNIST images form a^n b^n c^n (in any block order) | `t4_anbncn.sh` |
| T5 | Classify papers by topic from their words and citations (Citeseer, Cora) | `t5_citation.sh` |
| T6 | Answer short word algebra problems with three numbers | `t6_wap.sh` |
| Table 6 | Time to list every parse of a T2 formula, with and without tabling | `t2_parsing_time.sh` |
| Table 7 | Time to compute the probability of one T1 query | `t1_inference_time.sh` |

`commands/verify.sh` runs a quick CPU check of every task, without full training.

## Running

Python 3.12, SWI-Prolog (tested with 10.0.0), Git, curl and unzip are needed.

```bash
cd deepstochlog
pip install -r requirements.txt
bash commands/setup_upstream.sh       # official code into upstream/, patched

mkdir -p upstream/data/raw            # HWF (T2) and WAP (T6)
for name in hwf wap; do
  curl -L -o $name.zip https://github.com/ML-KULeuven/deepstochlog/releases/download/0.0.1/$name.zip
  unzip -q $name.zip -d upstream/data/raw
  rm $name.zip
done

CUDA_VISIBLE_DEVICES=0 bash commands/t1_addition_sumdp.sh
```

MNIST and the citation graphs download on first use. The official
`download_hwf.sh` no longer works because its Google Drive file is gone. T1 and
T6 run on a GPU and the rest on CPU, and `bash commands/t2_hwf.sh paper` runs
the paper's T2 model. Each script prints the mean ± sample standard deviation
over five runs and writes a log to `log/`.

## Attribution

All training and inference use the official code, at commit
`aed95319411b8b5190b5b418b65b523b1436bcfd` (Table 6 at release 0.0.1,
`2a6982f4e48bf8559df441e0c33859d149eaab0b`). Python modules adapted from an
official example name the upstream file in their header. The T1 evaluator has no
upstream counterpart, and the paper's T2 grammar is copied from the paper's
appendix.
