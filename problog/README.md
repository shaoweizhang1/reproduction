# ProbLog programs

Six standalone ProbLog examples are provided for probabilistic facts,
conditioning, annotated disjunctions, recursion and graph queries.
They are maintained separately from the DeepProbLog reproduction.

## Running

ProbLog is a pip package with no dependencies beyond Python:

```bash
cd problog
python -m pip install -r requirements.txt
problog 6_hmm_weather.pl
```

Every program has its queries written into the file, so `problog <file>` is
the whole interface. Output is one line per query:

```
$ problog montyhall.pl
  win_stay:	0.33333333
win_switch:	0.66666667
```

Browser execution is also supported through the
[ProbLog editor](https://dtai.cs.kuleuven.be/problog/editor.html).

## The programs

| File | What it shows | Prints |
|---|---|---|
| `alarm.pl` | probabilistic facts, rules with negation, a CPT written as four clauses | `alarm` 0.0025, `johncalls` 0.052 |
| `alarm_evidence.pl` | the same network conditioned on `evidence/2` | `burglary` 0.284, up from a prior of 0.001 |
| `montyhall.pl` | annotated disjunctions for host selection | `win_stay` 1/3, `win_switch` 2/3 |
| `6_hmm_weather.pl` | recursion over time: an HMM as three clauses | `weather(sun,10)` 0.333 |
| `7_probabilistic_graph.pl` | the reachability problem ProbLog was built for | `path(1,5)` 0.258, `path(1,6)` 0.217 |
| `8_smokers_network.pl` | probabilistic rules — `0.3::stress(X) :- person(X)` is one coin per person | `smokes(3)` 0.44, `asthma(3)` 0.176 |

`7_probabilistic_graph.pl` prints a warning about `\==`; it is upstream, and
the answer is unaffected.

## Additional operations

**Conditioning.** The same network is defined in `alarm.pl` and
`alarm_evidence.pl`. With two calls observed, the burglary probability
is updated from 0.001 to 0.284.

**Host probabilities.** When the car is behind door 1, the host's choice
is represented by `0.5::opens(2); 0.5::opens(3)`. If these probabilities
are changed to 0.9 and 0.1, a switching probability of 2/3 is retained.
If `opens(3) :- car(2).` is removed, `win_switch` is reduced to 1/3.

**Program transformation.** The transformed program is displayed by
`problog explain montyhall.pl`. The annotated disjunction is expanded
into three `choice/3` facts and three rules.

**Sampling.** Ten possible worlds and their query outcomes are sampled by
`problog sample montyhall.pl -N 10`.

**Most probable explanation.** The highest-probability assignment consistent
with the evidence is returned by `problog mpe alarm_evidence.pl`: no burglary,
no earthquake and an active alarm. A `resulttable` file is also produced
by the MaxSAT backend.

## Where they came from

`6_hmm_weather.pl`, `7_probabilistic_graph.pl` and `8_smokers_network.pl`
are copied unchanged from the ProbLog site's own example programs, keeping
their original numbering, comments and expected outcomes — the numbers in
those headers are what the current version still produces. `alarm.pl`,
`alarm_evidence.pl` and `montyhall.pl` are written here, from the standard
formulations of Pearl's alarm network and the Monty Hall problem.
