# problog_demo/

Six standalone ProbLog programs. Nothing here is part of the DeepProbLog
reproduction — this is where I learned the language before starting on it,
and it is kept because the programs are the shortest way to see what
ProbLog does.

## Running them

ProbLog is a pip package with no dependencies beyond Python:

```bash
pip install problog
problog 6_hmm_weather.pl
```

Every program has its queries written into the file, so `problog <file>` is
the whole interface. Output is one line per query:

```
$ problog montyhall.pl
  win_stay:	0.33333333
win_switch:	0.66666667
```

You can also paste any of them into the editor at
[dtai.cs.kuleuven.be/problog/editor.html](https://dtai.cs.kuleuven.be/problog/editor.html)
and run them in the browser.

## The programs

| File | What it shows | Prints |
|---|---|---|
| `alarm.pl` | probabilistic facts, rules with negation, a CPT written as four clauses | `alarm` 0.0025, `johncalls` 0.052 |
| `alarm_evidence.pl` | the same network conditioned on `evidence/2` | `burglary` 0.284, up from a prior of 0.001 |
| `montyhall.pl` | annotated disjunctions, and a puzzle people get wrong | `win_stay` 1/3, `win_switch` 2/3 |
| `6_hmm_weather.pl` | recursion over time: an HMM as three clauses | `weather(sun,10)` 0.333 |
| `7_probabilistic_graph.pl` | the reachability problem ProbLog was built for | `path(1,5)` 0.258, `path(1,6)` 0.217 |
| `8_smokers_network.pl` | probabilistic rules — `0.3::stress(X) :- person(X)` is one coin per person | `smokes(3)` 0.44, `asthma(3)` 0.176 |

`7_probabilistic_graph.pl` prints a warning about `\==`; it is upstream, and
the answer is unaffected.

## Things worth trying

**Watch evidence move a posterior.** `alarm.pl` and `alarm_evidence.pl` are
the same network; the second adds two lines. Burglary goes from 0.001 to
0.284 because two people called.

**Break the Monty Hall symmetry.** The host's choice when the car is behind
door 1 is `0.5::opens(2); 0.5::opens(3)`. Make it `0.9::opens(2);
0.1::opens(3)` and both answers stay put — switching still wins two thirds
of the time. Delete the `opens(3) :- car(2).` clause instead and `win_switch`
falls to 1/3 — that clause is the whole puzzle.

**See the sugar come off.** `problog explain montyhall.pl` prints the
program after transformation. The annotated disjunction becomes three
`choice/3` facts and three rules — the desugaring, spelled out.

**Sample instead of solving.** `problog sample montyhall.pl -N 10` draws ten
possible worlds and prints which queries hold in each, rather than
computing an exact probability.

**Ask for the most likely world.** `problog mpe alarm_evidence.pl` returns
the single assignment with the highest probability that is consistent with
the evidence — here, no burglary, no earthquake, alarm fires anyway. Its
MaxSAT solver drops a `resulttable` file in the working directory; delete
it.

## Where they came from

`6_hmm_weather.pl`, `7_probabilistic_graph.pl` and `8_smokers_network.pl`
are copied unchanged from the ProbLog site's own example programs, keeping
their original numbering, comments and expected outcomes — the numbers in
those headers are what the current version still produces. `alarm.pl`,
`alarm_evidence.pl` and `montyhall.pl` are written here, from the standard
formulations of Pearl's alarm network and the Monty Hall problem.
