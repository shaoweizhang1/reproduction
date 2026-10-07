% Pearl's classic burglary-alarm Bayesian network -- version with evidence
% priors
0.001::burglary.
0.002::earthquake.

% whether the alarm fires depends on the four burglary/earthquake combinations,
% each with its own probability (a conditional probability table)
0.95::alarm :- burglary, earthquake.
0.94::alarm :- burglary, \+earthquake.
0.29::alarm :- \+burglary, earthquake.
0.001::alarm :- \+burglary, \+earthquake.

% whether John/Mary calls depends on whether the alarm fires
0.90::johncalls :- alarm.
0.05::johncalls :- \+alarm.
0.70::marycalls :- alarm.
0.01::marycalls :- \+alarm.

% evidence: both John and Mary called; infer the posterior probability of alarm/burglary/earthquake
evidence(johncalls, true).
evidence(marycalls, true).

query(burglary).
query(earthquake).
query(alarm).
