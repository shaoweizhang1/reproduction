% The Monty Hall problem, encoded in ProbLog.
% Three doors hide a car behind exactly one of them. The player picks door 1,
% the host opens a different, car-free door, and the player decides whether
% to stay with door 1 or switch to the other unopened door.

% the car is placed uniformly at random behind one of the three doors
1/3::car(1); 1/3::car(2); 1/3::car(3).

% the player always picks door 1 first (by symmetry this does not affect the result)
chosen(1).

% host behavior, given the player picked door 1:
% - if the car is behind door 1, both other doors are car-free, so the host picks one at random
% - otherwise exactly one door is both car-free and not chosen, so the host is forced to open it
0.5::opens(2); 0.5::opens(3) :- car(1).
opens(3) :- car(2).
opens(2) :- car(3).

% winning by staying with the original choice (door 1)
win_stay :- car(1).

% winning by switching to whichever door the host did not open
win_switch :- car(2), opens(3).
win_switch :- car(3), opens(2).

query(win_stay).
query(win_switch).
