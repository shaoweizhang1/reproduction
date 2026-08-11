% Coin-ball (T6), variant V2: no learnable coin bias at all.
%
% A control, to localise the failure in coin_ball.pl. The urn ratios are
% learned exactly as in the paper's Listing 6; only the coin bias is
% dropped, so the
% side comes straight from the network. If the urn ratios come out right
% here but not with a coin bias in play, the fault is in how the bias is
% tied to the network, not in the rest of the program.

nn(m_colour, [X], C, [red, green, blue]) :: colour(X, C).
nn(m_coin, [X], S, [heads, tails]) :: coin(X, S).

t(0.5)::col(1,red); t(0.5)::col(1,blue).
t(0.333)::col(2,red); t(0.333)::col(2,green); t(0.333)::col(2,blue).

urn(ID, X, C) :- col(ID, C), colour(X, C).

outcome(heads, red, _, win).
outcome(heads, _, red, win).
outcome(_, C, C, win).
outcome(Coin, Colour1, Colour2, loss) :- \+outcome(Coin, Colour1, Colour2, win).

game(Coin, Urn1, Urn2, Result) :-
    coin(Coin, Side),
    urn(1, Urn1, C1),
    urn(2, Urn2, C2),
    outcome(Side, C1, C2, Result).
