% The coin-ball problem (T6), reconstructed from Listing 6 of the paper.
%
% The released library has no implementation of this task -- its Coins
% example is a different, two-coin game with no learnable parameters -- so
% this is transcribed from the paper's appendix. Three repairs were needed
% because the printed listing does not run as given:
%
%   1. Listing 6 defines coin/2 both by a neural AD and by two rules whose
%      bodies call coin/2, which is left recursive. The neural predicate is
%      renamed net_coin/2 here and coin/2 derives from it, which is plainly
%      the intent: the network proposes a side, is_heads weights it.
%   2. The listing declares colour/4 over (R,G,B,Colour) but calls it as
%      colour(Colour, C) inside urn/3. Here the network takes one tensor
%      argument, so colour/2 is used consistently.
%   3. Ball colours are drawn per urn by col/2, whose probabilities are the
%      learnable parameters; the listing's 0.5 / 0.333 are initial values,
%      not the distribution the data is drawn from.

nn(m_colour, [X], C, [red, green, blue]) :: colour(X, C).
nn(m_coin, [X], S, [heads, tails]) :: net_coin(X, S).

t(0.5)::col(1,red); t(0.5)::col(1,blue).
t(0.333)::col(2,red); t(0.333)::col(2,green); t(0.333)::col(2,blue).
t(0.5)::is_heads.

coin(X, heads) :- net_coin(X, heads), is_heads.
coin(X, tails) :- net_coin(X, tails), \+is_heads.

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
