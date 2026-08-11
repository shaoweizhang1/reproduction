% Coin-ball (T6), variant V3: the coin bias written as an annotated
% disjunction, matching how Listing 6 writes the urns.
%
% Listing 6 declares the urn compositions as ADs -- "t(0.5)::col(1,red);
% t(0.5)::col(1,blue)." -- but the coin bias as a lone fact, "t(0.5)::
% is_heads.", used positively in one clause and negated in the other.
% Those are equivalent on paper. They are not equivalent in deepproblog:
% SGD.step() renormalises each parameter group after every update, so AD
% members cannot collapse, while a lone fact can walk to 0 and stay there
% -- which is exactly what is_heads does in coin_ball.pl, taking the whole
% heads branch with it. Writing it as a two-way AD restores the symmetry.

nn(m_colour, [X], C, [red, green, blue]) :: colour(X, C).
nn(m_coin, [X], S, [heads, tails]) :: net_coin(X, S).

t(0.5)::col(1,red); t(0.5)::col(1,blue).
t(0.333)::col(2,red); t(0.333)::col(2,green); t(0.333)::col(2,blue).
t(0.5)::side(heads); t(0.5)::side(tails).

coin(X, S) :- side(S), net_coin(X, S).

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
