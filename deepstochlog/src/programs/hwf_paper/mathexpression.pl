dom_number(X) :- member(X, [0,1,2,3,4,5,6,7,8,9]).
nn(number, [X], Y, dom_number) :: is_number(Y) --> [X].

dom_operator(X) :- member(X, [plus, minus, times, div]).
nn(operator, [X], Y, dom_operator) :: operator(Y) --> [X].
factor(N) --> is_number(N).

dom_switch(Y) :- member(Y, [0,1,2]).

nn(term_switch, [], Y, dom_switch) :: term(N) --> term_switch(N, Y).
0.33 :: term_switch(N, 0) --> factor(N).
0.33 :: term_switch(N, 1) --> term(N1), operator(times), factor(N2), {N is N1 * N2}.
0.33 :: term_switch(N, 2) --> term(N1), operator(div), factor(N2), {N2>0, N is N1 / N2}.

nn(expression_switch, [], Y, dom_switch) :: expression(N) --> expression_switch(N, Y).
0.33 :: expression_switch(N, 0) --> term(N).
0.33 :: expression_switch(N, 1) --> expression(N1), operator(plus), term(N2), {N is N1 + N2}.
0.33 :: expression_switch(N, 2) --> expression(N1), operator(minus), term(N2), {N is N1 - N2}.
