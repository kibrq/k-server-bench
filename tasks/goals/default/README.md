# k-Server Problem

You are proposing a *potential* for the k-server problem. Recall that a state in the k-server problem can be described by its current work function: a function mapping every configuration (a multiset of k points) to a real number representing the minimum cost required to serve all previous requests and end in that configuration.
Your goal is to propose a potential: a function that maps each work function to a real value.

## Scoring

Your potential will be evaluated based on the number of violations it incurs. Given a work function ( w ), you can compute the updated work function after serving request ( r ), denoted ( w' ). The potential must satisfy:

[
\Phi(w') - \Phi(w) >= \max_X \big( w'(X) - w(X) \big)
]

or normalized version:

[
\Phi(w'_norm) - \Phi(w_norm) + (k + 1) (\min_X w'(X) - \min_X w(X)) >= \max_X \big( w'(X) - w(X) \big)
]

Your score, which you want to minimize, is called `violations_k`. Ideally, this number should be 0 (equivalently, `violations_k_score` should be 1).

Metrics also include `processed_normalized_edges_score` (from 0 to 1) which is the score of how many violations were actually processed. This depends on the computation speed of your proposed potential. Usually this will be 1. If `processed_normalized_edges_score` is low, the `violations_k_score` might be less accurate.
