## Metrics Hint: k3 Stress Test

In this regime, you are given exactly two benchmark instance files:

- `circle_k3_m6.pickle`
- `circle_k3_m8.pickle`

Treat these as:

- circle metrics
- `k = 3`
- `m = 6` and `m = 8`
- `circle_k3_m6.pickle`: `350` nodes and `2100` inequalities
- `circle_k3_m8.pickle`: `5240` nodes and `41920` inequalities

This is a multi-instance stress-test setting.

Practical implication for the agent:

- the solution should achieve `violations_k = 0` on both instances
- the combined score is the product of the `violations_k_score` values on the two instances
- improving only one instance is not enough if the other instance still has many violations
- you should optimize for robustness across both `m = 6` and `m = 8`, not just for the smaller instance
