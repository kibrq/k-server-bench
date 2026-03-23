## Metrics Hint: k3 Warmup

In this regime, you are given exactly one benchmark instance file:

- `circle_k3_m6.pickle`

Treat this as:

- a circle metric
- `k = 3`
- `m = 6`
- a small instance with about 40 nodes and 240 inequalities

This is a warmup setting with only one provided instance.

Practical implication for the agent:

- optimize for the circle `k = 3, m = 6` case
- do not assume additional training instances are available
- the search logic can be simpler and more targeted than in multi-instance regimes
