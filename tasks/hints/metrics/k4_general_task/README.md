## Metrics Hint: k4 General Task

In this regime, you are given exactly four benchmark instance files:

- `circle_k4_m6.pickle`
- `circle_k4_m8.pickle`
- `circle_taxi_k4_m6.pickle`
- `circle_taxi_k4_m8.pickle`

Treat these as:

- circle and circle-taxi metrics
- `k = 4`
- `m = 6` and `m = 8`
- `circle_k4_m6.pickle`: `1001` nodes and `6006` inequalities
- `circle_k4_m8.pickle`: `32650` nodes and `261200` inequalities
- `circle_taxi_k4_m6.pickle`: `166681` nodes and `7000602` inequalities
- `circle_taxi_k4_m8.pickle`: `49387` nodes and `159139` inequalities

This is a multi-instance general-task setting.

Practical implication for the agent:

- the solution should be robust across all instances
- `circle_k4_m6.pickle` is an exact subset of `circle_taxi_k4_m6.pickle`
- it is a good idea to first iterate on the smaller `circle_k4_m6.pickle` instance before testing on the larger taxi variant
- `circle_taxi_k4_m6.pickle` is especially large, so evaluation cost matters
- evaluating in one thread on `circle_taxi_k4_m6.pickle` may take over a minute
