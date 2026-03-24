def all_multicombinations(n: int, k: int) -> list[tuple[int, ...]]:
    """Generate nondecreasing length-k tuples over [0, n)."""
    if k < 0:
        raise ValueError("k must be non-negative")

    res: list[tuple[int, ...]] = []
    a = [0] * k

    def go(i: int, start: int) -> None:
        if i == k:
            res.append(tuple(a))
            return
        for v in range(start, n):
            a[i] = v
            go(i + 1, v)

    go(0, 0)
    return res
