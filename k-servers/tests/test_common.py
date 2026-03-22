import math

import pytest

from kserverclean.context.common import all_multicombinations


def test_all_multicombinations_negative_k_raises() -> None:
    with pytest.raises(ValueError, match="k must be non-negative"):
        all_multicombinations(4, -1)


def test_all_multicombinations_k_zero() -> None:
    assert all_multicombinations(5, 0) == [()]


def test_all_multicombinations_n_zero_positive_k() -> None:
    assert all_multicombinations(0, 3) == []


def test_all_multicombinations_small_exact_order() -> None:
    out = all_multicombinations(3, 2)
    assert out == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 2),
        (2, 2),
    ]


@pytest.mark.parametrize("n,k", [(1, 1), (2, 3), (3, 2), (4, 1), (4, 3)])
def test_all_multicombinations_count_matches_combinations_with_repetition(n: int, k: int) -> None:
    out = all_multicombinations(n, k)
    expected = math.comb(n + k - 1, k)
    assert len(out) == expected


@pytest.mark.parametrize("n,k", [(2, 3), (3, 3), (4, 2)])
def test_all_multicombinations_all_tuples_non_decreasing(n: int, k: int) -> None:
    out = all_multicombinations(n, k)
    for tup in out:
        assert len(tup) == k
        assert all(0 <= x < n for x in tup)
        assert list(tup) == sorted(tup)
