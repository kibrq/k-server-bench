import numpy as np
import pytest

from kserver.metrics.circle import Circle


def test_circle_discrete_m4_exact_matrix() -> None:
    got = Circle.discrete(4)
    expected = np.array(
        [
            [0, 1, 2, 1],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [1, 2, 1, 0],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(got, expected)


def test_circle_discrete_m5_exact_matrix() -> None:
    got = Circle.discrete(5)
    expected = np.array(
        [
            [0, 1, 2, 2, 1],
            [1, 0, 1, 2, 2],
            [2, 1, 0, 1, 2],
            [2, 2, 1, 0, 1],
            [1, 2, 2, 1, 0],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("m", [1, 2, 5, 9])
def test_circle_discrete_basic_invariants(m: int) -> None:
    d = Circle.discrete(m)
    assert d.shape == (m, m)
    np.testing.assert_array_equal(d, d.T)
    np.testing.assert_array_equal(np.diag(d), np.zeros(m, dtype=int))


@pytest.mark.parametrize(
    "m,a,b,expected",
    [
        (10, 1.0, 4.0, 3.0),
        (10, 1.0, 9.0, 2.0),
        (10, 0.0, 5.0, 5.0),
        (8, 7.0, 1.0, 2.0),
        (8, -1.0, 1.0, 2.0),
        (8, 9.0, 1.0, 0.0),
        (12, 3.5, 9.5, 6.0),
    ],
)
def test_circle_continous_distance_values(m: int, a: float, b: float, expected: float) -> None:
    dist = Circle.continous(m)
    assert dist(a, b) == pytest.approx(expected)
    assert dist(b, a) == pytest.approx(expected)
