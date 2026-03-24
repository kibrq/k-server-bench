import numpy as np

from kserver.metrics.circle import Circle
from kserver.metrics.utils import antipode_extension


def test_antipode_extension_block_structure() -> None:
    d = np.array(
        [
            [0, 1, 3],
            [1, 0, 2],
            [3, 2, 0],
        ],
        dtype=int,
    )
    ext = antipode_extension(d)
    r = int(np.max(d))
    m = d.shape[0]

    np.testing.assert_array_equal(ext[:m, :m], d)
    np.testing.assert_array_equal(ext[m:, m:], d)
    np.testing.assert_array_equal(ext[:m, m:], r - d)
    np.testing.assert_array_equal(ext[m:, :m], r - d)


def test_antipode_extension_matches_doubled_circle_metric() -> None:
    # Check antipode-pair distances and symmetry on a circle input.
    m = 4
    d = Circle.discrete(m)
    ext = antipode_extension(d)
    r = int(np.max(d))

    assert ext.shape == (2 * m, 2 * m)
    np.testing.assert_array_equal(ext, ext.T)
    np.testing.assert_array_equal(np.diag(ext), np.zeros(2 * m, dtype=ext.dtype))
    for i in range(m):
        assert ext[i, i + m] == r
        assert ext[i + m, i] == r
