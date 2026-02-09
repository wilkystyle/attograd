import numpy as np

from attograd import Tensor


def test_confirm_tinygrad_readme():
    x = Tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    y = Tensor(
        [
            [2.0, 0.0, -2.0],
        ],
        dtype=np.float32,
    )

    z = y.dot(x).sum()

    z.backward()

    assert np.array_equal(x.grad, np.array([[2.0, 2.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -2.0, -2.0]]))

    assert np.array_equal(y.grad, np.array([[1.0, 1.0, 1.0]]))
