from attograd import Node


def test_karpathy_micrograd_readme():
    """Forward pass operations to confirm we get the same values as Micrograd README.

    See https://github.com/karpathy/micrograd/blob/c911406e5ace8742e5841a7e0df113ecb5d54685/README.md#example-usage
    """

    a = Node(-4.0)
    b = Node(2.0)

    c = a + b
    d = a * b + b**3

    c += c + 1
    c += 1 + c + (-a)

    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()

    e = c - d
    f = e**2

    g = f / 2.0
    g += 10.0 / f

    assert g.value == 24.70408163265306

    g.backward()

    assert f"{a.grad:.4f}" == "138.8338"
    assert f"{b.grad:.4f}" == "645.5773"
