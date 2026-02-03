from __future__ import annotations

from typing import Callable


class Scalar:
    value: int | float
    children: set[Scalar]
    op: str | None
    grad: float
    _backward: Callable

    def __init__(
        self,
        value: int | float,
        children: tuple[Scalar, ...] = (),
        op: str | None = None,
    ):
        self.value = value
        self.children = set(children)
        self.op = op

        self.grad = 0.0
        self._backward = lambda: None

    def __str__(self):
        op_string = f", op={self.op}" if self.op else ""
        return f"<Scalar: value={self.value}{op_string}>"

    def __repr__(self):
        return str(self)

    def __add__(self, other: Scalar | int | float):
        other = other if isinstance(other, Scalar) else Scalar(other)

        parent = Scalar(
            value=self.value + other.value,
            children=(self, other),
            op="+",
        )

        def _backward():
            self.grad += parent.grad
            other.grad += parent.grad

        parent._backward = _backward

        return parent

    def __mul__(self, other: Scalar | int | float):
        other = other if isinstance(other, Scalar) else Scalar(other)

        parent = Scalar(
            value=self.value * other.value,
            children=(self, other),
            op="*",
        )

        def _backward():
            self.grad += parent.grad * other.value
            other.grad += parent.grad * self.value

        parent._backward = _backward

        return parent

    def __pow__(self, number: int | float):
        if not isinstance(number, (int, float)):
            raise ValueError("Only int or float are supported for power")

        parent = Scalar(
            value=self.value**number,
            children=(self,),
            op=f"**{number}",
        )

        def _backward():
            self.grad += (number * self.value ** (number - 1)) * parent.grad

        parent._backward = _backward

        return parent

    def __neg__(self):
        return self * -1

    def __sub__(self, value: Scalar | int | float):
        return self + (-value)

    def __radd__(self, value):
        return self + value

    def __rsub__(self, value):
        return self - value

    def __rmul__(self, value):
        return self * value

    def __truediv__(self, value):
        return self * value**-1

    def __rtruediv__(self, value):
        return value * self**-1

    def relu(self):
        parent = Scalar(
            value=0 if self.value < 0 else self.value,
            children=(self,),
            op="ReLU",
        )

        def _backward():
            self.grad += parent.grad * (parent.value > 0)

        parent._backward = _backward
        return parent

    def backward(self):
        nodes = []
        seen = set()

        # Defining post_order_dfs() inside the backward() function so that
        # references to `nodes` and `seen` are kept for all child nodes calling
        # post_order_dfs().
        def post_order_dfs(node: Scalar):
            if node not in seen:
                seen.add(node)

                for child in node.children:
                    post_order_dfs(child)

                # Post-order depth-first search: Adding self to the list of
                # nodes only after all child nodes have been visited. This is a
                # requirement for backpropagation.
                nodes.append(node)

        post_order_dfs(self)

        self.grad = 1

        # Starting at the parent, work backward and apply the chain rule,
        # propagating local derivatives.
        for node in reversed(nodes):
            node._backward()
