from typing import Any, Callable

import numpy as np


class Tensor:
    def __init__(
        self,
        values: Any,
        op="",
        children=(),
        dtype=np.float32,
    ):
        self.values = np.array(values)
        self.op: str = op
        self.children: set[Tensor] = set(children)
        self.dtype = dtype

        # All tensors start with zero gradients
        self.zero_grad()

        # The appropriate local derivative method will be set during a
        # mathematical operation.
        self._backward: Callable = lambda: None

    def __hash__(self):
        return id(self)

    def zero_grad(self):
        self.grad = np.zeros_like(self.values)

    def dot(self, other):
        parent = Tensor(
            values=self.values.dot(other.values),
            op="*",
            children=(self, other),
            dtype=self.dtype,
        )

        def _backward():
            self.grad += parent.grad.dot(other.values.T)
            other.grad += self.values.T.dot(parent.grad)

        parent._backward = _backward
        return parent

    def sum(self):
        parent = Tensor(
            values=self.values.sum(),
            op="+",
            children=(self,),
            dtype=self.dtype,
        )

        def _backward():
            self.grad += parent.grad * np.ones_like(self.values)

        parent._backward = _backward
        return parent

    def relu(self):
        parent = Tensor(
            values=(np.abs(self.values) + self.values) / 2,
            op="ReLU",
            children=(self,),
            dtype=self.dtype,
        )

        def _backward():
            self.grad += parent.grad * (self.values > 0)

        parent._backward = _backward
        return parent

    def backward(self):
        tensors = []
        seen: set[Tensor] = set()

        # Defining post_order_dfs() inside the backward() function so that
        # references to `nodes` and `seen` are kept for all child nodes calling
        # post_order_dfs().
        def post_order_dfs(tensor: Tensor):
            if tensor not in seen:
                seen.add(tensor)

                for child in tensor.children:
                    post_order_dfs(child)

                # Post-order depth-first search: Adding self to the list of
                # nodes only after all child nodes have been visited. This is a
                # requirement for backpropagation.
                tensors.append(tensor)

        post_order_dfs(self)

        self.grad = np.ones_like(self.values)

        # Starting at the parent, work backward and apply the chain rule,
        # propagating local derivatives.
        for tensor in reversed(tensors):
            tensor._backward()
