from __future__ import annotations


class Node:
    def __init__(self, value, operation: str | None = None):
        self.value = value
        self.operation = operation

    def __str__(self):
        op_string = f", op={self.operation}" if self.operation else ""
        return f"<Node: value={self.value}{op_string}>"

    def __repr__(self):
        return str(self)

    def __add__(self, value: Node | int | float):
        to_add = value.value if isinstance(value, Node) else value

        return Node(
            value=self.value + to_add,
            operation="+",
        )

    def __mul__(self, value: Node | int | float):
        to_mul = value.value if isinstance(value, Node) else value

        return Node(
            value=self.value * to_mul,
            operation="*",
        )

    def __pow__(self, number: int | float):
        return Node(
            value=self.value**number,
            operation="**",
        )

    def __neg__(self):
        return self * -1

    def __sub__(self, value: Node | int | float):
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
        return Node(
            value=0 if self.value < 0 else self.value,
            operation="ReLU",
        )
