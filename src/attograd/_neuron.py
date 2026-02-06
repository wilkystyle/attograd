import numpy as np

from ._scalar import Scalar

rng = np.random.default_rng()


class Neuron:
    """The fundamental element of a Neural network."""

    def __init__(self, n_inputs):
        # Naive weight initialization
        # TODO: Look into something like He/Xavier initialization
        self.weights: list[Scalar] = [
            Scalar(value=rng.uniform(low=-1.0, high=1.0)) for _ in range(n_inputs)
        ]

        self.bias: Scalar = Scalar(value=0.0)

    def input(self, inputs: list[Scalar]) -> Scalar:
        return sum(
            [w * x for w, x in zip(self.weights, inputs, strict=True)],
            self.bias,
        ).relu()

    @property
    def parameters(self) -> list[Scalar]:
        return self.weights + [self.bias]
