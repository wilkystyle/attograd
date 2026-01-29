# attograd: A toy autograd framework
This is not a production-grade framework. I built this framework to help me better understand low-level neural network internals.

"Atto" is the [SI prefix](https://www.nist.gov/pml/owm/metric-si-prefixes#Prefixes) for 10<sup>-18</sup>, and is a nice play on the word "autograd". Attograd is smaller and less-featurful than the following frameworks, hence the smaller SI prefix:

- https://github.com/tinygrad/tinygrad
- https://github.com/karpathy/micrograd
- https://github.com/PABannier/nanograd
- https://github.com/shubhamwagh/picograd
- https://github.com/queelius/femtograd

## Prerequisites
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Installation

```bash
# Create a virtual environment
uv venv -cp 3.12  # Or your desired Python version

# Install packages from lockfile into your virtual environment
uv sync --frozen
```

## Development
- Add new packages with `uv add PACKGE-NAME`
- Update the lockfile with `uv lock`
