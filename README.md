# attograd: A toy autograd framework
This is not a production-grade framework. I built this framework to help me better understand low-level neural network internals.

"Atto" is the [SI prefix](https://www.nist.gov/pml/owm/metric-si-prefixes#Prefixes) for 10<sup>-18</sup>, and is a nice play on the word "autograd". Attograd is smaller and less-featureful than the following frameworks, hence the smaller SI prefix:

- [tinygrad/tinygrad](https://github.com/tinygrad/tinygrad)
- [karpathy/micrograd](https://github.com/karpathy/micrograd)
- [PABannier/nanograd](https://github.com/PABannier/nanograd)
- [shubhamwagh/picograd](https://github.com/shubhamwagh/picograd)
- [queelius/femtograd](https://github.com/queelius/femtograd)

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
- Add new packages with `uv add PACKAGE-NAME`
- Update the lockfile with `uv lock`
