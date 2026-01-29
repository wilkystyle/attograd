# attograd: A toy autograd framework
This is not a production-grade framework. I built this framework to help me better understand low-level neural network internals.

"Atto" is the [SI prefix](https://www.nist.gov/pml/owm/metric-si-prefixes#Prefixes) for 10^(-18), and is a nice play on the word "autograd". Attograd is smaller and less-featurful than the following frameworks, hence the smaller SI prefix:

- https://github.com/tinygrad/tinygrad
- https://github.com/karpathy/micrograd
- https://github.com/PABannier/nanograd
- https://github.com/shubhamwagh/picograd
- https://github.com/queelius/femtograd

## Prerequisites
- uv
- Python 3.10+

## Installation

```bash
uv seed -cp 3.12  # Or your desired Python version

uv sync --frozen  # Installs packages from lockfile
```

## Development

- Add new packages with `uv add PACKGE-NAME`
- Update the lockfile with `uv lock`
