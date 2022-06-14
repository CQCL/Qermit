# Qermit

[![PyPI version](https://badge.fury.io/py/qermit.svg)](https://badge.fury.io/py/qermit)

`qermit` is a python module for running error-mitigation protocols on quantum processors using [`pytket`](https://github.com/CQCL/pytket), the Cambridge Quantum python module for interfacing with [CQC](https://cambridgequantum.com/) TKET, a set of quantum programming tools.

This repo containts API documentation, a user manual for getting started with `qermit` and source code.

## Getting Started

`qermit` is compatible with the `pytket` 1.0 release and so is available for Python 3.8, 3.9, 3.10 on Linux, MacOS and Windows.
To install, run:

``pip install qermit``

API documentation can be found at [cqcl.github.io/Qermit](https://cqcl.github.io/Qermit).

To get a more in depth explanation of Qermit and its features including how to construct custom methods see the [manual](https://cqcl.github.io/Qermit/manual/) which includes examples.



## Bugs

Please file bugs on the Github
[issue tracker](https://github.com/CQCL/Qermit/issues).

## How to cite

If you wish to cite Qermit in any academic publications, we generally recommend citing our [benchmarking paper](https://doi.org/10.48550/arXiv.2204.09725) for most cases.

## Contributing

Pull requests or feature suggestions are very welcome. To make a PR, first fork the repo, make your proposed
changes on the `main` branch, and open a PR from your fork. If it passes
tests and is accepted after review, it will be merged in.

### Code style

#### Formatting

All code should be formatted using
[black](https://black.readthedocs.io/en/stable/), with default options. 

#### Type annotation

On the CI, [mypy](https://mypy.readthedocs.io/en/stable/) is used as a static
type checker and all submissions must pass its checks. You should therefore run
`mypy` locally on any changed files before submitting a PR. 

### Tests

To run the tests:

1. `cd` into the `tests` directory;
2. ensure you have installed `pytest`;
3. run `pytest`.

When adding a new feature, please add a test for it. When fixing a bug, please
add a test that demonstrates the fix.
