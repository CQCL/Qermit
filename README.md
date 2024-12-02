# Qermit

[![PyPI version](https://badge.fury.io/py/qermit.svg)](https://badge.fury.io/py/qermit)

Qermit is a python module for running error-mitigation protocols on quantum processors.
It is an extension to the [pytket](https://docs.quantinuum.com/tket) quantum computing toolkit.

This repository contains source code and API documentation.
For details on building the docs please see `docs/README.md`

## Getting Started

To install, run:
```
pip install qermit
```
You may also wish to install the package from source:
```
pip install -e .
```
A `poetry.lock` file is included for use with [poetry](https://python-poetry.org/docs/cli/#install).

API documentation can be found at [qerm.it](https://qerm.it).

## Bugs

Please file bugs on the Github
[issue tracker](https://github.com/CQCL/Qermit/issues).

## Contributing

Pull requests or feature suggestions are very welcome.
To make a PR, first fork the repository, make your proposed changes, and open a PR from your fork.

## Code style

Style checks are run by continuous integration.
To install the dependencies required to run them locally run:
```
pip install qermit[tests]
```

### Formatting

This repository uses [ruff](https://docs.astral.sh/ruff/) for formatting and linting.
To check if your changes meet these standards run:
```
ruff check
ruff format --check
```

### Type annotation

[mypy](https://mypy.readthedocs.io/en/stable/) is used as a static type checker.
```
mypy -p qermit
```

## Tests

Tests are run by continuous integration.
To install the dependencies required to run them locally run:
```
pip install qermit[tests]
```

To run tests use:
```
cd tests
pytest
```

When adding a new feature, please add a test for it.
When fixing a bug, please add a test that demonstrates the fix.

## How to cite

If you wish to cite Qermit, we recommend citing our [benchmarking paper](https://quantum-journal.org/papers/q-2023-07-13-1059/) where possible.
