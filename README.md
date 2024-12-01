# Qermit

[![PyPI version](https://badge.fury.io/py/qermit.svg)](https://badge.fury.io/py/qermit)

Qermit is a python module for running error-mitigation protocols on quantum processors.
It is an extension to the [pytket](https://docs.quantinuum.com/tket) quantum computing toolkit.

This repo contains source code and API documentation.
For details on building the docs please see `docs/README.md`

## Getting Started

To install, run:
```
pip install qermit
```
API documentation can be found at [qerm.it](https://qerm.it).

## Bugs

Please file bugs on the Github
[issue tracker](https://github.com/CQCL/Qermit/issues).

## Contributing

Pull requests or feature suggestions are very welcome.
To make a PR, first fork the repo, make your proposed changes on the `main` branch, and open a PR from your fork.

## Code style

To install the dependencies required for the following run:
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

On the CI, [mypy](https://mypy.readthedocs.io/en/stable/) is used as a static type checker.
To check these checks would pass please run:
```
mypy -p qermit
```

## Tests

To run tests use:
```
cd tests
pytest
```

When adding a new feature, please add a test for it.
When fixing a bug, please add a test that demonstrates the fix.

## How to cite

If you wish to cite Qermit, we recommend citing our [benchmarking paper](https://quantum-journal.org/papers/q-2023-07-13-1059/) where possible.
