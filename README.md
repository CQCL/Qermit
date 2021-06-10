# qermit

`qermit` is a python module for running error-mitigation protocols on quantum processors using [`pytket`](https://github.com/CQCL/pytket), CQC's python module for interfacing with [CQC](https://cambridgequantum.com/) tket, a set of quantum programming tools.

This repo containts API documentation, a user manual for getting started with `qermit` and source code.

## Getting Started

`qermit` is available for ``python3.7`` or higher, on Linux, MacOS and Windows.
To install, ensure that you have `pip` version 19 or above, and install using `pip` from a cloned repository.

**Documentation** can be found at [cqcl.github.io/qermit](https://cqcl.github.io/qermit).

A **User Manual** can be found at [cqcl.github.io/qermit/manual](https://cqcl.github.io/qermit/manual/).


## Bugs and feature requests

Please file bugs and feature requests on the Github
[issue tracker](https://github.com/CQCL/qermit/issues).


## Contributing

Pull requests are welcome. To make a PR, first fork the repo, make your proposed
changes on the `master` branch, and open a PR from your fork. If it passes
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
