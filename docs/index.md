---
file_format: mystnb
kernelspec:
  name: python3
---
# Qermit

Qermit is a python module for running error-mitigation protocols.
It contains an assortment of ready to use error-mitigation schemes.
Qermit supports straightforward composition of these existing
error-mitigation methods, and the development of new error-mitigation
schemes.

Qermit is an extension to the pytket quantum software development kit. 
Qermit uses the pytket {py:class}`~pytket.backends.Backend` class,
and so supports an array of backends.

## Getting Started

```{code-block} console
pip install qermit
```

Please visit the
[github repository](https://github.com/CQCL/Qermit/issues) if you would
like to install from source. We also welcome contributions to Qermit there!

If you notice any bugs while using Qermit we would much appreciate you
raising them as issues at that repository.

If you would like an further
support with Qermit please contact <tket-support@quantinuum.com>.

## Key Notions

Error-mitigation methods in Qermit are either one of two types:

- {py:class}`~qermit.taskgraph.mitres.MitRes` methods which modify a distribution of counts.
- {py:class}`~qermit.taskgraph.mitex.MitEx` methods which modify the expectation value of some observable.

While Qermit includes many advanced error-mitigation schemes,
in their basic capacity, {py:class}`~qermit.taskgraph.mitres.MitRes` and
{py:class}`~qermit.taskgraph.mitex.MitEx` objects will run
experiments without error-mitigation, as in the following example.

```{code-cell} ipython3
from qermit import MitRes, CircuitShots
from pytket import Circuit
from pytket.extensions.qiskit import AerBackend

# Define the experiment to be run.
# In this case, a Bell pair is prepared and measured 1000 times.
circ = Circuit(2,2).H(0).CX(0,1).measure_all()
circ_shots = CircuitShots(Circuit=circ, Shots=1000)

# Define the way the experiment will be run.
# In this case the default MitRes, without error-mitigation,
# will be performed.
mitres = MitRes(backend = AerBackend())

# Finally, run the experiment.
results = mitres.run([circ_shots])
results[0].get_counts()
```

Here we have introduced the {py:obj}`~qermit.taskgraph.mittask.CircuitShots`.
This can be thought of a defining experiment to run to generated a
distribution of shots. In this case the Bell state preparation circuit will
be run 1000 times to generate the desired distribution. The
{py:class}`~qermit.taskgraph.mitres.MitRes` definition can then be thought
of as specifying how the experiment will run, in this case without
error-mitigation,
and by using {py:class}`~pytket.extensions.qiskit.AerBackend`.
The input to the {py:meth}`~qermit.taskgraph.mitres.MitRes.run` method
is then a list {py:obj}`~qermit.taskgraph.mittask.CircuitShots`,
with the output being a list of
{py:class}`~pytket.backends.backendresult.BackendResult`.

{py:class}`~qermit.taskgraph.mitres.MitRes` and
{py:class}`~qermit.taskgraph.mitex.MitEx` are built from a graph of
{py:class}`~qermit.taskgraph.mittask.MitTask` objects.
A {py:class}`~qermit.taskgraph.mittask.MitTask` object is a pure function
which computes some basic step in an experiment. These
{py:class}`~qermit.taskgraph.mittask.MitTask` are composed into a
{py:class}`~qermit.taskgraph.task_graph.TaskGraph` which defined how
inputs and outputs pass between the tasks.
Indeed, {py:class}`~qermit.taskgraph.mitres.MitRes` and
{py:class}`~qermit.taskgraph.mitex.MitEx` are instances of a
{py:class}`~qermit.taskgraph.task_graph.TaskGraph`.

In its default construction, a {py:class}`~qermit.taskgraph.mitres.MitRes`
object will simply run each circuit through the backend it is defined by.
In the following we see one task, `CircuitsToHandles`, submitting the circuits
and generating handles for those experiments, and a second, `HandlesToResults`,
retrieving the results using those handles.
The information passed along the central wire is simply
those handles.

```{code-cell} ipython3
mitres.get_task_graph()
```

Similarly, in its default construction a
{py:class}`~qermit.taskgraph.mitex.MitEx` object will simply estimate
the expectation of each observable 
desired without applying any mitigation method.

```{code-cell} ipython3
from qermit import (
   MitEx,
   AnsatzCircuit,
   ObservableExperiment,
   ObservableTracker,
   SymbolsDict,
)
from pytket import Qubit
from pytket.pauli import Pauli, QubitPauliString
from pytket.utils import QubitPauliOperator

# Define the experiment to be conducted. In this case that consists
# of two parts.
# 1.  The circuit to run. In the case of MitEx we use AnsatzCircuit
#     which allows parametrised circuit. In this case we do not use
#     this capability however.
ansatz_circuit = AnsatzCircuit(
   Circuit=Circuit(3,3).X(0).X(1),
   Shots=50,
   SymbolsDict=SymbolsDict(),
)

# 2.  The observable to be measured. In this case the observable
#     is only the ZZ observable on qubits 1 and 2.
qubit_pauli_string = QubitPauliString([Qubit(1), Qubit(2)], [Pauli.Z, Pauli.Z])
qubit_pauli_operator = QubitPauliOperator({qubit_pauli_string: 1.0})

# These are combined to define the experiment.
experiment = ObservableExperiment(
   AnsatzCircuit = ansatz_circuit,
   ObservableTracker = ObservableTracker(qubit_pauli_operator)
)

# Now we can define how the experiment is run.
# In this case the default MitEx, without error-mitigation, is performed.
mitex = MitEx(backend = AerBackend())
mitex.run([experiment])
```

The {py:meth}`~qermit.taskgraph.mitex.MitEx.run` method takes a list of
{py:class}`~qermit.taskgraph.mittask.ObservableExperiment`
as an argument. Each
{py:class}`~qermit.taskgraph.mittask.ObservableExperiment`
contains the basic information required to estimate the expectation value
of an observable; a state preparation circuit, a dictionary between symbols
and parameter values (where appropriate),
a {py:class}`~pytket.utils.QubitPauliOperator` detailing the 
operator being measured and used for preparing measurement circuits,
and the number of shots to run for each measurement circuit.

Each experiment returns a {py:class}`~pytket.utils.QubitPauliOperator`
containing an expectation value for each internal
{py:class}`~pytket._tket.pauli.QubitPauliString`. In its default
version, this is achieved by appending a measurement circuit for each
{py:class}`~pytket._tket.pauli.QubitPauliString` to the ansatz circuit and
executing through the {py:class}`~pytket.backends.Backend`
the {py:class}`~qermit.taskgraph.mitex.MitEx` object is defined by.

```{code-cell} ipython3
mitex.get_task_graph()
```

## Contents

```{toctree}
:hidden:

manual/manual_index.rst
```


```{toctree}
:caption: API Reference:
:maxdepth: 2

taskgraph.rst
mitres.rst
mitex.rst
utils.rst
noise_model.rst
mittask.rst
measurement_reduction.rst
spam.rst
frame_randomisation.rst
postselection.rst
leakage_detection.rst
coherent_pauli_checks.rst
clifford_noise_characterisation.rst
zero_noise_extrapolation.rst
probabilistic_error_cancellation.rst
spectral_filtering.rst
```
