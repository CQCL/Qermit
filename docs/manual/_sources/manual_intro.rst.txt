***************
What is qermit?
***************

.. Two-sentence overview

The ``qermit`` framework is a software platform for the development and execution of 
error-mitigation protocols.
The toolset is designed to aid platform-agnostic software, making running a range
of combined error-mitigation protocols as straightforward as running any experiment.

This user manual is targeted at readers who are already familiar with 
`CQC <https://cambridgequantum.com/>`_  `pytket <https://github.com/CQCL/pytket>`_,
a python module for interfacing with tket, a set of quantum programming tools. It provides
a comprehensive tour of the ``qermit`` platform, from running basic unmitigated experiments with
``pytket`` circuits, to running tailored combinations of error-mitigation protocols to get
as much performance out of devices as possible.


What is Error-Mitigation and how does qermit perform it?
--------------------------------------------------------

It is common knowledge that we are currently in the NISQ-era of Quantum Computers; Noisy, Intermediate-Scale
Quantum Computers that have too few high fidelity Qubits for running Quantum Error Correction protocols on,
but are characterised as having high error rates such that even for quantum circuits with very few gates (10's),
running experiments on such devices lead to errors accruing quickly and output states being
dominated by noise.

As dominating noise is a key problem facing Quantum Computation, naturally many approaches to address it are available. 
Better quantum circuit compilation is one such approach. Circuit optimisation to reduce the number of logical 
gates in a quantum circuit and mapping passes for fitting logical circuits to physical constraints can reduce noise
by producing circuits that compute identical processes with fewer operations. These methods can be improved by 
being 'noise-aware', having an understanding of the device via noise characterisation and using this to
produce circuits that user less noisy qubits, such as those characterised with higher fidelity operations. Error-mitigation
methods provide another approach.

The name *error-mitigation* often functions as an umbrella term for a wide range of loosely-connected techniques
at all levels of the quantum computing stack. The loose thread between such methods is that they
*mitigate* errors in quantum computation, caused by noise in quantum devices in some capacity. 

``qermit`` restricts the scope of such a range of methods to those that are defined in the quantum circuit layer of abstraction.
This is a reasonable restriction to make as in many cases a fine 
understanding of how noise manifests isn't required to correct for the errors it produces, but only an understanding 
of the error that is produced (though a fine understanding is always helpful).

As an example, we can attempt to suppress the coherent quantum computation error produced by a systematic over-rotation of an
operation rotating a Qubit's state in the z plane without having a fine understanding of what calibration and control
problems are occuring in the quantum device. If we can understand what error occurs with what operations, we can design tools
to suppress them. 

In designing ``qermit``, the goal was to make using error-mitigation methods *easy*, easy to integrate into a
typical experiment workflow, easy to access a wide range of useful error-mitigation techniques, and easy to use
different error-mitigation techniques in combination.

To do so, error-mitigation methods in ``qermit`` fit in to two distinctions, ``MitRes`` methods 
that result in a modification of a distribution of counts  retrieved from some 
quantum computer, and ``MitEx`` methods that result in the modification of the 
expectation value of some observable. These correspond to two common archetypes for useage of quantum computers,
meaning they are not only useful for improvung results, but there is a wide and ever growing area of research
dedicated to designing mitigation schemes that fit to these archetypes.

In this manner, often the use of a ``MitRes`` or ``MitEx`` object may be able to replace code performing the fiddly aspect 
of running and processing experiments, with or without error-mitigation. Furthermore, as they are written using the 
``pytket`` `Backend <https://cqcl.github.io/pytket/build/html/backends.html>`_ class, 
any hardware supported by ``pytket`` via the Backends available in the `pytket-extensions <https://github.com/CQCL/pytket-extensions>`_ 
can be used in conjunction with ``qermit``.


Installation
------------
To install using the ``pip`` package manager, run ``pip install qermit`` from your terminal.

