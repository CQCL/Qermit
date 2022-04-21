# Copyright 2019-2021 Cambridge Quantum Computing
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pytket.backends import Backend
from qermit import (
    MitEx,
    MitRes,
    ObservableTracker,
    AnsatzCircuit,
    MitTask,
    ObservableExperiment,
    TaskGraph,
)
from copy import copy
from pytket.pauli import QubitPauliString  # type: ignore
from enum import Enum
import numpy as np
from scipy.optimize import curve_fit  # type: ignore
from typing import List, Tuple, cast, Dict
from pytket import Circuit, OpType
from pytket.predicates import CompilationUnit  # type: ignore
from pytket.utils import QubitPauliOperator
import matplotlib.pyplot as plt  # type: ignore
from numpy.polynomial.polynomial import Polynomial
from pytket.circuit import Node  # type: ignore


box_types = {
    OpType.CircBox,
    OpType.ExpBox,
    OpType.PauliExpBox,
    OpType.Unitary1qBox,
    OpType.Unitary2qBox,
    OpType.Unitary3qBox,
}


class Folding(Enum):
    """Folding techniques used to increase the noise levels.

    :return: Circuit with gate count increased to scale noise. The unitary implemented by the
        circuit has not changed.
    :rtype: Circuit
    """

    # TODO: circ does not appear as input in docs
    # TODO Generalise with 'partial folding' to allow for non integer noise scaling
    def circuit(circ: Circuit, noise_scaling: int, **kwargs) -> Circuit:
        """Noise scaling by circuit folding. In this case the folded circuit is of
        the form :math:`CC^{-1}CC^{-1}...C` where :math:`C` is the original circuit. As such noise may be scaled by
        odd integers. The Unitary implemented is unchanged by this process.

        :param circ: Original circuit to be folded.
        :type circ: Circuit
        :param noise_scaling: Factor by which to scale the noise. This must be an odd integer.
        :type noise_scaling: int
        :raises ValueError: Raised if the amount by which the noise should be scaled is not an odd integer.
        :return: Folded circuit implementing identical unitary to the initial circuit.
        :rtype: Circuit
        """

        # Raise if the amount by which the noise should be scaled is not an odd integer
        if (not noise_scaling % 2) or noise_scaling % 1:
            raise ValueError(
                "When performing circuit folding, the noise scaling must be an odd integer."
            )

        folded_circ = circ.copy()
        for _ in range(noise_scaling // 2):

            # Add barrier between circuit and its inverse
            folded_circ.add_barrier(folded_circ.qubits + folded_circ.bits)

            # Add inverse circuit by iterating though commands and inverting them
            for gate in reversed(circ.get_commands()):
                if gate.op.type == OpType.Barrier:
                    folded_circ.add_barrier(gate.args)
                elif gate.op.type in box_types:
                    raise RuntimeError("Box types not supported when folding.")
                else:
                    folded_circ.add_gate(gate.op.dagger, gate.args)

            # Add barrier between circuit and its inverse
            folded_circ.add_barrier(folded_circ.qubits + folded_circ.bits)

            # Add original circuit
            for gate in circ.get_commands():
                if gate.op.type == OpType.Barrier:
                    folded_circ.add_barrier(gate.args)
                elif gate.op.type in box_types:
                    raise RuntimeError("Box types not supported when folding.")
                else:
                    folded_circ.add_gate(gate.op, gate.args)

        return folded_circ

    def gate(circ: Circuit, noise_scaling: float, **kwargs) -> Circuit:
        """Noise scaling by gate folding. In this case gates :math:`G` are replaced at random
        with :math:`GG^{-1}G` until the number of gates is sufficiently scaled.

        :param circ: Original circuit to be folded.
        :type circ: Circuit
        :param noise_scaling: Factor by which to increase the noise.
        :type noise_scaling: float

        :key _allow_approx_fold: Allows for the noise to be increased by an amount close to that requested, as
            opposed to by exactly the amount requested.
            This is necessary as there are cases where the exact noise scaling cannot be achieved.
            This occurs due to the discrete
            amounts by which the noise can be increased (i.e. the discrete amount by which one gate increases the noise).
        :type _allow_approx_fold: bool

        :raises ValueError: Raised if the requested noise scaling cannot be exactly achieved. This can be
            avoided by appropriately setting _allow_approx_fold.
        :return: Folded circuit implementing identical unitary to the initial circuit.
        :rtype: Circuit
        """

        _allow_approx_fold = kwargs.get("_allow_approx_fold", 0)

        c_dict = circ.to_dict()
        num_commands = len(c_dict["commands"])

        # Count and locate the number of commands that are not barriers
        c_dict_commands_non_barrier = [
            (i, command)
            for (i, command) in enumerate(c_dict["commands"])
            if command["op"]["type"] != "Barrier"
        ]
        num_non_barrier_commands = len(c_dict_commands_non_barrier)

        # Calculate the number of gates that need to be folded. Note that only
        # non barrier commands should be folded so only these are counted.
        num_additional_commands = (noise_scaling - 1) * num_non_barrier_commands
        num_folded_commands = int(num_additional_commands // 2)

        true_noise_scaling = 1 + ((num_folded_commands * 2) / num_non_barrier_commands)
        # This check isolates the case where the noise folding that can be achieved is different
        # from that requested. While noise_scaling can be any real value, as the gates increase
        # the noise by discrete values, not all folding values are possible.
        if (not _allow_approx_fold) and (
            abs(noise_scaling - true_noise_scaling) > 0.001
        ):
            raise ValueError(
                "The noise cannot be scaled by the amount inputted. The noise must be scaled by a factor of the form (#gates + 2i)/#gates, where #gates is the number of gates in the compiled circuit, and i is an integer."
            )

        # Choose a random selection of commands to fold. Commands are referenced by their index.
        # These are chosen with replacement. Only non barrier commands should
        # be folded, and so only the index of those are added here.
        commands_to_fold = np.random.choice(
            [i[0] for i in c_dict_commands_non_barrier], num_folded_commands
        )
        # Calculate how many times each individual command needs to be folded
        num_folds = {i: list(commands_to_fold).count(i) for i in range(num_commands)}

        command_circ_dict = {
            key: val for key, val in c_dict.items() if key != "commands"
        }

        folded_command_list = []
        # For each command, fold the appropriate number of times.
        for command_index in num_folds:

            command = c_dict["commands"][command_index]
            command_circ_dict.update({"commands": [command]})
            command_circ = Circuit().from_dict(command_circ_dict)

            # Commands which are not to be folded may not be invertible (for
            # example barriers) and so the inverse is not calculated in that
            # case.
            if num_folds[command_index] > 0:
                # Find the inverse of the command
                inverse_command_circ = command_circ.dagger()
                inverse_command_circ_dict = inverse_command_circ.to_dict()
                inverse_command = inverse_command_circ_dict["commands"]

            # Append command and inverse the appropriate number of times.
            folded_command_list.append(command)
            for _ in range(num_folds[command_index]):
                folded_command_list.append(
                    {
                        "args": command["args"],
                        "op": {
                            "signature": ["Q"] * len(command["args"]),
                            "type": "Barrier",
                        },
                    }
                )
                folded_command_list.append(*inverse_command)
                folded_command_list.append(
                    {
                        "args": command["args"],
                        "op": {
                            "signature": ["Q"] * len(command["args"]),
                            "type": "Barrier",
                        },
                    }
                )
                folded_command_list.append(command)

        folded_c_dict = c_dict.copy()
        folded_c_dict["commands"] = folded_command_list
        folded_c = Circuit().from_dict(folded_c_dict)

        return folded_c

    def odd_gate(circ: Circuit, noise_scaling: int, **kwargs) -> Circuit:
        """Noise scaling by gate folding. In this case odd gates :math:`G` are
        replaced :math:`GG^{-1}G` until the number of gates is sufficiently
        scaled.

        :param circ: Original circuit to be folded.
        :type circ: Circuit
        :param noise_scaling: Factor by which to increase the noise.
        :type noise_scaling: float
        :return: Folded circuit implementing identical unitary to the initial circuit.
        :rtype: Circuit
        """

        c_dict = circ.to_dict()

        fold = True
        folded_command_list = []

        command_circ_dict = {
            key: val for key, val in c_dict.items() if key != "commands"
        }

        for command in c_dict["commands"]:

            # Barriers are added to the circuit but otherwise effectively
            # skipped.
            if command["op"]["type"] == "Barrier":

                folded_command_list.append(command)

            elif fold:
                command_circ_dict.update({"commands": [command]})
                command_circ = Circuit.from_dict(command_circ_dict)

                # Find the inverse of the command
                inverse_command_circ = command_circ.dagger()
                inverse_command_circ_dict = inverse_command_circ.to_dict()
                inverse_command = inverse_command_circ_dict["commands"]

                # Append command and inverse the appropriate number of times.
                folded_command_list.append(command)
                for _ in range(noise_scaling - 1):
                    folded_command_list.append(
                        {
                            "args": command["args"],
                            "op": {
                                "signature": ["Q"] * len(command["args"]),
                                "type": "Barrier",
                            },
                        }
                    )
                    folded_command_list.append(*inverse_command)
                    folded_command_list.append(
                        {
                            "args": command["args"],
                            "op": {
                                "signature": ["Q"] * len(command["args"]),
                                "type": "Barrier",
                            },
                        }
                    )
                    folded_command_list.append(command)

                fold = not fold
            else:
                folded_command_list.append(command)

                fold = not fold

        c_dict.update({"commands": folded_command_list})
        folded_c = Circuit.from_dict(c_dict)

        return folded_c


def poly_exp_func(x: float, *params) -> float:
    """Definition of poly-exponential function for the purposes of fitting to data

    :param x: Value at which to evaluate function
    :type x: float
    :return: Evaluation of poly-exponential function
    :rtype: float
    """
    # Note that we use a list ending in 0 so that the constant in the polynomial is 0.
    # The constant can then be absorbed into the coefficient of the exponential.
    return params[0] + params[1] * np.exp(np.polyval([*params[2:], 0], x))


def cube_root_func(x: float, *params) -> float:
    """Definition of cube root function for the purposes of fitting to data.

    :param x: Value at which to evaluate cube root function
    :type x: float
    :return: Evaluation of cube root function
    :rtype: float
    """
    y = x + params[2]
    return params[0] + params[1] * np.sign(y) * (np.abs(y)) ** (1 / 3)


class Fit(Enum):
    """Functions to fit to expectation values as they change with noise.

    :return: Extrapolation of expectation values to the zero noise limit.
    :rtype: float
    """

    # TODO Consider adding adaptive exponential extrapolation
    def cube_root(
        self, x: List[float], y: List[float], _show_fit: bool, *args
    ) -> float:
        """Fit data to a cube root function. This is to say a function of the form :math:`a + b(x+c)^{1/3}`.

        :param x: Noise scaling values.
        :type x: List[float]
        :param y: Expectation values.
        :type y: List[float]
        :param _show_fit: Plot data and resulting fitted function.
        :type _show_fit: bool
        :return: Extrapolation of data to zero noise limit using the best fitting cube root function.
        :rtype: float
        """

        # Fit data to cube root function
        vals = curve_fit(cube_root_func, x, y, p0=[0, 1, 0], maxfev=100000)

        # Evaluate fitted function at 0
        fit_to_zero = cube_root_func(0, *vals[0])

        # Plot fitted function and data
        if _show_fit:

            fit_x = np.linspace(0, max(x), 100)
            fit_y = [cube_root_func(i, *vals[0]) for i in fit_x]

            plot_fit(x, y, cast(List[float], fit_x), fit_y, fit_to_zero)

        return float(fit_to_zero)

    def poly_exponential(
        self, x: List[float], y: List[float], _show_fit: bool, deg: int
    ) -> float:
        """Fit data to a poly-exponential, which is to say a function of the
        form :math:`a+e^{z}`, where :math:`z` is a polynomial.

        :param x: Noise scaling values.
        :type x: List[float]
        :param y: Expectation values.
        :type y: List[float]
        :param _show_fit: Plot data and resulting fitted function.
        :type _show_fit: bool
        :param deg: The degree of the polynomial in the exponential.
        :type deg: int
        :raises ValueError: Raised if the degree of the polynomial
            inputted is negative, or too high to fit to the data.
        :return: Extrapolation of data to the zero noise limit using the best fitting
            poly-exponential function of the specified degree.
        :rtype: float
        """

        # check that the degree of the polynomial is positive, and small enough to fit
        # to the inputted data
        if (deg + 2 > len(x)) or (deg < 0):
            raise ValueError(
                "The degree of the polynomial must be positive and can be at most is m-2, where m is the number of data points. In this case the largest permitted degree is %i."
                % (len(x) - 2)
            )

        # Fit data to polyexponential function
        # TODO: Improve bounds here.
        bounds = (
            [-1, -2, *[-np.inf for i in range(deg)]],
            [1, 2, *[np.inf for i in range(deg)]],
        )

        # Initialise decaying poly-exponential with intersect at
        # unfolded noisy value.
        least_noisy_y_index = x.index(1)
        p0 = [0, y[least_noisy_y_index], *[-1 for i in range(deg)]]

        vals = curve_fit(
            poly_exp_func,
            x,
            y,
            p0=p0,
            maxfev=10000,
            bounds=bounds,
        )

        # Extrapolate function to zero noise limit
        fit_to_zero = poly_exp_func(0, *vals[0])

        # Plot data and fitted function
        if _show_fit:

            fit_x = np.linspace(0, max(x), 100)
            fit_y = [poly_exp_func(i, *vals[0]) for i in fit_x]

            plot_fit(x, y, cast(List[float], fit_x), fit_y, fit_to_zero)

        return float(fit_to_zero)

    def exponential(
        self, x: List[float], y: List[float], _show_fit: bool, *args
    ) -> float:
        """Fit data to an exponential function. This is to say a function of the form :math:`a+e^{(b+x)}`.
        Note that this is a special case of the poly-exponential function.

        :param x: Noise scaling values.
        :type x: List[float]
        :param y: Expectation values.
        :type y: List[float]
        :param _show_fit: Plot data and resulting fitting function.
        :type _show_fit: bool
        :return: Extrapolation to zero noise limit using the best fitting exponential function.
        :rtype: float
        """

        # As the exponential function is a special case of the
        # poly-exponential function, it is called here
        return Fit.poly_exponential(self, x, y, _show_fit, 1)

    def polynomial(
        self, x: List[float], y: List[float], _show_fit: bool, deg: int
    ) -> float:
        """Fit data to a polynomial function.

        :param x: Noise scaling values.
        :type x: List[float]
        :param y: Expectation values.
        :type y: List[float]
        :param _show_fit: Plot data and resulting fitting function.
        :type _show_fit: bool
        :param deg: The degree of the function to fit to.
        :type deg: int
        :raises ValueError: Raised if the degree of the polynomial is negative,
            or too high to fit the data to.
        :return: Extrapolation to zero noise limit using the best fitting polynomial
            function of the specified degree.
        :rtype: float
        """

        # Raised if the degree of the polynomial requested is negative, or too high to fit the data to
        if (deg > len(x)) or (deg < 0):
            raise ValueError(
                "The degree of the polynomial must be positive and can be at most m-1, where m is the number of data points."
            )

        # Fit data to a polynomial
        fit = Polynomial.fit(x, [float(i) for i in y], deg=deg)

        # Extrapolate to the zero noise limit
        fit_to_zero = fit.convert().coef[0]

        # Plot data and fitted function
        if _show_fit:

            linspace = fit.linspace()
            fit_x = linspace[0]
            fit_y = linspace[1]

            plot_fit(x, y, fit_x, fit_y, fit_to_zero)

        return float(fit_to_zero)

    def linear(self, x: List[float], y: List[float], _show_fit: bool, *args) -> float:
        """Fit data to a linear function. This is to say a function of the form :math:`ax+b`.
        Note that this is a special case of the polynomial fitting function.

        :param x: Noise scaling values.
        :type x: List[float]
        :param y: Expectation values.
        :type y: List[float]
        :param _show_fit: Plot data and resulting fitted function.
        :type _show_fit: bool
        :return: Extrapolation to zero noise limit using the best fitting linear function.
        :rtype: float
        """
        # As this is a special case of a fit to a polynomial, the polynomial
        # fitting function is called here with a degree 1
        return Fit.polynomial(self, x, y, _show_fit, 1)

    def richardson(
        self, x: List[float], y: List[float], _show_fit: bool, *args
    ) -> float:
        """Use richardson extrapolation. This amounts to fitting to a polynomial of
        degree one less than the number of data points.

        :param x: Noise scaling values.
        :type x: List[float]
        :param y: Expectation values.
        :type y: List[float]
        :param _show_fit: Plot data and resulting fitted function.
        :type _show_fit: bool
        :return: Extrapolation to zero noise limit using Richardson extrapolation.
        :rtype: float
        """
        # As this is a special case of the polynomial fitting function, the polynomial fitting
        # function is called here with degree one less than the number of data points.
        return Fit.polynomial(self, x, y, _show_fit, len(x) - 1)


def plot_fit(
    x: List[float],
    y: List[float],
    fit_x: List[float],
    fit_y: List[float],
    fit_to_zero: float,
):
    """Plot expectation values at each noise level, and the fit to the data derived.

    :param x: Amounts by which the noise has been scaled
    :type x: List[float]
    :param y: Expectation values at each noise level
    :type y: List[float]
    :param fit_x: x coordinates at which to plot value of fitted function
    :type fit_x: List[float]
    :param fit_y: Value of fitted function at each noise scaling
    :type fit_y: List[float]
    :param fit_to_zero: The extrapolation of the fitted function to the zero noise limit
    :type fit_to_zero: float
    """

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(x, y, s=10, c="b", marker="s", label="data")
    ax1.plot(fit_x, fit_y, c="g", label="fit")

    ax1.scatter(0, fit_to_zero, s=10, c="r", marker="o", label="prediction")

    plt.legend()
    plt.show()


def digital_folding_task_gen(
    backend: Backend,
    noise_scaling: float,
    _folding_type: Folding,
    _allow_approx_fold: bool,
) -> MitTask:
    """
    Generates task transforming a circuit in order to amplify the noise. The noise
    is increased by a factor noise_scaling using the inputted folding method.

    :param backend: This will be used to compile the circuit after folding to ensure
        that the gate set matches those available on the backend.
    :type backend: Backend
    :param noise_scaling: The factor by which the noise is increased.
    :type noise_scaling: float
    :param _folding_type: The means by which the noise should be increased.
    :type _folding_type: Folding
    :param _allow_approx_fold:  Allows for the noise to be increased by an amount close to that requested, as
        opposed to by exactly the amount requested.
        This is necessary as there are cases where the exact noise scaling cannot be achieved.
        This occurs due to the discrete
        amounts by which the noise can be increased (i.e. the discrete amount by which one gate increases the noise).
    :type _allow_approx_fold: bool

    """

    def task(
        obj,
        mitex_wire: List[ObservableExperiment],
    ) -> Tuple[List[ObservableExperiment]]:
        """Increase the noise levels impacting the circuit by increasing the
        number of gates. This preserves the action of the circuit.

        :param mitex_wire: List of experiments
        :type mitex_wire: List[ObservableExperiment]
        :return: List of equivalent circuits, but with noise levels increased.
        :rtype: Tuple[List[ObservableExperiment]]
        """

        folded_circuits = []

        # For each circuit in the input wire, extract the circuit, apply the fold,
        # and perform the necessary compilation.
        for experiment in mitex_wire:

            # Apply the necessary folding method
            zne_circ = _folding_type(experiment.AnsatzCircuit.Circuit, noise_scaling, _allow_approx_fold=_allow_approx_fold)  # type: ignore

            # TODO: This additional compilation pass may result in the circuit noise being
            # increased too much, and should be removed or better accounted for.

            # This compilation pass was added to account for the case that
            # the inverse of a gate is not in the gateset of the backend.
            backend.rebase_pass().apply(zne_circ)

            folded_circuits.append(
                ObservableExperiment(
                    AnsatzCircuit=AnsatzCircuit(
                        Circuit=zne_circ,
                        Shots=experiment.AnsatzCircuit.Shots,
                        SymbolsDict=experiment.AnsatzCircuit.SymbolsDict,
                    ),
                    ObservableTracker=experiment.ObservableTracker,
                )
            )

        return (folded_circuits,)

    return MitTask(_label="DigitalFolding", _n_in_wires=1, _n_out_wires=1, _method=task)


def extrapolation_task_gen(
    noise_scaling_list: List[float], _fit_type: Fit, _show_fit: bool, deg: int
) -> MitTask:
    """Generates task extrapolating to the zero noise limit using results from many folded circuits.

    :param noise_scaling_list: A list of the values by which the noise has been folded.
    :type noise_scaling_list: List[float]
    :param _fit_type: The function used to fit to the resulting data.
    :type _fit_type: Fit
    :param _show_fit: Plot data and resulting fitted function.
    :type _show_fit: bool
    :param deg: The degree of polynomials used.
    :type deg: int
    """

    def task(
        obj, base_exp_list: List[QubitPauliOperator], *args
    ) -> Tuple[List[QubitPauliOperator]]:
        """Returns expectation values corrected by extrapolation

        :param base_exp_list: List of expectation values corresponding to each experiment
        :type base_exp_list: List[QubitPauliOperator]
        :return: Each element of this tuple is a list of expectations for each
        experiment, all with noise scaled by a fixed amount.
        :rtype: Tuple[List[QubitPauliOperator]]
        """
        # Combine noise folding levels with the unfolded experiment
        all_fold_vals = [1, *noise_scaling_list]

        # Reformats to create list, where each list has fixed noise folding. Each element of
        # the list is a list of experiments
        expanded_args = [i for i in args]

        # Reformats so that all_fold_qpo_list_floats is a list with each element
        # corresponding to a fixed experiment. Each element itself is a dictionary {qps:list}
        # where the list is a list of expectation values from each noise folded experiment
        all_fold_qpo = [qpo for qpo in zip(base_exp_list, *expanded_args)]
        all_fold_qpo_list_floats = [
            {k: [d[k] for d in qpo_tuple] for k in qpo_tuple[0]._dict.keys()}
            for qpo_tuple in all_fold_qpo
        ]

        extrapolated = []

        # The list of expectations for each experiment is now used to
        # extrapolate to the ideal value
        for qpo_list_float in all_fold_qpo_list_floats:

            extrapolated.append(
                QubitPauliOperator(
                    {
                        qpo_k: _fit_type(  # type: ignore
                            obj, all_fold_vals, qpo_list_float[qpo_k], _show_fit, deg
                        )
                        for qpo_k in qpo_list_float
                    }
                )
            )

        return (extrapolated,)

    return MitTask(
        _label="CollateZNEResults",
        _n_in_wires=len(noise_scaling_list) + 1,
        _n_out_wires=1,
        _method=task,
    )


def copy_mitex_wire(wire: ObservableExperiment) -> ObservableExperiment:
    """Returns a single copy of the inputted wire

    :param wire: Pair of ansatz circuit and ObservableTracker
    :type wire: ObservableExperiment
    :return: single copy of inputted wire
    :rtype: ObservableExperiment
    """

    # Copy ansatz circuit
    new_ansatz_circuit = AnsatzCircuit(
        Circuit=wire.AnsatzCircuit.Circuit.copy(),
        Shots=copy(wire.AnsatzCircuit.Shots),
        SymbolsDict=copy(wire.AnsatzCircuit.SymbolsDict),
    )

    # copy qps and instantiate new measurement setup
    new_ot = ObservableTracker(
        QubitPauliOperator(copy(wire.ObservableTracker._qubit_pauli_operator._dict))
    )
    new_wire = ObservableExperiment(
        AnsatzCircuit=new_ansatz_circuit, ObservableTracker=new_ot
    )
    return new_wire


def gen_duplication_task(duplicates: int, **kwargs) -> MitTask:
    """Duplicate the inputted experiment wire

    :param duplicates: The number of times to duplicate the input wire.
    :type duplicates: int
    """

    def task(
        obj,
        mitex_wire: List[ObservableExperiment],
    ) -> Tuple[List[ObservableExperiment], ...]:
        """Duplicate the inputted experiment wire

        :param mitex_wire: List of experiments
        :type mitex_wire: List[ObservableExperiment]
        :raises ValueError: Raised if the number of duplications is less than 1
        :return: Many copies of the inputted wire
        :rtype: Tuple[List[ObservableExperiment]]
        """

        # Raise error if the number of duplications requested is less than 1
        if duplicates <= 0:
            raise ValueError(
                "The number of duplications must be greater than or equal to 1."
            )

        # Compose coppies into wire format
        if duplicates == 1:
            me_copy = [copy_mitex_wire(circuit_tuple) for circuit_tuple in mitex_wire]
            return (me_copy,)
        else:
            me_copies = [
                [copy_mitex_wire(circuit_tuple) for circuit_tuple in mitex_wire]
            ]
            for _ in range(duplicates - 1):
                mc = [copy_mitex_wire(circuit_tuple) for circuit_tuple in mitex_wire]
                me_copies.append(mc)
            return tuple(me_copies)

    return MitTask(
        _label=kwargs.get("_label", "Duplicate"),
        _n_out_wires=duplicates,
        _n_in_wires=1,
        _method=task,
    )


def qpo_node_relabel(qpo: QubitPauliOperator, node_map: Dict[Node, Node]):
    """Relabel the nodes of qpo according to node_map

    :param qpo: Original qubit pauli operator
    :type qpo: QubitPauliOperator
    :param node_map: Map between nodes
    :type node_map: Dict[Node,Node]
    :return: Relabeled qubit pauli operator
    :rtype: QubitPauliOperator
    """

    orig_qpo_dict = qpo._dict.copy()
    new_qpo_dict = {}
    for orig_qps in orig_qpo_dict:
        orig_qps_dict = orig_qps.map
        new_qps_dict = {}
        for q in orig_qps_dict:
            new_qps_dict[node_map[q]] = orig_qps_dict[q]
        new_qps = QubitPauliString(new_qps_dict)
        new_qpo_dict[new_qps] = orig_qpo_dict[orig_qps]

    return QubitPauliOperator(new_qpo_dict)


def gen_initial_compilation_task(
    backend: Backend, optimisation_level: int = 1
) -> MitTask:
    """Perform compilation to the backend. Note that this will relabel the
    nodes of the device, and so should be followed by gen_qubit_relabel_task
    in the task graph.

    :param backend: Backend to compile to
    :type backend: Backend
    :param optimisation_level: level of default compiler, defaults to 1
    :type optimisation_level: int, optional
    """

    def task(
        obj, wire: List[ObservableExperiment]
    ) -> Tuple[List[ObservableExperiment], Dict[Node, Node]]:
        """Performs initial compilation before folding. This is to ensure minimal compilation
        after folding, as this could disrupt by how much the noise is increased.

        :param wire: List of experiments
        :type wire: List[ObservableExperiment]
        :return: List of experiments compiled to run on the inputted backend.
        Additionally a dictionary describing how the nodes have been mapped
        by compilation.
        :rtype: Tuple[List[ObservableExperiment], Dict[Node, Node]]
        """

        mapped_wire = []

        for obs_exp in wire:

            # Perform default compilation, tracking to which physical
            # qubits the initial qubits are mapped
            compiled_circ = obs_exp.AnsatzCircuit.Circuit.copy()

            cu = CompilationUnit(compiled_circ)
            backend.default_compilation_pass(
                optimisation_level=optimisation_level
            ).apply(cu)
            node_map = cu.final_map

            # Alter the qubit pauli operator so that it maps to the new physical qubits.
            qpo = obs_exp[1]._qubit_pauli_operator
            new_qpo = qpo_node_relabel(qpo, node_map)

            # Construct new list of experiments, but with compiled circuits.
            mapped_wire.append(
                ObservableExperiment(
                    AnsatzCircuit=AnsatzCircuit(
                        Circuit=cu.circuit,
                        Shots=obs_exp.AnsatzCircuit.Shots,
                        SymbolsDict=obs_exp.AnsatzCircuit.SymbolsDict,
                    ),
                    ObservableTracker=ObservableTracker(new_qpo),
                )
            )

        return (
            mapped_wire,
            node_map,
        )

    return MitTask(
        _label="CompileToBackend",
        _n_out_wires=2,
        _n_in_wires=1,
        _method=task,
    )


def gen_qubit_relabel_task() -> MitTask:
    """Task reversing the relabelling of qubits performed during compilation.
    This should follow gen_initial_compilation_task

    :return: Task performing relabelling.
    :rtype: MitTask
    """

    def task(
        obj, qpo_list: List[QubitPauliOperator], compilation_map: Dict[Node, Node]
    ) -> Tuple[List[QubitPauliOperator]]:
        """Use node map returned by compilation unit to undo the relabelling
        performed by gen_initial_compilation_task

        :param qpo_list: List of QubitPauliOperator
        :type qpo_list: List[QubitPauliOperator]
        :param compilation_map: Dictionary mapping nodes as returned by
        gen_initial_compilation_task task
        :type compilation_map: Dict[Node, Node]
        :return: List of QubitPauliOperator with relabeled nodes.
        :rtype: Tuple[List[QubitPauliOperator]]
        """

        node_map = {value: key for key, value in compilation_map.items()}
        new_qpo_list = [qpo_node_relabel(qpo, node_map) for qpo in qpo_list]

        return (new_qpo_list,)

    return MitTask(
        _label="RelabelQubits",
        _n_out_wires=1,
        _n_in_wires=2,
        _method=task,
    )


# TODO: Backend does not appear as input in documentation
def gen_ZNE_MitEx(backend: Backend, noise_scaling_list: List[float], **kwargs) -> MitEx:
    """Generates MitEx object which mitigates for noise using Zero Noise Extrapolation. This is the
    process by which noise is amplified incrementally, and the zero noise case arrived at by
    extrapolating backwards. For further explanantion see https://arxiv.org/abs/2005.10921.

    :param backend: Backend on which the circuits are to be run.
    :type backend: Backend
    :param noise_scaling_list: A list of the amounts by which the noise should be scaled.
    :type noise_scaling_list: List[float]
    :return: MitEx object performing noise mitigation by ZNE.
    :rtype: MitEx
    """
    _experiment_mitres = copy(
        kwargs.get(
            "experiment_mitres",
            MitRes(backend),
        )
    )

    _experiment_mitex = copy(
        kwargs.get(
            "experiment_mitex",
            MitEx(backend, _label="ExperimentMitex", mitres=_experiment_mitres),
        )
    )
    _experiment_taskgraph = TaskGraph().from_TaskGraph(_experiment_mitex)

    _optimisation_level = kwargs.get("optimisation_level", 0)
    _show_fit = kwargs.get("show_fit", False)
    _folding_type = kwargs.get("folding_type", Folding.circuit)
    _fit_type = kwargs.get("fit_type", Fit.linear)
    _deg = kwargs.get("deg", len(noise_scaling_list) - 1)
    _seed = kwargs.get("seed", None)
    _allow_approx_fold = kwargs.get("allow_approx_fold", True)

    np.random.seed(seed=_seed)

    for fold in noise_scaling_list:

        _label = str(fold) + "FoldMitEx"

        _fold_mitres = copy(
            kwargs.get(
                "experiment_mitres",
                MitRes(backend),
            )
        )

        _fold_mitex = copy(
            kwargs.get(
                "experiment_mitex",
                MitEx(backend, _label=_label, mitres=_fold_mitres),
            )
        )

        digital_folding_task = digital_folding_task_gen(
            backend, fold, _folding_type, _allow_approx_fold
        )
        _fold_mitex.prepend(digital_folding_task)
        _experiment_taskgraph.parallel(_fold_mitex)

    extrapolation_task = extrapolation_task_gen(
        noise_scaling_list, _fit_type, _show_fit, _deg
    )

    _experiment_taskgraph.prepend(gen_duplication_task(len(noise_scaling_list) + 1))
    _experiment_taskgraph.append(extrapolation_task)

    _experiment_taskgraph.add_wire()

    _experiment_taskgraph.prepend(
        gen_initial_compilation_task(backend, _optimisation_level)
    )
    _experiment_taskgraph.append(gen_qubit_relabel_task())
    return MitEx(backend).from_TaskGraph(_experiment_taskgraph)
