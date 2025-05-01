# Copyright 2019-2023 Quantinuum
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
from copy import copy, deepcopy
from enum import Enum
from math import isclose
from typing import Dict, List, Tuple, Union, cast

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from pytket import Circuit, OpType, Qubit
from pytket.backends import Backend
from pytket.circuit import Node
from pytket.pauli import Pauli, QubitPauliString
from pytket.predicates import CompilationUnit
from pytket.unit_id import UnitID, UnitType
from pytket.utils import QubitPauliOperator
from pytket.utils.operators import CoeffTypeAccepted
from scipy.optimize import curve_fit  # type: ignore
from sympy import Expr  # type: ignore

from qermit import (
    AnsatzCircuit,
    MitEx,
    MitRes,
    MitTask,
    ObservableExperiment,
    ObservableTracker,
    TaskGraph,
)
from qermit.noise_model import NoiseModel, PauliErrorTranspile

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
    """

    # TODO: circ does not appear as input in docs
    # TODO Generalise with 'partial folding' to allow for non integer noise scaling
    @staticmethod
    def circuit(circ: Circuit, noise_scaling: int, **kwargs) -> List[Circuit]:
        """Noise scaling by circuit folding. In this case the folded circuit is of
        the form :math:`CC^{-1}CC^{-1}...C` where :math:`C` is the original circuit. As such noise may be scaled by
        odd integers. The Unitary implemented is unchanged by this process.

        :param circ: Original circuit to be folded.
        :param noise_scaling: Factor by which to scale the noise. This must be an odd integer.
        :raises ValueError: Raised if the amount by which the noise should be scaled is not an odd integer.
        :return: Folded circuit implementing identical unitary to the initial circuit.
        """

        # Raise if the amount by which the noise should be scaled is not an odd integer
        if (not noise_scaling % 2) or noise_scaling % 1:
            raise ValueError(
                "When performing circuit folding, the noise scaling must be an odd integer."
            )

        folded_circ = circ.copy()
        for _ in range(noise_scaling // 2):
            # Add barrier between circuit and its inverse
            folded_circ.add_barrier(
                cast(List[UnitID], folded_circ.qubits + folded_circ.bits)
            )

            # Add inverse circuit by iterating though commands and inverting them
            for gate in reversed(circ.get_commands()):
                if gate.op.type == OpType.Barrier:
                    folded_circ.add_barrier(gate.args)
                elif gate.op.type in box_types:
                    raise RuntimeError("Box types not supported when folding.")
                else:
                    folded_circ.add_gate(gate.op.dagger, gate.args)

            # Add barrier between circuit and its inverse
            folded_circ.add_barrier(
                cast(List[UnitID], folded_circ.qubits + folded_circ.bits)
            )

            # Add original circuit
            for gate in circ.get_commands():
                if gate.op.type == OpType.Barrier:
                    folded_circ.add_barrier(gate.args)
                elif gate.op.type in box_types:
                    raise RuntimeError("Box types not supported when folding.")
                else:
                    folded_circ.add_gate(gate.op, gate.args)

        return [folded_circ]

    @staticmethod
    def two_qubit_gate(circ: Circuit, noise_scaling: float, **kwargs) -> List[Circuit]:
        """Noise scaling by folding 2 qubit gates. It is implicitly
        assumed that the noise on the 2 qubit gates dominate. Two qubit gates
        :math:`G` are replaced by :math:`GG^{-1}G...G^{-1}G`. If
        `noise_scaling` is of the form (#gates + 2i)/#gates,
        where #gates is the number of gates in the compiled circuit and i is
        an integer, then the noise scaling is exact. It will otherwise be
        as close as possible to but smaller then noise_scaling.

        :param circ: Original circuit to be folded.
        :param noise_scaling: Factor by which the noise should be scaled.

        :raises ValueError: Raised if noise_scaling is less than 1.
        :raises ValueError: Raised if the noise cannot be scaled by
            exactly noise_scaling and `_allow_approx_fold` is not True.
        :raises RuntimeError: Raised if there are no valid gates to fold.
        :raises RuntimeError: Raised if the circuit includes boxes.

        :key _allow_approx_fold: True or false depending on if
            approximate folding is allowed. Defaults to True.

        :return: Circuit with noise scaled.
        """

        if noise_scaling < 1:
            raise ValueError("noise_scaling must be greater than or equal to 1")

        _allow_approx_fold = kwargs.get("_allow_approx_fold", True)

        # All gates will be folded by an amount equal to the even number
        # less than noise_scaling-1. By doing so the noise is scaled by the
        # odd integer less than noise_scaling.
        num_folds_dict = {
            i: (int(noise_scaling - 1) // 2)
            for i, cmd in enumerate(circ.get_commands())
            if (
                not (cmd.op.type == OpType.Barrier)
                and (cmd.op.type not in box_types)
                and (len(cmd.qubits) == 2)
            )
        }

        if len(num_folds_dict) == 0:
            raise RuntimeError(
                "There are no valid q qubits gates in this circuit to fold. "
                "Your circuit should include 2 qubit gates other than "
                "Barrier and CircBox."
            )

        # The remaining noise scaling is achieved by randomly selecting gates
        # to scale. fold_frac gives the fraction of gates which need to be
        # folded to achieve noise_scaling. The appropriate fraction of
        # gates is then randomly selected.
        fold_frac = ((noise_scaling - 1) % 2) / 2
        to_fold = np.random.choice(
            list(num_folds_dict.keys()),
            size=int(len(num_folds_dict.keys()) * fold_frac),
            replace=False,
        )
        for i in to_fold:
            num_folds_dict[i] += 1

        true_noise_scaling = float(sum(2 * i + 1 for i in num_folds_dict.values()))
        true_noise_scaling /= len(num_folds_dict)

        if not (
            _allow_approx_fold
            or isclose(noise_scaling, true_noise_scaling, abs_tol=0.001)
        ):
            raise ValueError(
                "The noise cannot be scaled by the amount inputted."
                "The noise must be scaled by a factor of the form "
                "(#gates + 2i)/#gates, where #gates is the number of gates "
                "in the compiled circuit, and i is an integer."
            )

        # Copy qubit register of original circuit.
        folded_circuit = Circuit()
        for qubit in circ.qubits:
            folded_circuit.add_qubit(qubit)

        for i, gate in enumerate(circ.get_commands()):
            # Barriers are not folded and added as given.
            if gate.op.type == OpType.Barrier:
                folded_circuit.add_barrier(gate.args)
            # Boxes are not supported.
            elif gate.op.type in box_types:
                raise RuntimeError("Box types not supported when folding.")
            # 2 qubit gates are folded.
            elif len(gate.qubits) == 2:
                folded_circuit.add_gate(gate.op, gate.args)
                for _ in range(num_folds_dict[i]):
                    folded_circuit.add_barrier(gate.args)
                    folded_circuit.add_gate(gate.op.dagger, gate.args)
                    folded_circuit.add_barrier(gate.args)
                    folded_circuit.add_gate(gate.op, gate.args)
            # All other gates are added as given.
            else:
                folded_circuit.add_gate(gate.op, gate.args)

        return [folded_circuit]

    @staticmethod
    def gate(circ: Circuit, noise_scaling: float, **kwargs) -> List[Circuit]:
        """Noise scaling by gate folding. In this case gates :math:`G` are replaced at random
        with :math:`GG^{-1}G` until the number of gates is sufficiently scaled.

        :param circ: Original circuit to be folded.
        :param noise_scaling: Factor by which to increase the noise.

        :key _allow_approx_fold: Allows for the noise to be increased by an amount close to that requested, as
            opposed to by exactly the amount requested.
            This is necessary as there are cases where the exact noise scaling cannot be achieved.
            This occurs due to the discrete
            amounts by which the noise can be increased (i.e. the discrete amount by which one gate increases the noise).

        :raises ValueError: Raised if the requested noise scaling cannot be exactly achieved. This can be
            avoided by appropriately setting _allow_approx_fold.
        :return: Folded circuit implementing identical unitary to the initial circuit.
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

        return [folded_c]

    @staticmethod
    def odd_gate(circ: Circuit, noise_scaling: int, **kwargs) -> List[Circuit]:
        """Noise scaling by gate folding. In this case odd gates :math:`G` are
        replaced :math:`GG^{-1}G` until the number of gates is sufficiently
        scaled.

        :param circ: Original circuit to be folded.
        :param noise_scaling: Factor by which to increase the noise.
        :return: Folded circuit implementing identical unitary to the initial circuit.
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

        return [folded_c]

    @staticmethod
    def noise_aware(circ: Circuit, noise_scaling: int, **kwargs) -> List[Circuit]:
        """Scale noise in a circuit by adding noisy gates as defined
        by the given noise model.

        :param circ: Circuit with noise to be scaled.
        :param noise_scaling: Factor by which noise should be scaled.
        :return: List of circuits with additional noise gates added.

        :key noise_model: Noise model defining noise types and rates.
            Defaults to noiseless model.
        :key n_noisy_circuit_samples: The number of random noisy
            circuits to generate. Defaults to 1.
        """

        noise_model: NoiseModel = kwargs.get("noise_model", NoiseModel(noise_model={}))
        n_noisy_circuit_samples: int = kwargs.get("n_noisy_circuit_samples", 1)

        scaled_noise_model: NoiseModel = noise_model.scale(
            scaling_factor=noise_scaling - 1
        )
        error_transpiler = PauliErrorTranspile(noise_model=scaled_noise_model)

        scaled_circ_list = []
        for _ in range(n_noisy_circuit_samples):
            scaled_circ = circ.copy()
            error_transpiler.apply(scaled_circ)
            scaled_circ_list.append(scaled_circ)

        return scaled_circ_list


def poly_exp_func(x: float, *params) -> float:
    """Definition of poly-exponential function for the purposes of fitting to data

    :param x: Value at which to evaluate function
    :return: Evaluation of poly-exponential function
    """
    # Note that we use a list ending in 0 so that the constant in the polynomial is 0.
    # The constant can then be absorbed into the coefficient of the exponential.
    return params[0] + params[1] * np.exp(np.polyval([*params[2:], 0], x))


def cube_root_func(x: float, *params) -> float:
    """Definition of cube root function for the purposes of fitting to data.

    :param x: Value at which to evaluate cube root function
    :return: Evaluation of cube root function
    """
    y = x + params[2]
    return params[0] + params[1] * np.sign(y) * (np.abs(y)) ** (1 / 3)


class Fit(Enum):
    """Functions to fit to expectation values as they change with noise.

    :return: Extrapolation of expectation values to the zero noise limit.
    """

    # TODO Consider adding adaptive exponential extrapolation
    @staticmethod
    def cube_root(x: List[float], y: List[float], _show_fit: bool, *args) -> float:
        """Fit data to a cube root function. This is to say a function of the form :math:`a + b(x+c)^{1/3}`.

        :param x: Noise scaling values.
        :param y: Expectation values.
        :param _show_fit: Plot data and resulting fitted function.
        :return: Extrapolation of data to zero noise limit using the best fitting cube root function.
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

    @staticmethod
    def poly_exponential(
        x: List[float], y: List[float], _show_fit: bool, deg: int
    ) -> float:
        """Fit data to a poly-exponential, which is to say a function of the
        form :math:`a+e^{z}`, where :math:`z` is a polynomial.

        :param x: Noise scaling values.
        :param y: Expectation values.
        :param _show_fit: Plot data and resulting fitted function.
        :param deg: The degree of the polynomial in the exponential.
        :raises ValueError: Raised if the degree of the polynomial
            inputted is negative, or too high to fit to the data.
        :return: Extrapolation of data to the zero noise limit using the best fitting
            poly-exponential function of the specified degree.
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

    @staticmethod
    def exponential(x: List[float], y: List[float], _show_fit: bool, *args) -> float:
        """Fit data to an exponential function. This is to say a function of the form :math:`a+e^{(b+x)}`.
        Note that this is a special case of the poly-exponential function.

        :param x: Noise scaling values.
        :param y: Expectation values.
        :param _show_fit: Plot data and resulting fitting function.
        :return: Extrapolation to zero noise limit using the best fitting exponential function.
        """

        # As the exponential function is a special case of the
        # poly-exponential function, it is called here
        return Fit.poly_exponential(x, y, _show_fit, 1)

    @staticmethod
    def polynomial(x: List[float], y: List[float], _show_fit: bool, deg: int) -> float:
        """Fit data to a polynomial function.

        :param x: Noise scaling values.
        :param y: Expectation values.
        :param _show_fit: Plot data and resulting fitting function.
        :param deg: The degree of the function to fit to.
        :raises ValueError: Raised if the degree of the polynomial is negative,
            or too high to fit the data to.
        :return: Extrapolation to zero noise limit using the best fitting polynomial
            function of the specified degree.
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

            plot_fit(x, y, list(fit_x), list(fit_y), fit_to_zero)

        return float(fit_to_zero)

    @staticmethod
    def linear(x: List[float], y: List[float], _show_fit: bool, *args) -> float:
        """Fit data to a linear function. This is to say a function of the form :math:`ax+b`.
        Note that this is a special case of the polynomial fitting function.

        :param x: Noise scaling values.
        :param y: Expectation values.
        :param _show_fit: Plot data and resulting fitted function.
        :return: Extrapolation to zero noise limit using the best fitting linear function.
        """
        # As this is a special case of a fit to a polynomial, the polynomial
        # fitting function is called here with a degree 1
        return Fit.polynomial(x, y, _show_fit, 1)

    @staticmethod
    def richardson(x: List[float], y: List[float], _show_fit: bool, *args) -> float:
        """Use richardson extrapolation. This amounts to fitting to a polynomial of
        degree one less than the number of data points.

        :param x: Noise scaling values.
        :param y: Expectation values.
        :param _show_fit: Plot data and resulting fitted function.
        :return: Extrapolation to zero noise limit using Richardson extrapolation.
        """
        # As this is a special case of the polynomial fitting function, the polynomial fitting
        # function is called here with degree one less than the number of data points.
        return Fit.polynomial(x, y, _show_fit, len(x) - 1)


def plot_fit(
    x: List[float],
    y: List[float],
    fit_x: List[float],
    fit_y: List[float],
    fit_to_zero: float,
):
    """Plot expectation values at each noise level, and the fit to the data derived.

    :param x: Amounts by which the noise has been scaled
    :param y: Expectation values at each noise level
    :param fit_x: x coordinates at which to plot value of fitted function
    :param fit_y: Value of fitted function at each noise scaling
    :param fit_to_zero: The extrapolation of the fitted function to the zero noise limit
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
    **kwargs,
) -> MitTask:
    """
    Generates task transforming a circuit in order to amplify the noise. The noise
    is increased by a factor noise_scaling using the inputted folding method.

    :param backend: This will be used to compile the circuit after folding to ensure
        that the gate set matches those available on the backend.
    :param noise_scaling: The factor by which the noise is increased.
    :param _folding_type: The means by which the noise should be increased.
    :param _allow_approx_fold:  Allows for the noise to be increased by an amount close to that requested, as
        opposed to by exactly the amount requested.
        This is necessary as there are cases where the exact noise scaling cannot be achieved.
        This occurs due to the discrete
        amounts by which the noise can be increased (i.e. the discrete amount by which one gate increases the noise).
    """

    def task(
        obj,
        mitex_wire: List[ObservableExperiment],
    ) -> Tuple[List[ObservableExperiment], List[int]]:
        """Increase the noise levels impacting the circuit by increasing the
        number of gates. This preserves the action of the circuit.

        :param mitex_wire: List of experiments
        :return: List of equivalent circuits, but with noise levels increased.
            Each noise scaling value may generate multiple circuits.
            As such the return includes a lit of integers indicating to which
            of the original circuits the new circuits belong.
        """

        folded_circuits = []
        experiment_index = []

        # For each circuit in the input wire, extract the circuit, apply the fold,
        # and perform the necessary compilation.
        for index, experiment in enumerate(mitex_wire):
            # Apply the necessary folding method
            zne_circ_list = _folding_type(
                experiment.AnsatzCircuit.Circuit,
                noise_scaling,
                _allow_approx_fold=_allow_approx_fold,
                **kwargs,
            )  # type: ignore

            for zne_circ in zne_circ_list:
                # TODO: This additional compilation pass may result in the circuit noise being
                # increased too much, and should be removed or better accounted for.

                # This compilation pass was added to account for the case that
                # the inverse of a gate is not in the gateset of the backend.
                backend.rebase_pass().apply(zne_circ)

                folded_circuits.append(
                    ObservableExperiment(
                        AnsatzCircuit=AnsatzCircuit(
                            Circuit=zne_circ,
                            Shots=experiment.AnsatzCircuit.Shots // len(zne_circ_list),
                            SymbolsDict=deepcopy(experiment.AnsatzCircuit.SymbolsDict),
                        ),
                        ObservableTracker=deepcopy(experiment.ObservableTracker),
                    )
                )
                experiment_index.append(index)

        return (folded_circuits, experiment_index)

    return MitTask(_label="DigitalFolding", _n_in_wires=1, _n_out_wires=2, _method=task)


def merge_experiments_task_gen() -> MitTask:
    """Generates task merging qubit pauli strings when they belong to the
    same experiment.

    :return: MitTask performing the merge.
    """

    def task(
        obj, qpo_list: List[QubitPauliOperator], experiment_index_list: List[int]
    ) -> Tuple[List[QubitPauliOperator]]:
        """Merge a list of qubit pauli operators. Qubit pauli operators will
        merged if they belong to the same experiment. The experiment each
        qubit pauli operator belongs to is indicated by the index list.
        Merging here means that the mean of the pauli string coefficients
        is taken to be the coefficient of the qubit pauli string in the
        new qubit pauli operator.

        :param qpo_list: List of qubit pauli strings, some of which may belong
            to the same experiment.
        :param experiment_index_list: Indexes indicating to which experiment
            the qubit pauli strings belong to.
        :return: A list of merged qubit pauli operators.
        """

        # A dictionary mapping the experiment index to the dictionary
        # of the merged qubit pauli operator.
        index_to_merged_qpo_dict: Dict[
            int, Dict[QubitPauliString, CoeffTypeAccepted]
        ] = {}

        # For each qubit pauli operator, sum the coefficients of the
        # qubit pauli strings it contains.
        for index, qpo in zip(experiment_index_list, qpo_list):
            index_qpo_dict = index_to_merged_qpo_dict.get(
                index, {qps: 0 for qps in qpo._dict.keys()}
            )

            if not index_qpo_dict.keys() == qpo._dict.keys():
                raise Exception(
                    "The qubit pauli strings being merged do not contain "
                    + "matching qubit pauli strings. In particular "
                    + f"{index_qpo_dict.keys()} and {qpo._dict.keys()} differ."
                )

            # Sum qubit pauli string coefficients.
            index_to_merged_qpo_dict[index] = {
                qps: qpo._dict[qps] + index_qpo_dict[qps] for qps in qpo._dict.keys()
            }

        # For each index, find the mean coefficient.
        for index, merged_qpo_dict in index_to_merged_qpo_dict.items():
            index_to_merged_qpo_dict[index] = {
                qps: coeff / experiment_index_list.count(index)
                for qps, coeff in merged_qpo_dict.items()
            }

        # Convert the dictionary to a list and return.
        return (
            [
                QubitPauliOperator(dictionary=index_to_merged_qpo_dict.get(index, {}))
                for index in range(max(experiment_index_list) + 1)
            ],
        )

    return MitTask(
        _label="MergeExperiments", _n_in_wires=2, _n_out_wires=1, _method=task
    )


def extrapolation_task_gen(
    noise_scaling_list: List[float], _fit_type: Fit, _show_fit: bool, deg: int
) -> MitTask:
    """Generates task extrapolating to the zero noise limit using results from many folded circuits.

    :param noise_scaling_list: A list of the values by which the noise has been folded.
    :param _fit_type: The function used to fit to the resulting data.
    :param _show_fit: Plot data and resulting fitted function.
    :param deg: The degree of polynomials used.
    """

    def task(
        obj, base_exp_list: List[QubitPauliOperator], *args
    ) -> Tuple[List[QubitPauliOperator]]:
        """Returns expectation values corrected by extrapolation

        :param base_exp_list: List of expectation values corresponding to each experiment
        :return: Each element of this tuple is a list of expectations for each
        experiment, all with noise scaled by a fixed amount.
        """

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
                            noise_scaling_list, qpo_list_float[qpo_k], _show_fit, deg
                        )
                        for qpo_k in qpo_list_float
                    }
                )
            )

        return (extrapolated,)

    return MitTask(
        _label="CollateZNEResults",
        _n_in_wires=len(noise_scaling_list),
        _n_out_wires=1,
        _method=task,
    )


def copy_mitex_wire(wire: ObservableExperiment) -> ObservableExperiment:
    """Returns a single copy of the inputted wire

    :param wire: Pair of ansatz circuit and ObservableTracker
    :return: single copy of inputted wire
    """

    # Copy ansatz circuit
    new_ansatz_circuit = AnsatzCircuit(
        Circuit=deepcopy(wire.AnsatzCircuit.Circuit),
        Shots=deepcopy(wire.AnsatzCircuit.Shots),
        SymbolsDict=deepcopy(wire.AnsatzCircuit.SymbolsDict),
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
    """

    def task(
        obj,
        mitex_wire: List[ObservableExperiment],
    ) -> Tuple[List[ObservableExperiment], ...]:
        """Duplicate the inputted experiment wire

        :param mitex_wire: List of experiments
        :raises ValueError: Raised if the number of duplications is less than 1
        :return: Many copies of the inputted wire
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


def qpo_node_relabel(
    qpo: QubitPauliOperator,
    node_map: Dict[Union[UnitID, Qubit, Node], Union[UnitID, Qubit, Node]],
):
    """Relabel the nodes of qpo according to node_map

    :param qpo: Original qubit pauli operator
    :param node_map: Map between nodes
    :return: Relabeled qubit pauli operator
    """

    orig_qpo_dict = qpo._dict.copy()
    new_qpo_dict: Dict[QubitPauliString, Union[int, float, complex, Expr]] = {}
    for orig_qps in orig_qpo_dict:
        orig_qps_dict = orig_qps.map
        new_qps_dict = {}
        for q in orig_qps_dict:
            new_qps_dict[node_map[q]] = orig_qps_dict[q]
        new_qps = QubitPauliString(cast(Dict[Qubit, Pauli], new_qps_dict))
        new_qpo_dict[new_qps] = orig_qpo_dict[orig_qps]

    return QubitPauliOperator(new_qpo_dict)


def gen_initial_compilation_task(
    backend: Backend, optimisation_level: int = 1
) -> MitTask:
    """Perform compilation to the backend. Note that this will relabel the
    nodes of the device, and so should be followed by gen_qubit_relabel_task
    in the task graph.

    :param backend: Backend to compile to
    :param optimisation_level: level of default compiler, defaults to 1
    """

    def task(
        obj, wire: List[ObservableExperiment]
    ) -> Tuple[List[ObservableExperiment], List[Dict[UnitID, UnitID]]]:
        """Performs initial compilation before folding. This is to ensure minimal compilation
        after folding, as this could disrupt by how much the noise is increased.

        :param wire: List of experiments
        :return: List of experiments compiled to run on the inputted backend.
            Additionally a list of dictionaries describing how the nodes have
            been mapped by compilation.
        """

        mapped_wire = []
        node_map_list = []

        for obs_exp in wire:
            # Perform default compilation, tracking to which physical
            # qubits the initial qubits are mapped
            compiled_circ = obs_exp.AnsatzCircuit.Circuit.copy()

            cu = CompilationUnit(compiled_circ)
            backend.default_compilation_pass(
                optimisation_level=optimisation_level
            ).apply(cu)
            node_map = cu.final_map
            node_map_list.append(node_map)

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
            node_map_list,
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
    """

    def task(
        obj,
        qpo_list: List[QubitPauliOperator],
        compilation_map_list: List[Dict[Node, Node]],
    ) -> Tuple[List[QubitPauliOperator]]:
        """Use node map returned by compilation unit to undo the relabelling
        performed by gen_initial_compilation_task

        :param qpo_list: List of QubitPauliOperator
        :param compilation_map_list: List of Dictionaries mapping nodes as
            returned by gen_initial_compilation_task task
        :return: List of QubitPauliOperator with relabeled nodes.
        """

        new_qpo_list = []

        for compilation_map, qpo in zip(compilation_map_list, qpo_list):
            node_map = {value: key for key, value in compilation_map.items()}
            new_qpo_list.append(
                qpo_node_relabel(
                    qpo,
                    cast(
                        Dict[Union[UnitID, Qubit, Node], Union[UnitID, Qubit, Node]],
                        node_map,
                    ),
                )
            )

        return (new_qpo_list,)

    return MitTask(
        _label="RelabelQubits",
        _n_out_wires=1,
        _n_in_wires=2,
        _method=task,
    )


def gen_noise_scaled_mitex(
    backend: Backend,
    noise_scaling: float,
    **kwargs,
) -> MitEx:
    """Generates MitEx with noise scaled by the Qermit Folding methods.

    :param backend: Backend on which circuits are run.
    :param noise_scaling: Factor by which noise is scaled.

    :return: MitEx with scaled noise.

    :key experiment_mitres: MitRes on which circuits are run, defaults to
        a MitRes wrapped around the given backend.
    :key experiment_mitex: MitEx on which the circuits are run, defaults
        to a MitEx wrapped around experiment_mitres
    :key allow_approx_fold: Allow approximate folding which may occur as a
        result of discreet folding due to adding gates.
    :key folding_type: The noise scaling method to use.
    """

    _experiment_mitres = deepcopy(
        kwargs.get(
            "experiment_mitres",
            MitRes(backend),
        )
    )

    _experiment_mitex = deepcopy(
        kwargs.get(
            "experiment_mitex",
            MitEx(backend, _label="ExperimentMitex", mitres=_experiment_mitres),
        )
    )

    _allow_approx_fold = kwargs.get("allow_approx_fold", True)
    _folding_type = kwargs.get("folding_type", Folding.circuit)

    _fold_mitex = deepcopy(_experiment_mitex)
    _fold_mitex._label = str(noise_scaling) + "FoldMitEx" + _fold_mitex._label

    digital_folding_task = digital_folding_task_gen(
        backend, noise_scaling, _folding_type, _allow_approx_fold, **kwargs
    )

    _fold_taskgraph = TaskGraph().from_TaskGraph(_fold_mitex)
    _fold_taskgraph.add_wire()
    _fold_taskgraph.prepend(digital_folding_task)
    _fold_taskgraph.append(merge_experiments_task_gen())

    return MitEx(backend).from_TaskGraph(_fold_taskgraph)


# TODO: Backend does not appear as input in documentation
def gen_ZNE_MitEx(backend: Backend, noise_scaling_list: List[float], **kwargs) -> MitEx:
    """Generates MitEx object which mitigates for noise using Zero Noise Extrapolation. This is the
    process by which noise is amplified incrementally, and the zero noise case arrived at by
    extrapolating backwards. For further explanantion see https://arxiv.org/abs/2005.10921.

    :param backend: Backend on which the circuits are to be run.
    :param noise_scaling_list: A list of the amounts by which the noise should be scaled.
    :return: MitEx object performing noise mitigation by ZNE.
    """
    _optimisation_level = kwargs.get("optimisation_level", 0)
    _show_fit = kwargs.get("show_fit", False)
    _fit_type = kwargs.get("fit_type", Fit.linear)
    _deg = kwargs.get("deg", len(noise_scaling_list) - 1)
    _seed = kwargs.get("seed", None)

    np.random.seed(seed=_seed)

    _zne_taskgraph = TaskGraph().from_TaskGraph(
        gen_noise_scaled_mitex(
            noise_scaling=noise_scaling_list[0],
            backend=backend,
            **kwargs,
        )
    )

    for fold in noise_scaling_list[1:]:
        _zne_taskgraph.parallel(
            gen_noise_scaled_mitex(
                noise_scaling=fold,
                backend=backend,
                **kwargs,
            )
        )

    _zne_taskgraph.prepend(gen_duplication_task(len(noise_scaling_list)))
    _zne_taskgraph.append(
        extrapolation_task_gen(noise_scaling_list, _fit_type, _show_fit, _deg)
    )

    _zne_taskgraph.add_wire()

    _zne_taskgraph.prepend(gen_initial_compilation_task(backend, _optimisation_level))
    _zne_taskgraph.append(gen_qubit_relabel_task())

    return MitEx(backend).from_TaskGraph(_zne_taskgraph)
