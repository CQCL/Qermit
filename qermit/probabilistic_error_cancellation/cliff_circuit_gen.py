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


from typing import List, Tuple

import numpy as np
from pytket import Circuit


def sample_q_mallows(n_qubits: int) -> Tuple[List[int], List[int]]:
    """
    Samples definitions of a hadamard layer and and permutation layer. These quantities
    are sampled from the `quantum Mallows distribution' which is detailed in
    https://arxiv.org/abs/2003.09412.

    :param n_qubits: Number of qubits on which the layers should act
    :return: A description of the gate layers. The hadamard layer
    is defined by a vector {0,1}^n describing on which qubits a Hadamard gate acts. The
    permutation layer is defined by a permutation of the vector (0,1,...,n_qubits)
    describing where each qubit should be permuted to.
    """

    # Hadamard layer
    hadamard_layer = [0 for _ in range(n_qubits)]
    # Permutation layer
    permutation = [0 for _ in range(n_qubits)]

    # Contains qubits that have not been assigned by the permutation
    avail_qubits = list(range(n_qubits))

    log2 = np.log(2.0)

    for i in range(n_qubits):
        m = len(avail_qubits)

        # Sample the hadamard layer and k according to the quantum Mallows distribution
        r = np.random.uniform(0, 1)
        index = int(2 * m - np.ceil(np.log(r * (4**m - 1) + 1) / log2))
        hadamard_layer[i] = 1 * (index < m)
        if index < m:
            k = index
        else:
            k = 2 * m - index - 1

        # Set this entry to be the k largest element of avail_qubits
        permutation[i] = avail_qubits[k]
        del avail_qubits[k]

    return hadamard_layer, permutation


def clifford_canonical_F(
    pauli_layer: List[int], gamma: np.ndarray, delta: np.ndarray
) -> Circuit:
    """
    Returns a Hadamard free Clifford circuit using the canonical form of elements of the Borel group
    introduced in https://arxiv.org/abs/2003.09412. The canonical form has the structure O P CZ CX where
    O is a pauli operator, P is a layer of sqrt(Z) gates, CZ is a layer of CZ gates, and CX is a layer of
    CX gates. The inputs describe on which qubits the gates in these layers act.

    :param pauli_layer: Description of which Pauli gate should act on each qubits. This is an element of {0,1,2,3}^n
    with 0 -> I, 1->X, 2->Y, 3->Z.
    :param gamma: Describes on which qubits CX acts. In particular the circuit contains CX_{i,j} if
    gamma[i][j]=1. The gates are ordered such the control qubit index increases with time.
    :param delta: Describes on which qubits CZ acts. In particular the circuit contains CX_{i,j} if
    delta[i][j]=1. The gates are ordered such the control qubit index increases with time. The circuit include S_i
    if delta[i][i]=1.
    :return: A Hadamard free Clifford circuit.
    """

    circ = Circuit(len(pauli_layer))

    # Add layer of CX gates
    for j in range(len(delta)):
        for i in range(j):
            if delta[i][j]:
                circ.CX(i, j, opgroup="Clifford 2")

    # Add layer of CZ gates
    for j in range(len(gamma)):
        for i in range(j):
            if gamma[i][j]:
                circ.CZ(i, j, opgroup="Clifford 2")

    # Add layer of S gates
    for i in range(len(gamma)):
        if gamma[i][i]:
            circ.S(i, opgroup="Clifford 1")

    # Add Pauli gate
    for i, gate in enumerate(pauli_layer):
        if gate == 0:
            circ.X(i, opgroup="Clifford 1")
        elif gate == 1:
            circ.Y(i, opgroup="Clifford 1")
        elif gate == 2:
            circ.Z(i, opgroup="Clifford 1")

    return circ


def find_random_gamma_delta(
    n_qubits: int, hadamard_layer: List[int], permute_layer: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates random gamma and delta matrices for a clifford gates in its canonical
    representation as introduced in https://arxiv.org/pdf/2003.09412.pdf. This
    required 2 each of gamma and delta matrices, which is sufficient to
    generate 2 hadamard free Clifford gates, as required by the canonical representation.
    This assumes the generation Clifford gates uniformly at random.

    :param n_qubits: Number of qubits on which the Clifford circuit acts
    :param hadamard_layer: vector in {0,1}^n_qubits describing the qubits on
    which hadamard gates act. Forms part of the canonical representation of a Clifford gate.
    :param permute_layer: Permutation of (0,1,...,n_qubits) describing the permutation
    layer of the canonical representation.
    :return: gamma and delta matrices for 2 Hadamard free Clifford circuits.
    """

    # Identity matrix
    Delta1 = np.identity(n_qubits)
    Delta2 = Delta1.copy()
    # All zero matrix
    Gamma1 = np.zeros((n_qubits, n_qubits))
    Gamma2 = Gamma1.copy()

    for i in range(n_qubits):
        Gamma2[i][i] = np.random.randint(2)
        if hadamard_layer[i]:
            Gamma1[i][i] = np.random.randint(2)

    for j in range(n_qubits):
        for i in range(j + 1, n_qubits):
            # Sample off diagonal elements of Gamma2 and Delta2 uniformly at random
            b = np.random.randint(2)
            Gamma2[i][j] = b
            Gamma2[j][i] = b
            Delta2[i][j] = np.random.randint(2)

            # Elements of Gamma1 are conditional on the hadamard and permutation layers
            if hadamard_layer[i] == 1 and hadamard_layer[j] == 1:
                b = np.random.randint(2)
                Gamma1[i][j] = b
                Gamma1[j][i] = b
            if (
                hadamard_layer[i] == 1
                and hadamard_layer[j] == 0
                and permute_layer[i] < permute_layer[j]
            ):
                b = np.random.randint(2)
                Gamma1[i][j] = b
                Gamma1[j][i] = b
            if (
                hadamard_layer[i] == 0
                and hadamard_layer[j] == 1
                and permute_layer[i] > permute_layer[j]
            ):
                b = np.random.randint(2)
                Gamma1[i][j] = b
                Gamma1[j][i] = b

            # Elements of Delta1 are conditional on the hadamard and permutation layers
            if hadamard_layer[i] == 0 and hadamard_layer[j] == 1:
                Delta1[i][j] = np.random.randint(2)
            if (
                hadamard_layer[i] == 1
                and hadamard_layer[j] == 1
                and permute_layer[i] > permute_layer[j]
            ):
                Delta1[i][j] = np.random.randint(2)
            if (
                hadamard_layer[i] == 0
                and hadamard_layer[j] == 0
                and permute_layer[i] < permute_layer[j]
            ):
                Delta1[i][j] = np.random.randint(2)

    return Delta1, Delta2, Gamma1, Gamma2


def random_clifford_circ(n_qubits: int, **kwargs) -> Circuit:
    """
    Samples an n qubit Clifford gate, in the form of a circuit, uniformly at random.
    This is adapted from https://arxiv.org/abs/2003.09412.
    In particular, the circuit has the form FHSF', where: F and and F' are hadamard
    free Clifford gates, H is a layer of
    Hadamard gates acting on a subset of the qubits, and S is a permutation layer.

    :param n_qubits: The number of qubits on which the returned random Clifford circuit should act
    :return: A random Clifford circuit acting on n_qubits qubits
    """

    np.random.seed(kwargs.get("seed", None))

    circ = Circuit(n_qubits)

    hadamard, permute = sample_q_mallows(n_qubits)
    Delta1, Delta2, Gamma1, Gamma2 = find_random_gamma_delta(
        n_qubits, hadamard, permute
    )

    # Append Clifford gate. Here the Pauli gates are assigned at random.
    cliff_circ = clifford_canonical_F(
        [np.random.randint(2) for _ in range(n_qubits)], Gamma2, Delta2
    )
    circ = circ.add_circuit(cliff_circ, [i for i in range(n_qubits)], [])

    # Implement swap layer based on S
    for i in range(len(permute)):
        while not (permute[i] == i):
            temp = permute[i]
            circ.SWAP(i, temp, opgroup="Clifford 2")
            permute[i] = permute[temp]
            permute[temp] = temp

    # Implement layer of hadamard gates acting on a subset of qubits
    for i in range(len(hadamard)):
        if hadamard[i]:
            circ.H(i, opgroup="Clifford 1")

    # Append second Clifford gate. Note that in this case the random pauli gate is the identity.
    cliff_circ = clifford_canonical_F([0 for _ in range(n_qubits)], Gamma1, Delta1)
    circ = circ.add_circuit(cliff_circ, [i for i in range(n_qubits)], [])

    return circ
