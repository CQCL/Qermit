from qermit.spectral_filtering.spectral_filtering import (
    gen_fft_task,
    gen_result_extraction_task,
    gen_wire_copy_task,
    gen_inv_fft_task,
    gen_mitigation_task,
    gen_fft_task,
    gen_flatten_task,
    gen_reshape_task,
    gen_obs_exp_grid_gen_task,
    gen_param_grid_gen_task,
    gen_symbol_val_gen_task,
    gen_ndarray_to_dict_task,
    gen_spectral_filtering_MitEx,
)
from qermit.spectral_filtering import SmallCoefficientSignalFilter
import numpy as np
from pytket.utils import QubitPauliOperator
import math
from pytket import Circuit, Qubit
from pytket.pauli import QubitPauliString, Pauli
from qermit import AnsatzCircuit, SymbolsDict, ObservableExperiment, ObservableTracker
from sympy import Symbol
from qermit import TaskGraph
import scipy.fft
from pytket.extensions.qiskit import AerBackend


def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    y = np.sin((2 * np.pi) * x * freq)
    return x, y

n_shots = 16
n_sym_vals = 3

a = Symbol("alpha")
b = Symbol("beta")
c = Symbol("gamma")

sym_vals = np.linspace(0, 2, n_sym_vals, endpoint=False)

# ====== Experiment One ======

circ_one = Circuit(2)
circ_one.H(0).H(1).Rz(a, 0).Rz(b, 1).H(0).H(1)
sym_dict_one = SymbolsDict().symbols_from_dict({a:1.01, b:1})
ansatz_circuit_one = AnsatzCircuit(circ_one, n_shots, sym_dict_one)

qps_one = QubitPauliString(
    [Qubit(0), Qubit(1)], [Pauli.Z, Pauli.Z]
)
qpo_one = QubitPauliOperator({qps_one: 1.0})
obs_track_one = ObservableTracker(qpo_one)

exp_one = ObservableExperiment(
        ansatz_circuit_one,
        obs_track_one,
    )

sym_vals_one = [sym_vals for _ in sym_dict_one._symbolic_map.keys()]

# ====== Experiment Two ======

circ_two = Circuit(3)
circ_two.H(0).H(2).Rx(a, 0).Rz(b, 1).Rx(c, 2).H(0).H(2)
sym_dict_two = SymbolsDict().symbols_from_dict({a:1, b:1, c:2/3})
ansatz_circuit_two = AnsatzCircuit(circ_two, n_shots, sym_dict_two)

qps_two = QubitPauliString(
    [Qubit(0), Qubit(2)], [Pauli.Z, Pauli.X]
)
qpo_two = QubitPauliOperator(
    {
        qps_one: 0.5,
        qps_two: 0.5
    }
)
obs_track_two = ObservableTracker(qpo_two) 

exp_two = ObservableExperiment(
        ansatz_circuit_two,
        obs_track_two,
    )

sym_vals_two = [sym_vals for _ in sym_dict_two._symbolic_map.keys()]

def test_gen_symbol_val_gen_task():

    param_grid_gen_task = gen_symbol_val_gen_task(n_sym_vals=n_sym_vals)

    obs_exp_list = [exp_one, exp_two]

    in_wire = (obs_exp_list, )
    out_wire = param_grid_gen_task(in_wire)

    assert out_wire[0] == in_wire[0]
    assert len(out_wire[1]) == len(exp_two)

    assert len(out_wire[1][0]) == 2 # Check that all symbols in the circuit, of which there are two, has a list
    assert len(out_wire[1][1]) == 3 # Check that all symbols in the circuit, of which there are three, has a list

    assert (out_wire[1][0][0] == sym_vals).all()
    assert (out_wire[1][0][1] == sym_vals).all()

    assert (out_wire[1][1][0] == sym_vals).all()
    assert (out_wire[1][1][1] == sym_vals).all()
    assert (out_wire[1][1][2] == sym_vals).all()

def test_gen_wire_copy_task():

    # A couple of tests that the task works with unfamiliar inputs.
    wire_copy_task = gen_wire_copy_task(n_in_wires=2, n_wire_copies=2)

    in_wires = (circ_one, circ_two)
    out_wire = wire_copy_task(in_wires)

    assert out_wire[0] == circ_one
    assert out_wire[2] == circ_one

    assert out_wire[1] == circ_two
    assert out_wire[3] == circ_two

    wire_copy_task = gen_wire_copy_task(n_in_wires=2, n_wire_copies=3)

    in_wires = (exp_one, circ_one)
    out_wire = wire_copy_task(in_wires)

    assert out_wire[0] == exp_one
    assert out_wire[2] == exp_one
    assert out_wire[4] == exp_one

    assert out_wire[1] == circ_one
    assert out_wire[3] == circ_one
    assert out_wire[5] == circ_one

    # A test in the same contex in which the task is used.
    wire_copy_task = gen_wire_copy_task(n_in_wires=2, n_wire_copies=2)

    in_wires = ([exp_one, exp_two], [sym_vals_one, sym_vals_two])
    out_wire = wire_copy_task(in_wires)

    assert out_wire[0] == [exp_one, exp_two]
    assert out_wire[2] == [exp_one, exp_two]

    assert out_wire[1] == [sym_vals_one, sym_vals_two]
    assert out_wire[3] == [sym_vals_one, sym_vals_two]

def test_gen_param_grid_gen_task():

    param_grid_gen_task = gen_param_grid_gen_task()

    # Test in expected context
    in_wire = ([sym_vals_one, sym_vals_two],)
    out_wire = param_grid_gen_task(in_wire)

    assert out_wire[0][0][0].shape == (n_sym_vals,n_sym_vals)
    assert out_wire[0][0][1].shape == (n_sym_vals,n_sym_vals)

    assert out_wire[0][1][0].shape == (n_sym_vals,n_sym_vals,n_sym_vals)
    assert out_wire[0][1][1].shape == (n_sym_vals,n_sym_vals,n_sym_vals)
    assert out_wire[0][1][2].shape == (n_sym_vals,n_sym_vals,n_sym_vals)

    in_wire = ([[[1,2,3], [4,5]]],)
    out_wire = param_grid_gen_task(in_wire)

    assert out_wire[0][0][0][0][0] == 1
    assert out_wire[0][0][0][2][0] == 3
    assert out_wire[0][0][1][1][1] == 5
    assert out_wire[0][0][1][1][0] == 4

def test_gen_obs_exp_grid_gen_task():

    obs_exp_grid_gen_task = gen_obs_exp_grid_gen_task()

    in_wire = ([exp_one, exp_two], [np.meshgrid([0.5, 1.5], [0.25, 0.75], indexing='ij'), np.meshgrid([0.5, 1.5], [0.25, 0.75], [1, 1.5], indexing='ij')], )
    out_wire = obs_exp_grid_gen_task(in_wire)

    assert out_wire[0][0][0][0].AnsatzCircuit.SymbolsDict._symbolic_map[a] == 0.5
    assert out_wire[0][0][0][0].AnsatzCircuit.SymbolsDict._symbolic_map[b] == 0.25

    assert out_wire[0][0][1][0].AnsatzCircuit.SymbolsDict._symbolic_map[a] == 1.5
    assert out_wire[0][0][1][0].AnsatzCircuit.SymbolsDict._symbolic_map[b] == 0.25

    assert out_wire[0][1][0][0][0].AnsatzCircuit.SymbolsDict._symbolic_map[a] == 0.5
    assert out_wire[0][1][0][0][0].AnsatzCircuit.SymbolsDict._symbolic_map[b] == 0.25
    assert out_wire[0][1][0][0][0].AnsatzCircuit.SymbolsDict._symbolic_map[c] == 1

    assert out_wire[0][1][1][0][1].AnsatzCircuit.SymbolsDict._symbolic_map[a] == 1.5
    assert out_wire[0][1][1][0][1].AnsatzCircuit.SymbolsDict._symbolic_map[b] == 0.25
    assert out_wire[0][1][1][0][1].AnsatzCircuit.SymbolsDict._symbolic_map[c] == 1.5

def test_gen_flatten_reshape_task():

    flatten_task = gen_flatten_task()
    reshape_task = gen_reshape_task()

    grid_0 = np.array([
        [
            [1,2],
            [3,4],
        ],[
            [-1,-2],
            [-3,-4],
        ]
    ])

    grid_0_flattened = [1,2,3,4,-1,-2,-3,-4]
    grid_0_shape = (2,2,2)

    grid_1 = np.array([
        [
            [1,2,3],
            [4,5,6],
            [7,8,9],
        ],[
            [-1,-2,-3],
            [-4,-5,-6],
            [-7,-8,-9],
        ],[
            [10,20,30],
            [40,50,60],
            [70,80,90],
        ]
    ])

    grid_1_flattened = [1,2,3,4,5,6,7,8,9,-1,-2,-3,-4,-5,-6,-7,-8,-9,10,20,30,40,50,60,70,80,90]
    grid_1_shape = (3,3,3)

    length_list = [8,27]

    grid_list = [grid_0, grid_1]
    in_wire = (grid_list, )
    out_wire = flatten_task(in_wire)

    assert len(out_wire[0]) == 35

    assert out_wire[1] == length_list
    assert out_wire[0][:length_list[0]] == grid_0_flattened
    assert out_wire[0][length_list[0]:length_list[0]+length_list[1]] == [1,2,3,4,5,6,7,8,9,-1,-2,-3,-4,-5,-6,-7,-8,-9,10,20,30,40,50,60,70,80,90]

    assert out_wire[2][0] == grid_0_shape
    assert out_wire[2][1] == grid_1_shape

    in_wire = (grid_0_flattened + grid_1_flattened, length_list, [grid_0_shape, grid_1_shape])
    out_wire = reshape_task(in_wire)

    assert (out_wire[0][0] == grid_0).all()
    assert (out_wire[0][1] == grid_1).all()

def test_gen_ndarray_to_dict_task():

    SAMPLE_RATE = 20
    DURATION = 2
    FREQUENCY = 2
    _, sine_wave = generate_sine_wave(FREQUENCY, SAMPLE_RATE, DURATION)

    result_list_one = [[[QubitPauliOperator({qps_one: coef * amp}) for coef in sine_wave] for amp in sine_wave] for _ in sine_wave]
    result_grid_one = np.array(result_list_one)
    result_dict_one = {
        qps_one:np.array([[[coef * amp for coef in sine_wave] for amp in sine_wave] for _ in sine_wave])
    }

    result_list_two = [[QubitPauliOperator({qps_one: coef * amp , qps_two: coef}) for coef in sine_wave] for amp in sine_wave]
    result_grid_two = np.array(result_list_two)
    result_dict_two = {
        qps_one: np.array([[coef * amp for coef in sine_wave] for amp in sine_wave]),
        qps_two: [[coef for coef in sine_wave] for amp in sine_wave]
    }

    result_grid_list = [result_grid_one, result_grid_two]

    ndarray_to_dict_task = gen_ndarray_to_dict_task()
    in_wires = (result_grid_list, )
    out_wires = ndarray_to_dict_task(in_wires)

    assert (out_wires[0][0][qps_one] == result_dict_one[qps_one]).all()
    assert (out_wires[0][1][qps_one] == result_dict_two[qps_one]).all()
    assert (out_wires[0][1][qps_two] == result_dict_two[qps_two]).all()

def test_gen_fft_task():

    SAMPLE_RATE = 20
    DURATION = 2
    FREQUENCY = 2
    _, sine_wave = generate_sine_wave(FREQUENCY, SAMPLE_RATE, DURATION)

    result_dict_one = {
        qps_one:np.array([[[coef * amp for coef in sine_wave] for amp in sine_wave] for _ in sine_wave])
    }

    result_dict_two = {
        qps_one: np.array([[coef * amp for coef in sine_wave] for amp in sine_wave]),
        qps_two: [[coef for coef in sine_wave] for amp in sine_wave]
    }

    result_dict_list = [result_dict_one, result_dict_two]

    fft_task = gen_fft_task()
    in_wires = (result_dict_list, )
    out_wires = fft_task(in_wires)

    N = SAMPLE_RATE * DURATION

    yf = out_wires[0]

    assert math.isclose(abs(yf[0][qps_one][0][4][4]), N**3/4)
    assert math.isclose(abs(yf[0][qps_one][0][-4][4]), N**3/4)
    assert math.isclose(abs(yf[0][qps_one][0][4][-4]), N**3/4)
    assert math.isclose(abs(yf[0][qps_one][0][-4][-4]), N**3/4)

    assert math.isclose(abs(yf[1][qps_one][4][4]), N**2/4)
    assert math.isclose(abs(yf[1][qps_one][-4][4]), N**2/4)
    assert math.isclose(abs(yf[1][qps_one][4][-4]), N**2/4)
    assert math.isclose(abs(yf[1][qps_one][-4][-4]), N**2/4)

    assert math.isclose(abs(yf[1][qps_two][0][4]), N**2/2)
    assert math.isclose(abs(yf[1][qps_two][0][-4]), N**2/2)

def test_gen_fft_task_with_sine():
    
    SAMPLE_RATE = 20
    DURATION = 2
    FREQUENCY = 2
    _, sine_wave = generate_sine_wave(FREQUENCY, SAMPLE_RATE, DURATION)

    # Note that the QubitPauliString is being generated just to make the
    # input types match up. It's not used at any point.
    qps = QubitPauliString(
        [Qubit(0), Qubit(1)], [Pauli.Z, Pauli.Z]
    )
    result_grid = {qps:np.array([coef for coef in sine_wave])}
    result_grid_list = [result_grid]

    fft_task = gen_fft_task()
    in_wires = (result_grid_list, )
    out_wires = fft_task(in_wires)

    N = SAMPLE_RATE * DURATION

    yf = out_wires[0][0][qps]
    xf = scipy.fft.fftfreq(N, 1 / SAMPLE_RATE)
    fft_dict = {x:y for x, y in zip(xf, yf)}

    # N division by two as amplitude is split accross positive and negative
    assert math.isclose(abs(fft_dict[FREQUENCY]), N/2)
    assert math.isclose(abs(fft_dict[-FREQUENCY]), N/2)

def test_gen_inv_fft_task():

    inv_fft_task = gen_inv_fft_task()

    SAMPLE_RATE = 20
    DURATION = 2
    FREQUENCY = 1

    _, sine_wave = generate_sine_wave(FREQUENCY, SAMPLE_RATE, DURATION)

    ideal_x_fft_3D = np.zeros((40, 40, 40), dtype=complex)
    ideal_x_fft_3D[0][0][2] = 0-32000j
    ideal_x_fft_3D[0][0][-2] = 0+32000j

    ideal_x_fft_2D_one = np.zeros((40, 40), dtype=complex)
    ideal_x_fft_2D_one[2][2] = -400
    ideal_x_fft_2D_one[2][-2] = 400
    ideal_x_fft_2D_one[-2][2] = 400
    ideal_x_fft_2D_one[-2][-2] = -400

    in_wire = ([{qps_one:ideal_x_fft_3D}, {qps_one:ideal_x_fft_2D_one, qps_two:ideal_x_fft_2D_one}], )
    out_wire = inv_fft_task(in_wire)

    x_ifft_3D = out_wire[0][0]
    x_ifft_2D = out_wire[0][1]

    assert np.allclose(np.absolute(x_ifft_3D[qps_one][0][0]), np.absolute(sine_wave))
    assert np.allclose(np.absolute(x_ifft_3D[qps_one][0][8]), np.absolute(sine_wave))
    assert np.allclose(np.absolute(x_ifft_3D[qps_one][8][0]), np.absolute(sine_wave))
    assert np.allclose(np.absolute(x_ifft_3D[qps_one][8][8]), np.absolute(sine_wave))

    assert np.allclose(np.absolute(x_ifft_2D[qps_one][0]), np.zeros((40)))
    assert np.allclose(np.absolute(x_ifft_2D[qps_one][5]), np.absolute(sine_wave))

    assert np.allclose(np.absolute(x_ifft_2D[qps_two][0]), np.zeros((40)))
    assert np.allclose(np.absolute(x_ifft_2D[qps_two][5]), np.absolute(sine_wave))

def test_gen_mitigation_task():

    tol=5
    signal_filter = SmallCoefficientSignalFilter(tol=tol)
    mitigation_task = gen_mitigation_task(signal_filter=signal_filter)

    grid_one = np.zeros((40,40,40))
    grid_one[0][0][0] = 10
    grid_one[0][0][1] = 3

    grid_one_ideal = np.zeros((40,40,40))
    grid_one_ideal[0][0][0] = 10

    grid_two = np.zeros((40,40))
    grid_two[0][0] = 10
    grid_two[0][1] = 3

    grid_two_ideal = np.zeros((40,40))
    grid_two_ideal[0][0] = 10

    in_wire = ([{qps_one:grid_one}, {qps_one:grid_two, qps_two:grid_two}], )
    out_wire = mitigation_task(in_wire)

    assert (out_wire[0][0][qps_one] == grid_one_ideal).all()
    assert (out_wire[0][1][qps_one] == grid_two_ideal).all()
    assert (out_wire[0][1][qps_two] == grid_two_ideal).all()

def test_gen_result_extraction_task():

    result_extraction_task = gen_result_extraction_task()

    result_grid_one = np.array([[i for i in range(3)] for _ in range(3)])
    result_grid_two = np.array([[[i+j for i in range(3)] for j in range(3)] for _ in range(3)])

    in_wire = (
        [
            {qps_one:result_grid_one},
            {qps_one:result_grid_two, qps_two:result_grid_two}
        ], 
        [exp_one, exp_two], 
        [sym_vals_one, sym_vals_two]
    )
    out_wire = result_extraction_task(in_wire)

    assert out_wire[0][0] == QubitPauliOperator({qps_one: 1.5})
    assert out_wire[0][1] == QubitPauliOperator({qps_one: 2.5, qps_two: 2.5})

def test_gen_spectral_filtering_MitEx():

    signal_filter = SmallCoefficientSignalFilter(tol=20)
    noisy_backend = AerBackend()
    n_vals=16

    experiment_taskgraph = gen_spectral_filtering_MitEx(
        backend=noisy_backend,
        n_vals=n_vals,
        signal_filter=signal_filter,
    )

    a = Symbol("alpha")
    b = Symbol("beta")

    circ = Circuit(2)
    circ.H(0).H(1).Rz(a, 0).Rz(b, 1).H(0).H(1)

    sym_dict = SymbolsDict().symbols_from_dict({a:1.01, b:1})

    qubit_pauli_string_one = QubitPauliString(
        [Qubit(0), Qubit(1)], [Pauli.Z, Pauli.Z]
    )
    qubit_pauli_string_two = QubitPauliString(
        [Qubit(0), Qubit(1)], [Pauli.X, Pauli.X]
    )
    ansatz_circuit = AnsatzCircuit(circ, 1000, sym_dict)
    obs_track = ObservableTracker(QubitPauliOperator({qubit_pauli_string_one: 0.5, qubit_pauli_string_two: 0.5}))

    exp = ObservableExperiment(
            ansatz_circuit,
            obs_track,
        )

    exp_list = [exp]

    out_wires = experiment_taskgraph.run(exp_list)

    assert list(out_wires[0]._dict.keys()) == [qubit_pauli_string_one, qubit_pauli_string_two]
    assert math.isclose(abs(out_wires[0]._dict[qubit_pauli_string_one]), 0.5, abs_tol=0.01)
    assert math.isclose(abs(out_wires[0]._dict[qubit_pauli_string_two]), 0, abs_tol=0.01)

def test_small_coefficient_signal_filter():

    tol=5
    signal_filter = SmallCoefficientSignalFilter(tol=tol)

    grid = np.array(
        [
            [
                [0,1,2],
                [3,4,5],
                [6,7,8]
            ],[
                [-1,-2,-3],
                [1,2,3],
                [1.1,2.2,5.5]
            ]
        ]
    )
    ideal_filtered_grid = np.array(
        [
            [
                [0,0,0],
                [0,0,5],
                [6,7,8]
            ],[
                [0,0,0],
                [0,0,0],
                [0,0,5.5]
            ]
        ]
    )
    filtered_grid = signal_filter.filter(grid)

    assert (filtered_grid == ideal_filtered_grid).all()