from qermit.spectral_filtering.spectral_filtering import gen_fft_task
import numpy as np
from pytket import Qubit
from pytket.pauli import QubitPauliString, Pauli
from pytket.utils import QubitPauliOperator
from scipy.fft import fftfreq
import math

def test_gen_fft_task():

    def generate_sine_wave(freq, sample_rate, duration):
        x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
        y = np.sin((2 * np.pi) * x * freq)
        return x, y
    
    SAMPLE_RATE = 20
    DURATION = 2
    FREQUENCY = 2
    _, sine_wave = generate_sine_wave(FREQUENCY, SAMPLE_RATE, DURATION)

    # Note that the QubitPauliString is being generated just to make the
    # input types match up. It's not used at any point.
    qps = QubitPauliString(
        [Qubit(0), Qubit(1)], [Pauli.Z, Pauli.Z]
    )
    result_list = [QubitPauliOperator({qps: coef}) for coef in sine_wave]
    result_grid = np.array(result_list)
    result_grid_list = [result_grid]

    fft_task = gen_fft_task()
    in_wires = (result_grid_list, )
    out_wires = fft_task(in_wires)

    N = SAMPLE_RATE * DURATION

    yf = out_wires[0][0]
    xf = fftfreq(N, 1 / SAMPLE_RATE)
    fft_dict = {x:y for x, y in zip(xf, yf)}

    # N division by two as amplitude is split accross positive and negative
    assert math.isclose(abs(fft_dict[FREQUENCY]), N/2)
    assert math.isclose(abs(fft_dict[-FREQUENCY]), N/2)