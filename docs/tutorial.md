---
file_format: mystnb
kernelspec:
  name: python3
---

# Tutorial

```{code-cell} ipython3
from pytket import Circuit
from pytket.circuit.display import render_circuit_jupyter

circ = Circuit(2,2).H(0).CX(0,1).measure_all()
render_circuit_jupyter(circ)
```

```{code-cell} ipython3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_state_probs(state):
    state_dict = {'State':[i for i in range(len(result_state))], 'Probability':abs(state)**2}
    state_df = pd.DataFrame(state_dict)
    sns.catplot(x='State', y='Probability', kind='bar', data=state_df, aspect = 3, height=2)
    plt.show()
    
def plot_counts(counts):
    counts_record = [{"State":str(state), "Count":count} for state, count in counts.items()]
    count_df = pd.DataFrame().from_records(counts_record)
    sns.catplot(x='State', y='Count', kind='bar', data=count_df, aspect = 3, height=2)
    plt.show()
```

```{code-cell} ipython3
import qiskit_aer.noise as noise

def depolarizing_noise_model(n_qubits, prob_1, prob_2, prob_ro):

    noise_model = noise.NoiseModel()

    error_2 = noise.depolarizing_error(prob_2, 2)
    for edge in [[i,j] for i in range(n_qubits) for j in range(i)]:
        noise_model.add_quantum_error(error_2, ['cx'], [edge[0], edge[1]])
        noise_model.add_quantum_error(error_2, ['cx'], [edge[1], edge[0]])

    error_1 = noise.depolarizing_error(prob_1, 1)
    for node in range(n_qubits):
        noise_model.add_quantum_error(error_1, ['h', 'rx', 'rz', 'u'], [node])
        
    probabilities = [[1-prob_ro, prob_ro],[prob_ro, 1-prob_ro]]
    error_ro = noise.ReadoutError(probabilities)
    for i in range(n_qubits):
        noise_model.add_readout_error(error_ro, [i])
        
    return noise_model
```

```{code-cell} ipython3
from qermit.taskgraph.mitex import gen_compiled_MitRes
from pytket.extensions.qiskit import AerBackend
from qermit import CircuitShots

n_shots = 100000
noisy_backend = AerBackend(
    depolarizing_noise_model(5, 0.001, 0.01, 0.05)
)
noisy_mitres = gen_compiled_MitRes(noisy_backend, optimisation_level=0)

circ_shots_list = [CircuitShots(circ, n_shots)]

noisy_result_list = noisy_mitres.run(circ_shots_list)
noisy_result_counts = noisy_result_list[0].get_counts()
plot_counts(noisy_result_counts)
```

```{code-cell} ipython3
noisy_mitres.get_task_graph()
```

```{code-cell} ipython3
from qermit.spam import gen_UnCorrelated_SPAM_MitRes

spam_mr = gen_UnCorrelated_SPAM_MitRes(noisy_backend, n_shots)
spam_result_list = spam_mr.run(circ_shots_list)
spam_result_counts = spam_result_list[0].get_counts()
plot_counts(spam_result_counts)
```

```{code-cell} ipython3
spam_mr.get_task_graph()
```

```{code-cell} ipython3
import numpy as np
from scipy.stats import unitary_group
from pytket.circuit import Unitary2qBox

def random_circ(n_qubits: int, depth: int, seed:int = None) -> Circuit:
    
    np.random.seed(seed)

    c = Circuit(n_qubits)

    for _ in range(depth):

        qubits = np.random.permutation([i for i in range(n_qubits)])
        qubit_pairs = [[qubits[i], qubits[i + 1]] for i in range(0, n_qubits - 1, 2)]

        for pair in qubit_pairs:

            # Generate random 4x4 unitary matrix.
            SU4 = unitary_group.rvs(4)  # random unitary in SU4
            SU4 = SU4 / (np.linalg.det(SU4) ** 0.25)
            SU4 = np.matrix(SU4)

            # Add gate corresponding to unitary.
            c.add_unitary2qbox(Unitary2qBox(SU4), *pair)

    return c
```

```{code-cell} ipython3
n_qubits = 4
rand_circ = random_circ(n_qubits,n_qubits,seed=23126)
render_circuit_jupyter(rand_circ)
```

```{code-cell} ipython3
from pytket.utils import QubitPauliOperator
from qermit import (
    AnsatzCircuit, SymbolsDict, 
    ObservableExperiment, ObservableTracker,
    MitEx
)
from pytket.pauli import Pauli, QubitPauliString
from pytket import Qubit

ideal_backend = AerBackend()
ideal_mitex = MitEx(ideal_backend)

qps = QubitPauliString(
    [Qubit(i) for i in range(n_qubits)], 
    [Pauli.Z for i in range(n_qubits)]
)

obs_exp = ObservableExperiment(
    AnsatzCircuit(rand_circ, n_shots, SymbolsDict()),
    ObservableTracker(QubitPauliOperator({qps:1}))
)
obs_exp_list = [obs_exp]

ideal_expectation = ideal_mitex.run(obs_exp_list)
print(f"Ideal expectation: {ideal_expectation[0]}")
```

```{code-cell} ipython3
noisy_mitex = MitEx(noisy_backend)

noisy_expectation = noisy_mitex.run(obs_exp_list)
print(f"Noisy expectation: {noisy_expectation[0]}")
```

```{code-cell} ipython3
from qermit.zero_noise_extrapolation import (
    gen_ZNE_MitEx, Fit, Folding
)

zne_me = gen_ZNE_MitEx(
    backend=noisy_backend, 
    noise_scaling_list=[9,7,5,3,1], 
    fit_type=Fit.exponential, 
    folding_type=Folding.circuit,
    show_fit=True,
)
```

```{code-cell} ipython3
import seaborn as sns 
sns.set_style("whitegrid")
```

```{code-cell} ipython3
zne_me.get_task_graph()
```

```{code-cell} ipython3
zne_me.run(obs_exp_list)
```

```{code-cell} ipython3
zne_spam_me = gen_ZNE_MitEx(
    backend=noisy_backend, 
    noise_scaling_list=[9,7,5,3,1], 
    fit_type=Fit.exponential, 
    show_fit=True,
    experiment_mitres=spam_mr,
)
```

```{code-cell} ipython3
zne_spam_me.run(obs_exp_list)
```

```{code-cell} ipython3
zne_spam_me.get_task_graph()
```