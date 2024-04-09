from pytket import Circuit, OpType, wasm
import itertools
from pytket.unit_id import BitRegister, QubitRegister
from pathlib import Path


def get_wfh():

    rus_dir = Path("/Users/daniel.mills/Qermit/qermit_rus/target/wasm32-unknown-unknown/release")
    wasm_file = rus_dir.joinpath('qermit_rus.wasm')
    return wasm.WasmFileHandler(wasm_file)


def gen_h_series_randomised_circuit(circuit):

    wfh = get_wfh()
    
    randomised_circuit = Circuit()
    randomisation_reg_dict = {}
    for q_register in circuit.q_registers:
        randomised_circuit.add_q_register(q_register)
        for qubit in q_register:
            randomisation_reg = randomised_circuit.add_c_register(f"randomisation_{qubit.to_list()}", 4)
            randomisation_reg_dict[qubit] = randomisation_reg
    for c_register in circuit.c_registers:
        randomised_circuit.add_c_register(c_register)

    seed_size = 16
    seed_c_reg = BitRegister(name='seed_c', size=seed_size)
    randomised_circuit.add_c_register(seed_c_reg)

    if circuit.n_qubits < seed_size:
        seed_q_reg = QubitRegister(
            name='seed_q',
            size=seed_size-sum(q_register.size for q_register in randomised_circuit.q_registers)
        )
        randomised_circuit.add_q_register(seed_q_reg)

    all_qubits = list(itertools.chain.from_iterable([q_register.to_list() for q_register in randomised_circuit.q_registers]))
    for qubit, cbit in zip(all_qubits[:seed_size], seed_c_reg.to_list()):
        randomised_circuit.H(qubit)
        randomised_circuit.Measure(qubit, cbit)
        randomised_circuit.Reset(qubit)
    randomised_circuit.add_wasm_to_reg("seed_randomisation", wfh, [seed_c_reg], [])

    for command in circuit:

        if command.op.type == OpType.ZZMax:
            randomised_circuit.add_wasm_to_reg("write_randomisation", wfh, [], [randomisation_reg_dict[command.args[0]]])
            randomised_circuit.Z(command.qubits[0], condition=randomisation_reg[0])
            randomised_circuit.Z(command.qubits[1], condition=randomisation_reg[1])
            randomised_circuit.X(command.qubits[0], condition=randomisation_reg[2])
            randomised_circuit.X(command.qubits[1], condition=randomisation_reg[3])

        randomised_circuit.add_gate(command.op, command.args)

        if command.op.type == OpType.ZZMax:
            randomised_circuit.Z(
                command.qubits[0],
                condition=randomisation_reg[0] ^ randomisation_reg[2] ^ randomisation_reg[3]
            )
            randomised_circuit.Z(
                command.qubits[1],
                condition=randomisation_reg[1] ^ randomisation_reg[2] ^ randomisation_reg[3]
            )
            randomised_circuit.X(command.qubits[0], condition=randomisation_reg[2])
            randomised_circuit.X(command.qubits[1], condition=randomisation_reg[3])

            randomised_circuit.Phase(0.5, condition=randomisation_reg[2])
            randomised_circuit.Phase(0.5, condition=randomisation_reg[3])
            randomised_circuit.Phase(-1, condition=randomisation_reg[2] & randomisation_reg[3])
    
    return randomised_circuit
