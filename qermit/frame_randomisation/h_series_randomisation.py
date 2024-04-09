from pytket import Circuit, OpType, wasm
import itertools
from pytket.unit_id import BitRegister, QubitRegister
from pathlib import Path


def get_wfh():

    rus_dir = Path("/Users/daniel.mills/Qermit/qermit_rus/target/wasm32-unknown-unknown/release")
    wasm_file = rus_dir.joinpath('qermit_rus.wasm')
    return wasm.WasmFileHandler(wasm_file)


def gen_randomised_circuit(circuit):

    wfh = get_wfh()
    
    randomised_circuit = Circuit()
    for q_register in circuit.q_registers:
        randomised_circuit.add_q_register(q_register)
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

    randomisation_reg = randomised_circuit.add_c_register("randomisation", 4)

    # rand_vals = [10, 7]
    # command_count = 0

    for command in circuit:

        if command.op.type == OpType.ZZMax:
            randomised_circuit.add_wasm_to_reg("write_randomisation", wfh, [], [randomisation_reg])
            # randomised_circuit.add_c_setreg(rand_vals[command_count], randomisation_reg)
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
            
            # randomised_circuit.Z(command.qubits[0], condition=randomisation_reg[4])
            # randomised_circuit.Z(command.qubits[1], condition=randomisation_reg[5])
            # randomised_circuit.X(command.qubits[0], condition=randomisation_reg[6])
            # randomised_circuit.X(command.qubits[1], condition=randomisation_reg[7])

            # randomised_circuit.Z(command.qubits[0], condition=randomisation_reg[0])
            # randomised_circuit.Z(command.qubits[0], condition=randomisation_reg[2])
            # randomised_circuit.Z(command.qubits[0], condition=randomisation_reg[3])
            # randomised_circuit.X(command.qubits[0], condition=randomisation_reg[2])

            # randomised_circuit.Z(command.qubits[1], condition=randomisation_reg[1])
            # randomised_circuit.Z(command.qubits[1], condition=randomisation_reg[2])
            # randomised_circuit.Z(command.qubits[1], condition=randomisation_reg[3])
            # randomised_circuit.X(command.qubits[1], condition=randomisation_reg[3])

            randomised_circuit.Phase(0.5, condition=randomisation_reg[2])
            randomised_circuit.Phase(0.5, condition=randomisation_reg[3])
            randomised_circuit.Phase(-1, condition=randomisation_reg[2] & randomisation_reg[3])

            # command_count += 1
    
    return randomised_circuit, wfh
