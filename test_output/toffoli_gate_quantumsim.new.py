import numpy as np
from quantumsim.circuit import Circuit
from quantumsim.circuit import uniform_noisy_sampler
from quantumsim.circuit import CNOT as cnot
from quantumsim.circuit import Hadamard as h
from quantumsim.circuit import RotateEuler as RotateEuler
from quantumsim.circuit import ResetGate as ResetGate
import quantumsim.sparsedm as sparsedm

# print("GPU is used:", sparsedm.using_gpu)


def t(q, time):

    return RotateEuler(q, time=time, theta=0, phi=np.pi/4, lamda=0)


def tdag(q, time):

    return RotateEuler(q, time=time, theta=0, phi=-np.pi/4, lamda=0)


def toffoli_gate_decomposition_circuit(q_t1=np.inf, q_t2=np.inf):
    # create a circuit
    c = Circuit(title="toffoli_gate")

    # add qubits
    c.add_qubit("q0", q_t1, q_t2)
    c.add_qubit("q1", q_t1, q_t2)
    c.add_qubit("q2", q_t1, q_t2)
    c.add_qubit("q3", q_t1, q_t2)
    c.add_qubit("q4", q_t1, q_t2)

    # add gates
    # c.add_gate(prepz("q0", time=1))
    # c.add_gate(prepz("q1", time=1))
    # c.add_gate(prepz("q2", time=1))

    c.add_gate(ResetGate("q0", time=1, state=0))
    c.add_gate(ResetGate("q1", time=1, state=0))
    c.add_gate(ResetGate("q2", time=1, state=0))

    c.add_gate(tdag("q0", time=3))
    c.add_gate(tdag("q1", time=3))
    c.add_gate(h("q2", time=3))
    c.add_gate(cnot("q2", "q0", time=5))
    c.add_gate(t("q0", time=9))
    c.add_gate(cnot("q1", "q2", time=9))
    c.add_gate(cnot("q1", "q0", time=13))
    c.add_gate(t("q2", time=13))
    c.add_gate(tdag("q0", time=17))
    c.add_gate(cnot("q1", "q2", time=17))
    c.add_gate(cnot("q2", "q0", time=21))
    c.add_gate(t("q0", time=25))
    c.add_gate(tdag("q2", time=25))
    c.add_gate(cnot("q1", "q0", time=27))
    c.add_gate(h("q2", time=27))

    sampler = uniform_noisy_sampler(readout_error=0.03, seed=42)
    c.add_qubit("m0")
    c.add_measurement("q0", time=31, output_bit="m0", sampler=sampler)

    sampler = uniform_noisy_sampler(readout_error=0.03, seed=42)
    c.add_qubit("m1")
    c.add_measurement("q1", time=31, output_bit="m1", sampler=sampler)

    sampler = uniform_noisy_sampler(readout_error=0.03, seed=42)
    c.add_qubit("m2")
    c.add_measurement("q2", time=28, output_bit="m2", sampler=sampler)

    return c


# CIRCUIT DECLARATION
c = toffoli_gate_decomposition_circuit(10, 10)
c_clean = toffoli_gate_decomposition_circuit()

# SIMULATING
sdm = sparsedm.SparseDM(c.get_qubit_names())

measurements = []

for i in range(1000):
    c.apply_to(sdm)
    measurements.append(
        [sdm.classical["m0"], sdm.classical["m1"], sdm.classical["m2"]])


print(measurements)


# FIDELITY CALCULATION (TO UNDERSTAND!!!!!!!!!)
state_clean = sparsedm.SparseDM(c_clean.get_qubit_names())
c_clean.apply_to(state_clean)

state_decay = sparsedm.SparseDM(c.get_qubit_names())
c.apply_to(state_decay)

print("Fidelity: ", np.dot(
    state_decay.full_dm.dm.ravel(), state_clean.full_dm.dm.ravel()))
