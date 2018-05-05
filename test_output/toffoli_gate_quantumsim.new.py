import numpy as np
from quantumsim.circuit import Circuit
from quantumsim.circuit import uniform_noisy_sampler
from quantumsim.circuit import CNOT as cnot
from quantumsim.circuit import Hadamard as h
from quantumsim.circuit import RotateZ as RotateZ
import quantumsim.sparsedm as sparsedm


def t(q, time):

    return RotateZ(q, time=time, angle=np.pi/4)


def tdag(q, time):

    return RotateZ(q, time=time, angle=-np.pi/4)


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

    return c


c = toffoli_gate_decomposition_circuit(10, 10)
c_clean = toffoli_gate_decomposition_circuit()

# sampler = uniform_noisy_sampler(readout_error=0.03, seed=42)
# c.add_qubit("m2")
# c.add_measurement("q2", time=28, output_bit="m2", sampler=sampler)

# sampler = uniform_noisy_sampler(readout_error=0.03, seed=42)
# c.add_qubit("m0")
# c.add_measurement("q0", time=31, output_bit="m0", sampler=sampler)

# sampler = uniform_noisy_sampler(readout_error=0.03, seed=42)
# c.add_qubit("m1")
# c.add_measurement("q1", time=31, output_bit="m1", sampler=sampler)

# sdm = sparsedm(c.get_qubit_names())


# print("GPU is used:", sparsedm.using_gpu)

state_clean = sparsedm.SparseDM(c_clean.get_qubit_names())
c_clean.apply_to(state_clean)

state_decay = quantumsim.sparsedm.SparseDM(c.get_qubit_names())
c.apply_to(state_decay)

print("Bell state fidelity: ", np.dot(
    state_decay.full_dm.dm.ravel(), state_clean.full_dm.dm.ravel()))
