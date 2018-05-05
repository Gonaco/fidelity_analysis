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


# print("GPU is used:", sparsedm.using_gpu)

# create a circuit
c = Circuit(title="toffoli_gate")


# add qubits
c.add_qubit("q0", 10, 10)
c.add_qubit("q1", 10, 10)
c.add_qubit("q2", 10, 10)
c.add_qubit("q3", 10, 10)
c.add_qubit("q4", 10, 10)


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

sampler = uniform_noisy_sampler(readout_error=0.03, seed=42)
c.add_qubit("m2")
c.add_measurement("q2", time=28, output_bit="m2", sampler=sampler)

sampler = uniform_noisy_sampler(readout_error=0.03, seed=42)
c.add_qubit("m0")
c.add_measurement("q0", time=31, output_bit="m0", sampler=sampler)

sampler = uniform_noisy_sampler(readout_error=0.03, seed=42)
c.add_qubit("m1")
c.add_measurement("q1", time=31, output_bit="m1", sampler=sampler)

# sdm = sparsedm(c.get_qubit_names())


print("Bell state fidelity: ", np.dot(
    c.full_dm.dm.ravel(), c.full_dm.dm.ravel()))
