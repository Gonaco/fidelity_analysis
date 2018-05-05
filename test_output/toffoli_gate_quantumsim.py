# Quantumsim program generated OpenQL
# Please modify at your wil to obtain extra information from Quantumsim

import numpy as np
from quantumsim.circuit import Circuit
from quantumsim.circuit import uniform_noisy_sampler


# create a circuit
c = Circuit(title="toffoli_gate")


# add qubits
c.add_qubit("q0", 10, 10)
c.add_qubit("q1", 10, 10)
c.add_qubit("q2", 10, 10)
c.add_qubit("q3", 10, 10)
c.add_qubit("q4", 10, 10)

# add gates
c.add_prepz("q0", time=1)
c.add_prepz("q1", time=1)
c.add_prepz("q2", time=1)
c.add_tdag("q0", time=3)
c.add_tdag("q1", time=3)
c.add_hadamard("q2", time=3)
c.add_cnot("q2", "q0", time=5)
c.add_t("q0", time=9)
c.add_cnot("q1", "q2", time=9)
c.add_cnot("q1", "q0", time=13)
c.add_t("q2", time=13)
c.add_tdag("q0", time=17)
c.add_cnot("q1", "q2", time=17)
c.add_cnot("q2", "q0", time=21)
c.add_t("q0", time=25)
c.add_tdag("q2", time=25)
c.add_cnot("q1", "q0", time=27)
c.add_h("q2", time=27)

sampler = uniform_noisy_sampler(readout_error=0.03, seed=42)
c.add_qubit("m2")
c.add_measurement("q2", time=28, output_bit="m2", sampler=sampler)

sampler = uniform_noisy_sampler(readout_error=0.03, seed=42)
c.add_qubit("m0")
c.add_measurement("q0", time=31, output_bit="m0", sampler=sampler)

sampler = uniform_noisy_sampler(readout_error=0.03, seed=42)
c.add_qubit("m1")
c.add_measurement("q1", time=31, output_bit="m1", sampler=sampler)
