import os
import re

import numpy as np
import qxelarator


def analysis():

    success_registry = []
    fidelity_registry = []
    N_exp = 1000
    qasm_f_path = "test_output/toffoli_gate.qasm"

    expected_measurement, expected_q_state = qx_simulation(qasm_f_path, 3)

    # add_error_model(qasm_f_path, 0.01)

    for i in range(N_exp):

        measurement, q_state = qx_simulation(
            "test_output/toffoli_gate_error.qasm", 3)

        print(expected_measurement)
        print(measurement)

        succes_registry[i] = 1 if np.array_equal(
            measurement, expected_measurement) else 0

        fidelity_registry[i] = fidelity(expected_q_state, q_state)

        print(success_registry)
        print(fidelity_registry)
        print(probability_of_success(succes_registry, N_exp))


def output_quantum_state(q_state, N_qubits):
    # Defines the quantum state based on the output string of QX get_state() function

    m = re.search(
        r"\(([\+\-]\d[\.\d]*),([\+\-]\d[\.\de-]*)\) \|(\d+)>", q_state)
    amplitude = complex(float(m.group(1)), float(m.group(2)))

    base_state = np.zeros(2**N_qubits)
    base_state[int(m.group(3), 2)] = 1

    return amplitude*base_state


# def add_error_model(qasm_f_path, errprob):

#     error_model = "error_model depolarizing_channel, " + str(errprob)

#     add2qasm(qasm_f_path, "qubits ", error_model)


# def add2qasm(qasm_f_path, before, after):
#     # Look for some regular expression in a file (before) and add something new after it in a copy of this file

#     i = open(qasm_f_path, "r")
#     o = open("."+qasm_f_path+"~", "w")
#     for line in i.readlines():
#         if re.search(before, line):
#             o.write(line+"\n"+after)
#         else:
#             o.write(line)


def fidelity(expected, actual):
    # Fidelity calculation

    f = -1

    if expected.ndim > 1:
        # Super hard calculation.

        print("Expected quantum mixed state detected.")
        print("I'm not ready for this (T.T)")

    elif actual.ndim > 1:
        # Hard calculation

        f = np.sqrt(np.vdot(expected, np.dot(actual, expected)))

    else:
        # Simple calculation

        f = np.vdot(expected, actual)

    return f


def probability_of_success(success_registry, N_exp):

    # return sum(self.success_registry)/self.total_n_experiments #?
    return sum(success_registry)/N_exp


def qx_simulation(qasm_f_path, N_qubits):

    qx = qxelarator.QX()
    qx.set(qasm_f_path)

    # Check all possible states?
    # for k in range(2**self.total):

    qx.execute()                            # execute

    # Measure
    c_buff = []
    for q in range(N_qubits):
        c_buff.append(qx.get_measurement_outcome(q))

    measurement = np.array(c_buff[::-1], dtype=float)
    q_state = output_quantum_state(qx.get_state(), N_qubits)

    print(measurement)
    print(q_state)

    return q_state, measurement


analysis()
