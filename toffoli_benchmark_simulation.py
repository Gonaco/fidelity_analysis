import os
import re

import numpy as np
import qxelarator

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def analysis(N_qubits, all_states_matrix):

    success_registry = []
    fidelity_registry = []
    N_exp = 1000
    qasm_f_path = "test_output/toffoli_gate.qasm"

    expected_q_state, expected_measurement = qx_simulation(
        qasm_f_path, N_qubits)

    # add_error_model(qasm_f_path, 0.01)

    for i in range(N_exp):

        q_state, measurement = qx_simulation(
            "test_output/toffoli_gate_error.qasm", N_qubits)

        # print(expected_q_state)
        # print(q_state)

        print(expected_measurement)
        print(measurement)

        exp_m_int = int(''.join(str(int(e))
                                for e in expected_measurement.tolist()), 2)
        m_int = int(''.join(str(int(e)) for e in measurement.tolist()), 2)

        all_states_matrix[exp_m_int,
                          m_int] = all_states_matrix[exp_m_int, m_int] + 1/N_exp

        success_registry.append(1 if np.array_equal(
            measurement, expected_measurement) else 0)

        fidelity_registry.append(fidelity(expected_q_state, q_state))

        # if fidelity_registry[i] - success_registry[i] != 0:
        #     input("Fidelity and Success not equal")

    # print(probability_of_success(success_registry, N_exp))
    # print(fidelity_registry)
    # print(np.mean(fidelity_registry))

    return probability_of_success(success_registry, N_exp), all_states_matrix


def all_states_analysis(N_qubits):

    tomography_matrix = np.zeros((2**N_qubits, 2**N_qubits))

    for q in range(2**N_qubits):

        all_inpt_f(N_qubits, q)
        prob_succ, tomography_matrix = analysis(N_qubits, tomography_matrix)

    print(tomography_matrix)

    graph(N_qubits, tomography_matrix)


def graph(N_qubits, matrix):

    fig = plt.figure()
    ax = Axes3D(fig)
    # ax = fig.add_subplot(121, projection='3d')
    x = np.arange(2**N_qubits)
    y = np.arange(2**N_qubits)
    xpos, ypos = np.meshgrid(x, y)

    # axis = ['a','b','c','d','e']

    xpos = xpos.flatten()   # Convert positions to 1D array
    ypos = ypos.flatten()
    zpos = np.zeros(2**(2*N_qubits))

    dx = np.ones_like(zpos)
    dy = dx.copy()
    dz = matrix.flatten()

    print(xpos, ypos, zpos, dx, dy, dz)

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz)

    # #sh()
    # ax.w_xaxis.set_ticklabels(column_names)
    # ax.w_yaxis.set_ticklabels(row_names)
    # ax.set_xlabel('Letter')
    # ax.set_ylabel('Day')
    # ax.set_zlabel('Occurrence')

    # plt.show()
    plt.savefig('tomography_graph')


def output_quantum_state(q_state, N_qubits):
    # Defines the quantum state based on the output string of QX get_state() function

    m = re.search(
        r"\(([\+\-]\d[\.\de-]*),([\+\-]\d[\.\de-]*)\) \|(\d+)>", q_state)
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

def all_inpt_f(N_qubits, init_state):

    init_state_file = "test_output/toffoli_state.qst"

    with open(init_state_file, "w") as f:
        f.write("0.0 0.0 |000>\n 1.0 0.0 |"+format(init_state,
                                                   "0"+str(N_qubits)+"b")[::-1]+">")


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

        f = np.absolute(np.vdot(expected, actual))**2

    print(f)

    return np.around(f, decimals=5)


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
    print(qx.get_state())
    q_state = output_quantum_state(qx.get_state(), N_qubits)

    return q_state, measurement


all_states_analysis(3)
