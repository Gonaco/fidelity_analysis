import os
import re

import numpy as np
import qxelarator

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from palettable.mycarta import Cube1_20

# QUANTUMSIM ##################################################################

from quantumsim.circuit import Circuit
from quantumsim.circuit import uniform_noisy_sampler
from quantumsim.circuit import CNOT as cnot
from quantumsim.circuit import Hadamard as h
from quantumsim.circuit import RotateEuler as RotateEuler
from quantumsim.circuit import ResetGate as ResetGate
import quantumsim.sparsedm as sparsedm


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

        measurement = measurement[::-1]

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


def quantumsim_analysis(N_qubits, all_states_matrix, init_state):

    success_registry = []
    fidelity_registry = []
    N_exp = 1000
    qasm_f_path = "test_output/toffoli_gate.qasm"

    expected_q_state, expected_measurement = qx_simulation(
        qasm_f_path, N_qubits)

    # add_error_model(qasm_f_path, 0.01)

    # for i in range(N_exp):

    #     measurement = quantumsim_simulation(0.01, init_state)
    #     measurement = measurement[::-1]

    #     print("Expected Measurement:")
    #     print(expected_measurement)
    #     print("Actual Measurement:")
    #     print(measurement)

    #     exp_m_int = int(''.join(str(int(e))
    #                             for e in expected_measurement.tolist()), 2)
    #     m_int = int(''.join(str(int(e)) for e in measurement.tolist()), 2)

    #     all_states_matrix[exp_m_int,
    #                       m_int] = all_states_matrix[exp_m_int, m_int] + 1/N_exp

    #     success_registry.append(1 if np.array_equal(
    #         measurement, expected_measurement) else 0)

    # return probability_of_success(success_registry, N_exp), all_states_matrix

    return quantumsim_simulation(0.01, init_state, N_exp, expected_measurement, all_states_matrix)


def all_states_analysis(N_qubits):

    tomography_matrix = np.zeros((2**N_qubits, 2**N_qubits))

    for q in range(2**N_qubits):

        init_state = all_inpt_f(N_qubits, q)
        # prob_succ, tomography_matrix = analysis(N_qubits, tomography_matrix)
        prob_succ, tomography_matrix = quantumsim_analysis(
            N_qubits, tomography_matrix, init_state)

    print(tomography_matrix)

    graph(N_qubits, tomography_matrix)


def graph(N_qubits, matrix):

    fig = plt.figure(figsize=(7, 7))

    # First graph (3D histogram)
    # ax = Axes3D(fig)
    ax = fig.add_subplot(111, projection='3d')

    # Tableau Colors
    # ax.set_color_cycle(Tableau_20.mpl_colors)

    # Background color
    ax.set_facecolor("white")

    # Set perspective
    ax.view_init(35, -45)

    x = np.arange(2**N_qubits)
    y = np.arange(2**N_qubits)
    xpos, ypos = np.meshgrid(x, y)

    axis = [format(i, "0"+str(N_qubits)+"b") for i in range(2**N_qubits)]

    xpos = xpos.flatten()   # Convert positions to 1D array
    ypos = ypos.flatten()
    zpos = np.zeros(2**(2*N_qubits))

    dx = 0.75 * np.ones_like(zpos)
    dy = dx.copy()
    dz = matrix.flatten()

    ratio = int(20/(2**N_qubits))
    end = 2**N_qubits * ratio
    # cs = Tableau_20.mpl_colors[:8] * 2**N_qubits
    cs_y = Cube1_20.mpl_colors[:end:ratio] * 2**N_qubits
    order = [i for i in range(2**N_qubits)] * 2**N_qubits
    cs_x = [x for _, x in sorted(zip(order, cs_y))]

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz,
             color=cs_x, shade=False, edgecolor="k")

    # ax.bar3d(xpos, ypos, zpos, dx, dy, dz,
    #          cmap=Cube1_20.mpl_colormap, edgecolor='b')

    # sh()
    ax.w_xaxis.set_ticklabels(axis)
    ax.w_yaxis.set_ticklabels(axis)

    # Ensure that the axis ticks only show up on the bottom and left of the plot.
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.
    ax.get_xaxis().tick_bottom()
    # ax.get_yaxis().tick_left()

    ax.set_xlabel("Actual Results")
    ax.set_ylabel("Expected Results (Correct)")
    ax.set_zlabel("Prob. Success")

    fig.tight_layout()

    plt.savefig("tomography_graph")

    # Second plot. Heatmap

    fig2 = plt.figure(figsize=(7, 7))
    ax2 = fig2.add_subplot(111)

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    im = ax2.imshow(matrix, cmap="jet")
    # im = ax2.imshow(matrix, cmap=Cube1_20.mpl_colormap)

    ax2.set_xticks(np.arange(2**N_qubits))
    ax2.set_yticks(np.arange(2**N_qubits))
    ax2.set_xticklabels(axis)
    ax2.set_yticklabels(axis)

    for i in range(2**N_qubits):
        for j in range(2**N_qubits):
            text = ax2.text(j, i, round(matrix[i, j], 2),
                            ha="center", va="center", color="w")

    ax2.set_xlabel("Expected Results (Correct)")
    ax2.set_ylabel("Actual Results")
    ax2.set_title("Prob. Success")

    plt.colorbar(im, cax=cax)

    # plt.show()
    fig2.tight_layout()

    plt.savefig("heatmap")


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
        f.write("0.0 0.0 |"+format(0, "0"+str(N_qubits)+"b")+">\n"+"1.0 0.0 |"+format(init_state,
                                                                                      "0"+str(N_qubits)+"b")[::-1]+">")

    return format(init_state, "0"+str(N_qubits)+"b")


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

# QUANTUMSIM ##################################################################


def quantumsim_simulation(error, init_state, N_exp, expected_measurement, all_states_matrix):

    # CIRCUIT DECLARATION
    c = toffoli_gate_decomposition_circuit(10, 10, error, init_state)

    # SIMULATING
    sdm = sparsedm.SparseDM(c.get_qubit_names())

    measurements = []

    # c.apply_to(sdm)
    # measurements = [sdm.classical["m0"],
    #                 sdm.classical["m1"], sdm.classical["m2"]]

    # return np.array(measurements, dtype=float)

    for i in range(N_exp):
        c.apply_to(sdm)
        measurement = np.array([sdm.classical["m2"],
                                sdm.classical["m1"], sdm.classical["m0"]])
        print("Expected Measurement:")
        print(expected_measurement)
        print("Actual Measurement:")
        print(measurement)

        exp_m_int = int(''.join(str(int(e))
                                for e in expected_measurement.tolist()), 2)
        m_int = int(''.join(str(int(e)) for e in measurement.tolist()), 2)

        all_states_matrix[exp_m_int,
                          m_int] = all_states_matrix[exp_m_int, m_int] + 1/N_exp

        success_registry.append(1 if np.array_equal(
            measurement, expected_measurement) else 0)

    return probability_of_success(success_registry, N_exp), all_states_matrix


def t(q, time):

    return RotateEuler(q, time=time, theta=0, phi=np.pi/4, lamda=0)


def tdag(q, time):

    return RotateEuler(q, time=time, theta=0, phi=-np.pi/4, lamda=0)


def toffoli_gate_decomposition_circuit(q_t1=np.inf, q_t2=np.inf, error=0.01, init_state="000"):
    # create a circuit
    c = Circuit(title="toffoli_gate")

    # add qubits
    c.add_qubit("q0", q_t1, q_t2)
    c.add_qubit("q1", q_t1, q_t2)
    c.add_qubit("q2", q_t1, q_t2)
    c.add_qubit("q3", q_t1, q_t2)
    c.add_qubit("q4", q_t1, q_t2)

    # add gates
    c.add_gate(ResetGate("q0", time=1, state=int(init_state[0])))
    c.add_gate(ResetGate("q1", time=1, state=int(init_state[1])))
    c.add_gate(ResetGate("q2", time=1, state=int(init_state[2])))

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

    sampler = uniform_noisy_sampler(readout_error=error, seed=42)
    c.add_qubit("m0")
    c.add_measurement("q0", time=31, output_bit="m0", sampler=sampler)

    sampler = uniform_noisy_sampler(readout_error=error, seed=42)
    c.add_qubit("m1")
    c.add_measurement("q1", time=31, output_bit="m1", sampler=sampler)

    sampler = uniform_noisy_sampler(readout_error=error, seed=42)
    c.add_qubit("m2")
    c.add_measurement("q2", time=28, output_bit="m2", sampler=sampler)

    return c


all_states_analysis(3)
