import qxelarator
import numpy as np
import qutip as qt


qx = qxelarator.QX()

# set the required qasm to be executed on qx
qx.set("test_output/toffoli_gate.qasm")

# Ideal Toffoli gate
# qx.set("ideal_toffoli.qasm")

N_exp = 1000

d_hilbert_spc = 2 ** 3

mean = 0

rho = qt.fock(d_hilbert_spc, 0)

for i in range(N_exp):

    # state_n = i % d_hilbert_spc
    # rho = qt.fock(d_hilbert_spc, state_n)

    print("Experiment {}".format(i))
    print("-")
    qx.execute()                            # execute

    c0 = qx.get_measurement_outcome(0)
    c1 = qx.get_measurement_outcome(1)
    c2 = qx.get_measurement_outcome(2)

    print("{} {} {}\n".format(c2, c1, c0))

    print(qx.get_state())

    sigma_states = np.array([c2, c1, c0], dtype=int)
    state_n = int("".join(map(str, sigma_states)), 2)
    sigma = qt.fock(d_hilbert_spc, state_n)
    f = qt.metrics.fidelity(rho, sigma)
    print(f)

    if f-int(f) != float(0):
        print("Alert! not perpendicular state")
        quit()
    elif f == float(1):
        mean += f

mean = mean/N_exp
print(mean)
