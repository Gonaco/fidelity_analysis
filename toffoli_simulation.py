import qxelarator
import numpy as np
import qutip as qt


qx = qxelarator.QX()

# set the required qasm to be executed on qx
qx.set("test_output/toffoli_gate.qasm")

N_exp = 1000

d_hilbert_spc = 2 ** 3

for i in range(N_exp):

    state_n = i % d_hilbert_spc
    rho = qt.fock(d_hilbert_spc, state_n)

    qx.execute()                            # execute

    c0 = qx.get_measurement_outcome(0)
    c1 = qx.get_measurement_outcome(1)
    c2 = qx.get_measurement_outcome(2)

    print("Experiment {}".format(i))
    print("-")
    print("{} {} {}\n".format(c0, c1, c2))

    sigma_states = np.array([c0, c1, c2], dtype=int)
    state_n = int("".join(map(str, sigma_states)), 2)
    sigma = qt.fock(d_hilbert_spc, state_n)
    f = qt.metrics.fidelity(rho, sigma)
    print(f)
