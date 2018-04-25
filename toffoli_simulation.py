import qxelarator

qx = qxelarator.QX()

# set the required qasm to be executed on qx
qx.set("test_output/toffoli_gate.qasm")

N_exp = 1000

for i in range(N_exp):

    c0 = qx.get_measurement_outcome(0)
    c1 = qx.get_measurement_outcome(1)
    c2 = qx.get_measurement_outcome(2)

    qx.execute()                            # execute

    c0 = qx.get_measurement_outcome(0)
    c1 = qx.get_measurement_outcome(1)
    c2 = qx.get_measurement_outcome(2)

    print("Experiment {}".format("string", i))
    print("-\n")
    print("{} {} {}".format(c0, c1, c2))
