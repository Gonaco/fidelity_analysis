import numpy as np

I = np.array([[1, 0],
              [0, 1]])

T = np.array([[1, 0],
              [0, np.sqrt(0+1j)]])

T_dag = np.array([[1, 0],
                  [0, np.sqrt(0-1j)]])

H = np.array([[1/(np.sqrt(2)), 1/(np.sqrt(2))],
              [1/(np.sqrt(2)), -1/(np.sqrt(2))]])

CNOT_2_0 = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
                     [],
                     [],
                     [],
                     [],
                     [],
                     [],
                     []])

CNOT_1_2 = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]])

CNOT_1_0 = np.array([[0, 1, 0, 0],
                     [1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

Toffoli = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 1, 0]])


# print(np.kron(T_dag, T_dag))
a = np.kron(H, np.kron(T_dag, T_dag))
# print(a)
b = np.kron(CNOT_1_2, T)
# print(b)
c = np.kron(T, CNOT_1_0)
d = np.kron(CNOT_1_2, T_dag)
e = np.kron(T_dag, np.kron(I, T))
f = np.kron(H, CNOT_1_0)

toffoli_approx = a * CNOT_2_0 * b * c * d * CNOT_2_0 * e * f
print(toffoli_approx)

if toffoli_approx == Toffoli:
    print("Succeed")
