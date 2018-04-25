import numpy as np
import qutip as qt


# rho = np.array([1, 0, 0, 0, 0, 0, 0, 0])
# sigma = np.array([0, 0, 0, 0, 1, 0, 0, 0])

# a = np.outer(rho, sigma)
# print(a)
# print(a * rho)
# print(np.trace(a * rho))

N = 8
# rho = qt.fock(N, 0)

# for n in range(0, N-1):
#     sigma = qt.fock(N, n)
#     f = qt.metrics.fidelity(rho, sigma)
#     print(f)

sigma = qt.coherent(N, alpha=1.0)
print(sigma)

rho = qt.fock(N, 7)
print(rho)

f = qt.metrics.fidelity(rho, sigma)
print(f)

# Denstity matrix calculation
A = qt.states.ket2dm(rho)
print(A)
# Density matrix calculation
B = qt.states.ket2dm(sigma)
print(B)
# Fidelity
print(np.real((A * (B * A)).sqrtm().tr()))
