from openql import openql as ql
import os
import numpy as np

curdir = os.path.dirname(__file__)
output_dir = os.path.join(curdir, 'test_output')
ql.set_option('output_dir', output_dir)
ql.set_option('optimize', 'no')
ql.set_option('scheduler', 'ASAP')
ql.set_option('log_level', 'LOG_WARNING')

# IDEAL CASE
# ----------
config_fn = os.path.join(
    curdir, 'config_cc_light_fidelity_analysis.json')


# quantumsim QASM
# ----------------
# config_fn = os.path.join(
#     curdir, 'config_quantumsim.json')


# SC-7 LIMITATIONS
# ----------------
# config_fn = os.path.join(
#     curdir, 'hardware_config_cc_light.json')


platform = ql.Platform('platform_none', config_fn)
sweep_points = [1, 2]
num_circuits = 1
num_qubits = 3
p = ql.Program('toffoli_gate', num_qubits, platform)
p.set_sweep_points(sweep_points, num_circuits)
k = ql.Kernel('toffoli_gate', platform)

k.gate('prepz', [0])
k.gate('prepz', [1])
k.gate('prepz', [2])

k.gate('tdag', [0])
k.gate('tdag', [1])
k.gate('h', [2])
k.gate('cnot', [2, 0])
k.gate('t', [0])
k.gate('cnot', [1, 2])
k.gate('cnot', [1, 0])
k.gate('t', [2])
k.gate('tdag', [0])
k.gate('cnot', [1, 2])
k.gate('cnot', [2, 0])
k.gate('t', [0])
k.gate('tdag', [2])
k.gate('cnot', [1, 0])
k.gate('h', [2])

k.gate('measure', [0])
k.gate('measure', [1])
k.gate('measure', [2])

k.gate('display', [0, 1, 2])

p.add_kernel(k)
p.compile()
