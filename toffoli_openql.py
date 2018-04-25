from openql import openql as ql
import os
import numpy as np

curdir = os.path.dirname(__file__)
output_dir = os.path.join(curdir, 'test_output')
ql.set_output_dir(output_dir)
config_fn = os.path.join(
    curdir, '/home/daniel/Master/Quantum_Computing_and_Quantum_Information/tools/OpenQL/tests/hardware_config_cc_light.json')
platform = ql.Platform('platform_none', config_fn)
sweep_points = [1, 2]
num_circuits = 1
num_qubits = 3
p = ql.Program('toffoli_gate', num_qubits, platform)
p.set_sweep_points(sweep_points, num_circuits)
k = ql.Kernel('toffoli_gate', platform)

k.gate('tdag', 0)
k.gate('tdag', 1)
k.gate('h', 2)
k.gate('cnot', 2, 0)
k.gate('t', 0)
k.gate('cnot', 1, 2)
k.gate('cnot', 1, 0)
k.gate('t', 2)
k.gate('tdag', 0)
k.gate('cnot', 1, 2)
k.gate('cnot', 2, 0)
k.gate('t', 0)
k.gate('tdag', 2)
k.gate('cnot', 1, 0)
k.gate('h', 0)

p.add_kernel(k)
p.compile(optimize=False)
