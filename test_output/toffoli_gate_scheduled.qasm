qubits 3

.toffoli_gate
    { prepz q0 | prepz q1 | prepz q2 }
    qwait 1
    { tdag q0 | tdag q1 | hadamard q2 }
    qwait 1
    cnot q2,q0
    qwait 3
    { t q0 | cnot q1,q2 }
    qwait 3
    { cnot q1,q0 | t q2 }
    qwait 3
    { tdag q0 | cnot q1,q2 }
    qwait 3
    cnot q2,q0
    qwait 3
    { t q0 | tdag q2 }
    qwait 1
    { cnot q1,q0 | h q2 }
    measure q2
    qwait 2
    { measure q0 | measure q1 }
    qwait 9

