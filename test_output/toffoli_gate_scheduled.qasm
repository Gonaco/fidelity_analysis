qubits 3

.toffoli_gate
    { prepz q2  | prepz q0  }
    qwait 1
    { h q2  | tdag q0  }
    qwait 1
    { cnot q2,q0  | prepz q1  }
    qwait 1
    tdag q1 
    qwait 1
    cnot q1,q2 
    qwait 1
    t q0 
    qwait 1
    cnot q1,q0 
    qwait 1
    t q2 
    qwait 1
    cnot q1,q2 
    qwait 1
    tdag q0 
    qwait 1
    cnot q2,q0 
    qwait 3
    t q0 
    qwait 1
    { cnot q1,q0  | tdag q2  }
    qwait 1
    h q2 
    qwait 1
    { measure q2  | measure q1  | measure q0  }
    qwait 1

