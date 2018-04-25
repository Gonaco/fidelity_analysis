qubits 3

.toffoli_gate
    { h q2  | tdag q0  }
    qwait 1
    cnot q2,q0 
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
    cnot q1,q0 
    qwait 3
    { h q0  | tdag q2  }
    qwait 1

