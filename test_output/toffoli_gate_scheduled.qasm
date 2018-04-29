qubits 3

.toffoli_gate
    prepz q2 
    qwait 1
    { h q2  | prepz q0  }
    qwait 1
    { ry90 q2  | tdag q0  | prepz q1  }
    qwait 1
    { cz q2,q0  | tdag q1  }
    qwait 1
    ry90 q1 
    qwait 1
    cz q1,q2 
    qwait 1
    ry90 q0 
    qwait 1
    { ry90 q1  | t q0  }
    qwait 1
    cz q1,q0 
    qwait 1
    ry90 q2 
    qwait 1
    { ry90 q1  | t q2  }
    qwait 1
    cz q1,q2 
    qwait 3
    { ry90 q2  | ry90 q0  }
    qwait 1
    { ry90 q2  | tdag q0  }
    qwait 1
    cz q2,q0 
    qwait 3
    ry90 q0 
    qwait 1
    { ry90 q1  | t q0  }
    qwait 1
    cz q1,q0 
    qwait 3
    ry90 q0 
    qwait 1
    { h q0  | tdag q2  }
    qwait 1
    { measure q2  | measure q1  | measure q0  }
    qwait 1

