# this file has been automatically generated by the OpenQL compiler please do not modify it manually.
qubits 3

.toffoli_gate
   prepz q0
   prepz q1
   prepz q2
   tdag q0
   tdag q1
   h q2
   ry90 q2
   cz q2,q0
   ry90 q0
   t q0
   ry90 q1
   cz q1,q2
   ry90 q2
   ry90 q1
   cz q1,q0
   ry90 q0
   t q2
   tdag q0
   ry90 q1
   cz q1,q2
   ry90 q2
   ry90 q2
   cz q2,q0
   ry90 q0
   t q0
   tdag q2
   ry90 q1
   cz q1,q0
   ry90 q0
   h q0
   measure q0
   measure q1
   measure q2
