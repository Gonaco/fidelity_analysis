qubits 3

.ideal_toffoli

   prepz q0
   prepz q1
   prepz q2
   toffoli q0, q1, q2
   measure q0
   measure q1
   measure q2
