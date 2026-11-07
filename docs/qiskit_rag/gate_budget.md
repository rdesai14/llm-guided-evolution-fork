# Reducing gate cost without breaking a Qiskit circuit

The size objective is the plain gate count: every gate in the circuit costs 1, one-
or two-qubit, including the 4 input-encoding RY gates (the seed has 28). Barriers and
measurements cost 0.
Ways to lower it while keeping the circuit valid:
- Use a linear chain instead of a ring (saves one two-qubit gate per layer).
- Entangle only the pairs that feed the readout qubits (0 and 1); gates outside their light cone
  cannot change the prediction.
- An RZ applied directly to |0> only adds a phase - remove it.
- Fewer layers (`N_LAYERS`) remove a full rotation + entangling block each.
- Merge rotations: RY(a) followed by RY(b) on the same qubit equals RY(a + b).
If you remove gates that consumed trainable parameters, also shrink the
`ParameterVector` size in the circuit-assembly function or keep offsets consistent,
so every weight in the returned list is actually used by the circuit.
