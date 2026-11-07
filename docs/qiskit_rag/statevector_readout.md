# Statevector readout

Qiskit's `Statevector` can simulate a bound circuit and return measurement
probabilities. The chosen qubit subset defines the marginal distribution; keep
the readout dimension consistent with the classification labels. This classifier
has 3 classes (Iris): `READOUT = [0, 1]` gives 4 outcomes, and
`readout_probabilities` keeps the first `N_CLASSES = 3` and renormalizes them (the
|11> outcome is dropped). Keep `N_CLASSES = 3` and return 3 probabilities that sum
to 1. This simulation is a local
statevector metric, not a hardware-noise estimate.

Reference: https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.Statevector
