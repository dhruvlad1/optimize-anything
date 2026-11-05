# Track I Design: Quantum Algorithms on Simulators

Every quantum result here is **SIMULATED** on a classical computer; no real quantum hardware is used. Statevector and dense/sparse Hamiltonian simulation cost grows exponentially with qubit count. Labels: **MEASURED** means a timing or count taken in this repo; **SIMULATED** means a quantum computation run on a classical simulator or noise model; **THEORETICAL** means a formula.

| Algorithm | Problem and circuit | Query/gate scale | Classical comparison |
|---|---|---|---|
| Deutsch–Jozsa | Decide whether a promised Boolean function is constant or balanced using phase kickback and Hadamards. | One oracle query. | Deterministic worst case $2^{n-1}+1$ queries. |
| Bernstein–Vazirani | Recover a hidden bit string with a phase-kickback oracle and Hadamards. | One oracle query. | $n$ classical queries. |
| Grover | Search an unstructured set by alternating a phase oracle and inversion about the mean. | About $(\pi/4)\sqrt N$ queries for one marked item. | $N/2$ queries on average. |
| QFT | Apply the discrete Fourier transform using Hadamards, controlled phases, and swaps. | $O(n^2)$ gates. | Classical FFT takes $O(N\log N)$ operations; amplitudes cannot be read out directly in one run, so this is not a way to compute a classical Fourier transform faster. |
| Quantum phase estimation | Estimate the phase $\phi$ of an eigenvalue using controlled powers and inverse QFT. | A register of $t$ qubits resolves phase to about $t$ bits. | Classical phase estimation needs other problem-specific methods. |
| Shor order finding | Estimate the phase of modular multiplication $a x\bmod N$, use continued fractions to infer the order, then classically post-process for factors. | Phase estimation uses controlled modular multiplications. | Trial division and brute-force order finding are used as tiny-instance classical baselines. |
| Quantum annealing | Interpolate $H(s)=-(1-s)\sum_i X_i+sH_P$. | Adiabatic runtime scales with the inverse square of the minimum relevant gap. | Classical exact search is feasible for the tiny instances studied. |

Annealing gaps are evaluated in the global-spin-flip-even sector because the Max-Cut Hamiltonian commutes with $P=\prod_i X_i$ and the initial state has parity +1. The full Hamiltonian's lowest gap closes at the endpoint due to the twofold spin-flip degeneracy; that is not the relevant gap here.

Annealing embedding counts are theoretical lower bounds derived from maximum coupler counts alone. They do not construct hardware embeddings and no annealing hardware is used. The Shor modular multiplication unitary is a matrix-based simulator oracle, not a gate-level decomposition.

## Quantum machine learning and search-cost comparisons

**Quantum kernel SVM.** The kernel is $k(x,x')=|\langle\phi(x)|\phi(x')\rangle|^2$. The two-repeat ZZ-style feature map applies Hadamards, $RZ(x_i)$ rotations, and $RZZ(x_ix_j)$ rotations for each feature pair. A classical SVM consumes the precomputed kernel. Shot estimates can be non-PSD; the implementation symmetrizes them, clips negative eigenvalues, and renormalizes the diagonal before SVM fitting.

**Variational classifier.** It uses the same feature map followed by $L$ layers of $RY$ and $RZ$ rotations with a CNOT ladder. Output is $\langle Z_0\rangle$, labels are $\pm1$, and the objective is mean squared error. Adam uses parameter-shift gradients; feature encoding values are fixed data, not trainable parameters.

**Fair comparison protocol.** Models share data splits and seeds. Hyperparameters are selected by five-fold cross-validation on training data only; the test set is evaluated once per final model. A parameter-matched classical MLP is included. Cost reporting includes training time, trainable parameters (or support-vector count for kernel methods), circuit evaluations and shots, and the number of kernel circuit pairs.

**Grover against linear search.** We compare oracle calls, transpiled gate counts, and wall-clock times. Classical statevector simulation still performs about $\sqrt{N}$ iterations over $N$ amplitudes; it cannot outperform a linear scan for searching classical data. A physical query speedup would require fault-tolerant hardware and an oracle cheaper than loading or reading the data.

All quantum results in these comparisons are **SIMULATED**. Tables use the labels **MEASURED**, **SIMULATED**, and **THEORETICAL** on every row.
