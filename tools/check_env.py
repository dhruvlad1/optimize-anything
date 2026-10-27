#!/usr/bin/env python3
"""check_env.py - Environment & dependencies verification script.

Verifies Python version, OS platform, imports of required optimization
and quantum computing libraries, and executes minimal sanity checks
for Qiskit Aer, PennyLane, and PuLP.
"""

import importlib
import platform
import sys

PACKAGES = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("scikit-learn", "sklearn"),
    ("networkx", "networkx"),
    ("pulp", "pulp"),
    ("ortools", "ortools"),
    ("psutil", "psutil"),
    ("pytest", "pytest"),
    ("qiskit", "qiskit"),
    ("qiskit-aer", "qiskit_aer"),
    ("qiskit-algorithms", "qiskit_algorithms"),
    ("qiskit-machine-learning", "qiskit_machine_learning"),
    ("pennylane", "pennylane"),
]


def print_system_info() -> None:
    print("=" * 60)
    print("System Information")
    print("=" * 60)
    print(f"Python Version : {sys.version.split()[0]} ({sys.executable})")
    print(f"OS Platform    : {platform.platform()}")
    print(f"Architecture   : {platform.machine()}")
    print("=" * 60)
    print()


def check_packages() -> tuple[dict[str, str], list[str]]:
    print("=" * 60)
    print("Checking Package Imports")
    print("=" * 60)
    installed = {}
    failed = []

    for pkg_name, mod_name in PACKAGES:
        try:
            mod = importlib.import_module(mod_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"[OK]   {pkg_name:<25} version {version}")
            installed[mod_name] = version
        except ImportError as e:
            print(f"[FAIL] {pkg_name:<25} Import error: {e}")
            failed.append(f"Import failed: {pkg_name} ({e})")
        except Exception as e:
            print(f"[FAIL] {pkg_name:<25} Unexpected error on import: {e}")
            failed.append(f"Import failed: {pkg_name} ({e})")

    print()
    return installed, failed


def test_qiskit_aer() -> list[str]:
    print("=" * 60)
    print("Functional Test: Qiskit Aer Bell-State Circuit")
    print("=" * 60)
    failures = []
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        sim = AerSimulator()
        job = sim.run(qc, shots=1000)
        result = job.result()
        counts = result.get_counts()
        print(f"Circuit: 2-qubit Bell state (|00> + |11>) / sqrt(2)")
        print(f"Result counts (1000 shots): {counts}")
        print("[OK]   Qiskit Aer Bell-state simulation passed.")
    except Exception as e:
        print(f"[FAIL] Qiskit Aer Bell-state simulation failed: {e}")
        failures.append(f"Qiskit Aer Bell-state check: {e}")

    print()
    return failures


def test_pennylane() -> list[str]:
    print("=" * 60)
    print("Functional Test: PennyLane Simple Circuit")
    print("=" * 60)
    failures = []
    try:
        import pennylane as qml

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        val = circuit()
        print(f"Circuit: single-qubit Hadamard, <Z> expectation value")
        print(f"Expectation value: {float(val):.6f}")
        print("[OK]   PennyLane test passed.")
    except Exception as e:
        print(f"[FAIL] PennyLane test failed: {e}")
        failures.append(f"PennyLane check: {e}")

    print()
    return failures


def test_pulp_lp() -> list[str]:
    print("=" * 60)
    print("Functional Test: PuLP 2-Variable LP with CBC Solver")
    print("=" * 60)
    failures = []
    try:
        import pulp

        prob = pulp.LpProblem("Trivial_2Var_LP", pulp.LpMaximize)
        x = pulp.LpVariable("x", lowBound=0)
        y = pulp.LpVariable("y", lowBound=0)

        # Maximize x + 2y subject to x + y <= 4, x <= 2
        prob += x + 2 * y, "Objective"
        prob += x + y <= 4, "Constraint_1"
        prob += x <= 2, "Constraint_2"

        solver = pulp.PULP_CBC_CMD(msg=False)
        status = prob.solve(solver)
        status_label = pulp.LpStatus.get(status, f"Status_{status}")

        val_x = pulp.value(x)
        val_y = pulp.value(y)
        val_obj = pulp.value(prob.objective)

        print(f"LP Status : {status_label}")
        print(f"Solution  : x = {val_x}, y = {val_y}, Objective = {val_obj}")

        if status_label != "Optimal":
            raise RuntimeError(f"Expected Optimal status but got {status_label}")
        print("[OK]   PuLP LP solve passed.")
    except Exception as e:
        print(f"[FAIL] PuLP LP solve failed: {e}")
        failures.append(f"PuLP LP check: {e}")

    print()
    return failures


def main() -> None:
    print_system_info()
    _, import_failures = check_packages()
    aer_failures = test_qiskit_aer()
    pennylane_failures = test_pennylane()
    pulp_failures = test_pulp_lp()

    all_failures = (
        import_failures + aer_failures + pennylane_failures + pulp_failures
    )

    print("=" * 60)
    print("ENVIRONMENT CHECK SUMMARY")
    print("=" * 60)
    if not all_failures:
        print("RESULT: PASS")
        print("All required packages are installed and functional tests passed.")
        sys.exit(0)
    else:
        print("RESULT: FAIL")
        print(f"Total failed checks: {len(all_failures)}")
        for idx, failure in enumerate(all_failures, start=1):
            print(f"  {idx}. {failure}")
        sys.exit(1)


if __name__ == "__main__":
    main()
