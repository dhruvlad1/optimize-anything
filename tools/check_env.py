#!/usr/bin/env python3
"""check_env.py - Environment & dependencies verification script.

Verifies Python version, OS platform, imports of required optimization
and quantum computing libraries, and executes minimal sanity checks
for Qiskit Aer, PennyLane, OR-Tools (SCIP), and SciPy (HiGHS milp).
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


# Verified against OR-Tools version 9.15.6755
def test_ortools_scip() -> list[str]:
    print("=" * 60)
    print("Functional Test: OR-Tools Integer Program (SCIP Backend)")
    print("=" * 60)
    failures = []
    try:
        from ortools.linear_solver import pywraplp

        # Maximize 3x + 2y subject to x + y <= 4, x + 3y <= 6, x, y >= 0 integers
        solver = pywraplp.Solver.CreateSolver("SCIP")
        if not solver:
            raise RuntimeError("SCIP solver unavailable in OR-Tools")

        x = solver.IntVar(0, solver.infinity(), "x")
        y = solver.IntVar(0, solver.infinity(), "y")

        solver.Add(x + y <= 4)
        solver.Add(x + 3 * y <= 6)
        solver.Maximize(3 * x + 2 * y)

        status = solver.Solve()
        if status != pywraplp.Solver.OPTIMAL:
            raise RuntimeError(f"Solver did not find optimal solution (status {status})")

        val_obj = solver.Objective().Value()
        val_x = x.solution_value()
        val_y = y.solution_value()

        print("Problem   : maximize 3x + 2y s.t. x + y <= 4, x + 3y <= 6 (integers >= 0)")
        print(f"Solution  : x = {val_x:.1f}, y = {val_y:.1f}, Objective = {val_obj:.1f}")

        assert round(val_obj) == 12, f"Expected objective 12, got {val_obj}"
        print("[OK]   OR-Tools SCIP integer program solve passed.")
    except Exception as e:
        print(f"[FAIL] OR-Tools SCIP test failed: {e}")
        failures.append(f"OR-Tools SCIP check: {e}")

    print()
    return failures


# Verified against SciPy version 1.18.1
def test_scipy_milp() -> list[str]:
    print("=" * 60)
    print("Functional Test: SciPy Mixed-Integer LP (HiGHS)")
    print("=" * 60)
    failures = []
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp

        # Maximize 3x + 2y  <=>  Minimize -3x - 2y
        # Subject to:
        #   x + y <= 4  => -inf <= 1x + 1y <= 4
        #   x + 3y <= 6 => -inf <= 1x + 3y <= 6
        #   x, y >= 0, integers
        c = np.array([-3.0, -2.0])
        A = np.array([[1.0, 1.0], [1.0, 3.0]])
        constraints = LinearConstraint(A, -np.inf, [4.0, 6.0])
        integrality = np.array([1, 1])  # 1 indicates integer variable
        bounds = Bounds(lb=0.0, ub=np.inf)

        res = milp(c=c, integrality=integrality, constraints=constraints, bounds=bounds)
        if not res.success:
            raise RuntimeError(f"SciPy MILP failed to find optimal solution: {res.status}")

        val_obj = -res.fun
        val_x, val_y = res.x

        print("Problem   : maximize 3x + 2y s.t. x + y <= 4, x + 3y <= 6 (integers >= 0)")
        print(f"Solution  : x = {val_x:.1f}, y = {val_y:.1f}, Objective = {val_obj:.1f}")

        assert round(val_obj) == 12, f"Expected objective 12, got {val_obj}"
        print("[OK]   SciPy milp (HiGHS) solve passed.")
    except Exception as e:
        print(f"[FAIL] SciPy milp test failed: {e}")
        failures.append(f"SciPy milp check: {e}")

    print()
    return failures


def main() -> None:
    print_system_info()
    _, import_failures = check_packages()
    aer_failures = test_qiskit_aer()
    pennylane_failures = test_pennylane()
    ortools_failures = test_ortools_scip()
    scipy_failures = test_scipy_milp()

    all_failures = (
        import_failures
        + aer_failures
        + pennylane_failures
        + ortools_failures
        + scipy_failures
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
