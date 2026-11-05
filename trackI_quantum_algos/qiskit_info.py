"""Runtime version/API disclosure for experiment entry points."""
import inspect
import qiskit
import qiskit_aer
from qiskit import transpile
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


def print_qiskit_info():
    print(f"Qiskit version: {qiskit.__version__}", flush=True)
    print(f"Qiskit Aer version: {qiskit_aer.__version__}", flush=True)
    for fn in (transpile, QuantumCircuit.mcx, QuantumCircuit.cp,
               QuantumCircuit.measure_all, QuantumCircuit.append, Gate.control,
               Statevector.from_instruction, AerSimulator,
               NoiseModel.add_all_qubit_quantum_error, depolarizing_error):
        print(f"Inspected API {fn.__qualname__}{inspect.signature(fn)}", flush=True)
