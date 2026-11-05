import itertools
import math
import numpy as np
from qiskit.quantum_info import Operator, Statevector

from trackI_quantum_algos.circuits import (
    balanced_oracle_from_truth_table, bv_circuit, classical_order,
    continued_fraction_order, constant_oracle, dj_circuit, factors_from_order,
    grover_circuit, inverse_qft_circuit, qft_circuit, qpe_circuit,
)
from trackI_quantum_algos.annealing import chain_lower_bound, ising_maxcut, sector_gap


def _input_probs(qc, n):
    p=Statevector.from_instruction(qc).probabilities()
    return np.array([sum(p[i] for i in range(len(p)) if i%(1<<n)==x) for x in range(1<<n)])


def test_dj_all_balanced_n3_and_constant():
    n=3
    assert _input_probs(dj_circuit(n,constant_oracle(0)),n)[0] > 1-1e-12
    assert _input_probs(dj_circuit(n,constant_oracle(1)),n)[0] > 1-1e-12
    for ones in itertools.combinations(range(1<<n), 1<<(n-1)):
        tab=[int(i in ones) for i in range(1<<n)]
        assert _input_probs(dj_circuit(n,balanced_oracle_from_truth_table(tab)),n)[0] < 1e-12


def test_bv_all_n4():
    for s in range(16): assert _input_probs(bv_circuit(4,s),4)[s] > 1-1e-12


def test_grover_analytic():
    for n,k in [(2,0),(3,2),(4,1),(5,3),(6,5)]:
        p=_input_probs(grover_circuit(n,0,k),n)[0]
        theta=math.asin(1/math.sqrt(1<<n))
        assert abs(p-math.sin((2*k+1)*theta)**2)<1e-9
    assert abs(_input_probs(grover_circuit(3,0,2),3)[0]-.9453125)<1e-9


def test_qft_matrix_and_inverse():
    for n in range(1,5):
        N=1<<n; F=np.exp(2j*np.pi*np.outer(np.arange(N),np.arange(N))/N)/math.sqrt(N)
        U=Operator(qft_circuit(n)).data
        assert np.max(np.abs(U-F))<1e-10
        assert np.max(np.abs(Operator(inverse_qft_circuit(n)).data-U.conj().T))<1e-10


def test_qpe_exact():
    p=Statevector.from_instruction(qpe_circuit(3,3/8)).probabilities().reshape(2,8).sum(axis=0)
    assert p[3]>1-1e-9


def test_shor_postprocessing():
    assert classical_order(7,15)==4
    assert continued_fraction_order(64,8,15,7)==4
    assert factors_from_order(7,4,15)==(3,5)


def test_dj_constants_n3():
    for v in (0,1): assert _input_probs(dj_circuit(3,constant_oracle(v)),3)[0]>1-1e-12


def test_ising_triangle_and_embedding_bounds():
    edges=[(0,1,1),(1,2,1),(0,2,1)]
    _,_,_,E,_=ising_maxcut(edges,3)
    assert np.min(E)==-2
    # From the stated coupler-count expression ceil((d-2)/4).
    assert [chain_lower_bound(d) for d in (2,6,10)]==[1,1,2]


def test_two_spin_sector_gap_analytic():
    # H_P=-Z0 Z1 gives energies [-1,+1,+1,-1]. In P=+ sector,
    # gap(s)=2*sqrt(s^2+4(1-s)^2), minimized at s=0.8.
    _,_,gap,smin,_=sector_gap(2,np.array([-1.,1.,1.,-1.]),points=1001)
    assert abs(smin-.8)<1e-12
    assert abs(gap-2*math.sqrt(.8))<1e-8
