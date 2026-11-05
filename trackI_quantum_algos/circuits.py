"""Hand-written quantum circuits and classical post-processing for Track I.

Qiskit qubit q[0] is the least-significant bit of integer basis states. Bit
strings supplied to BV/Grover are written most-significant bit first.
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Callable, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import UnitaryGate


def _positive_int(v, name):
    if not isinstance(v, (int, np.integer)) or v < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(v)


def _mc_z(qc: QuantumCircuit, qubits: Sequence[int]):
    if len(qubits) == 1:
        qc.z(qubits[0]); return
    target = qubits[-1]
    qc.h(target)
    qc.mcx(list(qubits[:-1]), target)
    qc.h(target)


def constant_oracle(value: int) -> Callable[[QuantumCircuit, Sequence[int], int], None]:
    if value not in (0, 1): raise ValueError("constant value must be 0 or 1")
    def append(qc, inputs, ancilla):
        if value: qc.x(ancilla)
    return append


def balanced_oracle_from_truth_table(table: Sequence[int]):
    vals = tuple(int(v) for v in table)
    if not vals or any(v not in (0, 1) for v in vals) or len(vals) & (len(vals)-1):
        raise ValueError("truth table length must be a power of two and contain bits")
    n = int(math.log2(len(vals)))
    if sum(vals) != len(vals)//2: raise ValueError("truth table must be balanced")
    def append(qc, inputs, ancilla):
        if len(inputs) != n: raise ValueError("oracle input width mismatch")
        for x, bit in enumerate(vals):
            if not bit: continue
            zeros = [inputs[i] for i in range(n) if not (x >> i) & 1]
            for q in zeros: qc.x(q)
            qc.mcx(list(inputs), ancilla)
            for q in reversed(zeros): qc.x(q)
    return append


def dj_circuit(n: int, oracle) -> QuantumCircuit:
    n = _positive_int(n, "n")
    qc = QuantumCircuit(n+1, name="Deutsch-Jozsa")
    anc = n
    qc.x(anc); qc.h(anc)
    for q in range(n): qc.h(q)
    oracle(qc, list(range(n)), anc)
    for q in range(n): qc.h(q)
    return qc


def bv_circuit(n: int, secret: int | str | Sequence[int]) -> QuantumCircuit:
    n = _positive_int(n, "n")
    if isinstance(secret, str):
        if len(secret) != n or any(c not in "01" for c in secret): raise ValueError("secret must be an n-bit string")
        bits = [int(c) for c in reversed(secret)]
    elif isinstance(secret, (int, np.integer)):
        if secret < 0 or secret >= 1 << n: raise ValueError("secret integer does not fit n bits")
        bits = [(int(secret) >> i) & 1 for i in range(n)]
    else:
        bits = [int(b) for b in secret]
        if len(bits) != n or any(b not in (0, 1) for b in bits): raise ValueError("secret must contain n bits")
    qc = QuantumCircuit(n+1, name="Bernstein-Vazirani")
    anc=n; qc.x(anc); qc.h(anc)
    for q in range(n): qc.h(q)
    for q, bit in enumerate(bits):
        if bit: qc.cx(q, anc)
    for q in range(n): qc.h(q)
    return qc


def grover_circuit(n: int, marked: int | str | Sequence[int], k: int, measure_all: bool=False) -> QuantumCircuit:
    n = _positive_int(n, "n")
    if not isinstance(k, (int, np.integer)) or k < 0: raise ValueError("k must be a non-negative integer")
    if isinstance(marked, str):
        if len(marked) != n or any(c not in "01" for c in marked): raise ValueError("marked must be an n-bit string")
        marks=[int(marked,2)]
    elif isinstance(marked, (int,np.integer)): marks=[int(marked)]
    else: marks=[int(x) for x in marked]
    if not marks or any(m<0 or m>=1<<n for m in marks) or len(set(marks))!=len(marks): raise ValueError("invalid marked basis states")
    qc=QuantumCircuit(n, name="Grover")
    qs=list(range(n))
    for q in qs: qc.h(q)
    for _ in range(int(k)):
        for m in marks:
            zeros=[q for q in qs if not (m>>q)&1]
            for q in zeros: qc.x(q)
            _mc_z(qc,qs)
            for q in reversed(zeros): qc.x(q)
        for q in qs: qc.h(q)
        for q in qs: qc.x(q)
        _mc_z(qc,qs)
        for q in qs: qc.x(q)
        for q in qs: qc.h(q)
    if measure_all: qc.measure_all()
    return qc


def qft_circuit(n: int) -> QuantumCircuit:
    n=_positive_int(n,"n"); qc=QuantumCircuit(n,name="QFT")
    for j in reversed(range(n)):
        qc.h(j)
        for k in reversed(range(j)): qc.cp(math.pi/(2**(j-k)),j,k)
    for j in range(n//2): qc.swap(j,n-1-j)
    return qc


def inverse_qft_circuit(n: int) -> QuantumCircuit:
    return qft_circuit(n).inverse()


def qpe_circuit(t: int, phi: float) -> QuantumCircuit:
    t=_positive_int(t,"t")
    if not np.isfinite(phi): raise ValueError("phi must be finite")
    qc=QuantumCircuit(t+1,name="QPE"); eig=t
    qc.x(eig)
    for q in range(t):
        qc.h(q)
        qc.cp(2*math.pi*phi*(2**q),q,eig)
    qc.compose(inverse_qft_circuit(t),qubits=list(range(t)),inplace=True)
    return qc


def shor_order_circuit(a: int, N: int, t: int) -> QuantumCircuit:
    """Phase-estimation order-finding circuit.

    Modular multiplication is a MATRIX-based simulation via a permutation
    UnitaryGate, not a gate-level decomposition suitable for hardware.
    """
    if not isinstance(N,(int,np.integer)) or N<3: raise ValueError("N must be >= 3")
    if not isinstance(a,(int,np.integer)) or not 1<a<N or math.gcd(a,N)!=1: raise ValueError("a must satisfy 1<a<N and gcd(a,N)=1")
    t=_positive_int(t,"t"); m=math.ceil(math.log2(N)); qc=QuantumCircuit(t+m,name="Shor order finder")
    work=list(range(t,t+m)); qc.x(work[0])
    for q in range(t): qc.h(q)
    for j in range(t):
        mult=pow(int(a),1<<j,int(N)); dim=1<<m
        U=np.zeros((dim,dim),dtype=complex)
        for x in range(dim): U[(mult*x)%N if x<N else x,x]=1
        controlled=UnitaryGate(U,label=f"x*{mult} mod {N}").control(1)
        qc.append(controlled,[j]+work)
    qc.compose(inverse_qft_circuit(t),qubits=list(range(t)),inplace=True)
    return qc


def classical_order(a: int,N: int) -> int:
    if N<2 or math.gcd(a,N)!=1: raise ValueError("a and N must be coprime with N >= 2")
    x=1
    for r in range(1,N+1):
        x=(x*a)%N
        if x==1:return r
    raise ValueError("order not found")


def continued_fraction_order(measurement: int,t: int,N: int,a: int) -> int | None:
    if t<1 or measurement<0 or measurement>=1<<t: raise ValueError("invalid measurement or t")
    frac=Fraction(int(measurement),1<<int(t)).limit_denominator(N)
    den=frac.denominator
    # The reduced denominator may divide the order; test its multiples.
    for r in range(den, N+1, den):
        if pow(a,r,N)==1: return r
    return None


def factors_from_order(a: int,r: int,N: int) -> tuple[int,int] | None:
    if r<=0 or r%2 or pow(a,r,N)!=1:return None
    x=pow(a,r//2,N)
    if x in (1,N-1):return None
    p=math.gcd(x-1,N);q=math.gcd(x+1,N)
    if p in (1,N) or q in (1,N):return None
    return tuple(sorted((p,q)))
