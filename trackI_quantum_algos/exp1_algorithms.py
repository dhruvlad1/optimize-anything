"""Exact small-instance circuit demonstrations for standard quantum algorithms."""
import argparse, itertools, json, math, time
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import GroverOperator
from qiskit.quantum_info import Statevector, Operator

from trackI_quantum_algos.circuits import *
from trackI_quantum_algos.qiskit_info import print_qiskit_info

ROOT=Path(__file__).resolve().parents[1]; RESULTS=ROOT/'results'; RESULTS.mkdir(exist_ok=True)


def _marginal(qc,n):
    p=Statevector.from_instruction(qc).probabilities()
    return np.array([sum(p[x+(z<<n)] for z in range(1<<(qc.num_qubits-n))) for x in range(1<<n)])


def _truth_oracle(bits): return balanced_oracle_from_truth_table(bits)


def _table(title,rows):
    df=pd.DataFrame(rows)
    print(f"\n{title}",flush=True)
    if len(df): print(df.to_string(index=False),flush=True)
    return df


def run(quick=False):
    print_qiskit_info()
    estimate=12 if quick else 75
    print(f"Estimated runtime: {estimate}s ({'quick reduced sweep' if quick else 'full sweep'}).",flush=True)
    if quick: print("--quick reduces random DJ/BV samples, Grover sizes, and Shor cases.",flush=True)
    rng=np.random.default_rng(101); allrows=[]
    # Deutsch-Jozsa
    print("Progress: Deutsch-Jozsa functions n=2..6",flush=True)
    rows=[]
    for n in range(2,7):
        tables=[]
        for v in (0,1): tables.append(("constant",[v]*(1<<n)))
        if n<=3:
            for inds in itertools.combinations(range(1<<n),1<<(n-1)):
                tab=[int(i in inds) for i in range(1<<n)]; tables.append(("balanced",tab))
        else:
            for _ in range(20 if quick else 200):
                inds=rng.choice(1<<n,size=1<<(n-1),replace=False); tables.append(("balanced",[int(i in inds) for i in range(1<<n)]))
        for kind,tab in tables:
            oracle=constant_oracle(tab[0]) if kind=="constant" else _truth_oracle(tab)
            qc=dj_circuit(n,oracle); p0=float(_marginal(qc,n)[0])
            if (kind=="constant" and p0<1-1e-10) or (kind=="balanced" and p0>1e-10): raise AssertionError("Deutsch-Jozsa promise check failed")
            decomp=transpile(qc,basis_gates=['h','x','cx','p'],optimization_level=0)
            rows.append({"label":"SIMULATED","n":n,"function":kind,"p_zero":p0,"oracle_queries":1,
                         "classical_worst":"THEORETICAL: %d"%(2**(n-1)+1),"gate_count":sum(decomp.count_ops().values())})
    allrows+=rows; _table("Deutsch-Jozsa (every row labeled)",rows)
    # Bernstein-Vazirani
    print("Progress: Bernstein-Vazirani secrets n=2..8",flush=True)
    rows=[]
    for n in range(2,9):
        secrets=range(1<<n) if n<=5 else rng.choice(1<<n,size=20 if quick else 200,replace=True)
        for s in secrets:
            p=_marginal(bv_circuit(n,int(s)),n); success=float(p[int(s)])
            if success<1-1e-10: raise AssertionError("BV secret recovery failed")
            rows.append({"label":"SIMULATED","n":n,"secret":format(int(s),f'0{n}b'),"success_probability":success,
                         "quantum_queries":1,"classical_queries":"THEORETICAL: %d"%n})
    allrows+=rows; _table("Bernstein–Vazirani",rows)
    # Grover
    print("Progress: Grover exact success curves",flush=True)
    rows=[]; plot_rows=[]
    ns=range(2,8) if quick else range(2,11)
    for n in ns:
        N=1<<n; marked=int(rng.integers(N)); opt=math.floor(math.pi/4*math.sqrt(N));
        for k in range(0,2*opt+1):
            p=float(_marginal(grover_circuit(n,marked,k),n)[marked]); theta=math.asin(1/math.sqrt(N)); theory=math.sin((2*k+1)*theta)**2
            if abs(p-theory)>1e-9: raise AssertionError(f"Grover formula mismatch n={n},k={k}: {p} vs {theory}")
            rows.append({"label":"SIMULATED","n":n,"k":k,"marked":marked,"success_probability":p,
                         "theory":"THEORETICAL: %.12f"%theory,"optimal_k":"THEORETICAL: %d"%opt})
            plot_rows.append((n,k,p))
    if not quick:
        n=4; marked=6; oracle=QuantumCircuit(n); zeros=[q for q in range(n) if not (marked>>q)&1]
        for q in zeros: oracle.x(q)
        oracle.h(n-1); oracle.mcx(list(range(n-1)),n-1); oracle.h(n-1)
        for q in reversed(zeros): oracle.x(q)
        gq=QuantumCircuit(n); [gq.h(q) for q in range(n)]; gq.append(GroverOperator(oracle),list(range(n)))
        cross=float(_marginal(gq,n)[marked]); ours=float(_marginal(grover_circuit(n,marked,1),n)[marked])
        if abs(cross-ours)>1e-10: raise AssertionError("GroverOperator cross-check failed")
        print(f"GroverOperator cross-check n=4,k=1: ours={ours:.12f}, library={cross:.12f}",flush=True)
    multi=float(sum(_marginal(grover_circuit(4,[1,5,9],2),4)[i] for i in [1,5,9]))
    rows.append({"label":"SIMULATED","n":4,"k":2,"marked":3,"success_probability":multi,"theory":"three marked items"})
    allrows+=rows; _table("Grover search",rows)
    print("Grover overshoot check: success probabilities follow sin^2((2k+1)theta) and fall again past the first peak; the printed rows include k beyond each optimal_k.",flush=True)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    for n in sorted(set(n for n,_,_ in plot_rows)):
        pts=[(k,p) for nn,k,p in plot_rows if nn==n]; ax.plot([x for x,_ in pts],[y for _,y in pts],label=f"n={n}")
    ax.set(xlabel='Grover iterations k',ylabel='Success probability',ylim=(0,1.03));ax.grid(alpha=.3);ax.legend();fig.tight_layout();fig.savefig(RESULTS/'trackI_grover.png',dpi=120);plt.close(fig)
    # QFT
    print("Progress: QFT matrices and decomposed gate counts",flush=True)
    rows=[]
    for n in range(1,7):
        N=1<<n; F=np.exp(2j*np.pi*np.outer(np.arange(N),np.arange(N))/N)/math.sqrt(N); U=Operator(qft_circuit(n)).data
        err=float(np.max(abs(U-F))); ierr=float(np.max(abs(Operator(inverse_qft_circuit(n)).data-U.conj().T)))
        if err>1e-10 or ierr>1e-10: raise AssertionError(f"QFT mismatch n={n}: {err}, inverse {ierr}")
        tc=transpile(qft_circuit(n),basis_gates=['h','x','cx','p'],optimization_level=0)
        rows.append({"label":"SIMULATED","n":n,"max_abs_error":err,"inverse_error":ierr,"gates":sum(tc.count_ops().values()),
                     "hadamards":n,"controlled_phases":n*(n-1)//2,"final_swaps":n//2,
                     "classical_fft":"THEORETICAL: %.1f operations"%(N*n)})
    allrows+=rows; _table("QFT (little-endian integer basis, final swaps give standard DFT)",rows)
    print("QFT caveat: amplitudes cannot be read out in one run, so this is not a speedup for computing a classical Fourier transform.",flush=True)
    # QPE
    print("Progress: phase-estimation distributions",flush=True)
    rows=[]; qpe_dist=[]
    for t in range(3,9):
        for phi in (3/8,0.3):
            p=_marginal(qpe_circuit(t,phi),t); estimate_idx=int(np.argmax(p)); nearest=int(round(phi*(1<<t)))%(1<<t)
            if phi==3/8 and p[nearest]<1-1e-9: raise AssertionError("exact QPE did not resolve phase")
            dist=';'.join(f"{i:0{t}b}:{v:.4g}" for i,v in enumerate(p))
            rows.append({"label":"SIMULATED","t":t,"phi":phi,"distribution":dist,"most_likely":estimate_idx,
                         "nearest_probability":float(p[nearest]),"nearest_estimate":nearest/(1<<t),"absolute_error":abs(nearest/(1<<t)-phi)})
            if phi==0.3 and t==5:qpe_dist=[(i,float(v)) for i,v in enumerate(p)]
    allrows+=rows; _table("Quantum phase estimation distributions",rows)
    fig,ax=plt.subplots(figsize=(7,4));ax.bar([i for i,p in qpe_dist],[p for i,p in qpe_dist]);ax.set(xlabel='5-bit phase estimate',ylabel='Probability',title='QPE distribution, phi=0.3, t=5');fig.tight_layout();fig.savefig(RESULTS/'trackI_qpe.png',dpi=120);plt.close(fig)
    # Shor
    print("Progress: Shor matrix-oracle distributions",flush=True)
    rows=[]; shor_plot=None; cases=[(15,a) for a in (2,4,7,8,11,13)]+[(21,2),(21,5)]
    if quick: cases=cases[:2]
    for N,a in cases:
        t=2*math.ceil(math.log2(N)); start=time.perf_counter(); qc=shor_order_circuit(a,N,t); p=_marginal(qc,t); runtime=time.perf_counter()-start
        rtrue=classical_order(a,N); ps=0.;pf=0.;
        for y,prob in enumerate(p):
            r=continued_fraction_order(y,t,N,a)
            if r==rtrue: ps+=prob
            if r is not None and factors_from_order(a,r,N) is not None: pf+=prob
        trial=0
        for d in range(2,int(math.sqrt(N))+1):
            trial+=1
            if N%d==0: break
        rows.append({"label":"SIMULATED","N":N,"a":a,"qubits":t+math.ceil(math.log2(N)),"true_order":"THEORETICAL: %d"%rtrue,
                     "one_run_correct_order_probability":ps,"up_to_3_runs_factor_probability":1-(1-pf)**3,
                     "trial_division_checks":"MEASURED: %d"%max(1,trial),"bruteforce_order_steps":"MEASURED: %d"%rtrue,
                     "simulation_seconds":"MEASURED: %.6f"%runtime,
                     "exact_counting_distribution":json.dumps({str(y):round(float(v),12) for y,v in enumerate(p) if v>1e-12})})
        if N==15 and a==7: shor_plot=p
        print(f"Shor progress N={N}, a={a} complete ({runtime:.3f}s)",flush=True)
    allrows+=rows; _table("Shor order finding",rows)
    print("Shor caveat: this demonstrates order-finding logic on tiny numbers with matrix-based modular multiplication; it shows no speedup.",flush=True)
    if shor_plot is not None:
        fig,ax=plt.subplots(figsize=(7,4));ax.bar(range(len(shor_plot)),shor_plot);ax.set(xlabel='Counting-register measurement',ylabel='Probability',title='Shor measurement distribution: a=7, N=15');fig.tight_layout();fig.savefig(RESULTS/'trackI_shor.png',dpi=120);plt.close(fig)
    # A unified CSV with sparse columns keeps every table row auditable.
    csv=[]
    for x in allrows: csv.append(x)
    pd.DataFrame(csv).to_csv(RESULTS/'trackI_algorithms.csv',index=False)
    print(f"Saved {RESULTS/'trackI_algorithms.csv'} and algorithm plots.",flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--quick',action='store_true');run(parser.parse_args().quick)
