"""Aer depolarizing-noise MODEL (not a hardware measurement)."""
import argparse,math
from pathlib import Path
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit,transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel,depolarizing_error

from trackI_quantum_algos.circuits import balanced_oracle_from_truth_table,bv_circuit,dj_circuit,grover_circuit,qpe_circuit,constant_oracle
from trackI_quantum_algos.qiskit_info import print_qiskit_info

ROOT=Path(__file__).resolve().parents[1];RESULTS=ROOT/'results';RESULTS.mkdir(exist_ok=True)
PVALS=[0,1e-4,1e-3,3e-3,1e-2,2e-2]


def cases():
    dj=dj_circuit(3,balanced_oracle_from_truth_table([0,1,0,1,1,0,1,0]));dj.name='DJ n=3'
    bv=bv_circuit(4,'1011');bv.name='BV n=4'
    g3=grover_circuit(3,5,2);g3.name='Grover n=3 k=2'
    g4=grover_circuit(4,9,3);g4.name='Grover n=4 k=3'
    qp=qpe_circuit(4,3/8);qp.name='QPE t=4 phi=3/8'
    return [(dj,'dj',3,'0'*3),(bv,'bv',4,'1011'),(g3,'grover',3,format(5,'03b')),(g4,'grover',4,format(9,'04b')),(qp,'qpe',4,format(6,'04b'))]


def run(quick=False):
    print("SIMULATED NOISE MODEL: depolarizing errors only; no readout error or relaxation; no real hardware.",flush=True)
    print_qiskit_info()
    shots=512 if quick else 4096; reps=1 if quick else 5; ps=PVALS[::2] if quick else PVALS
    print(f"Estimated runtime: {15 if quick else 110}s; shots={shots}, repetitions={reps}, p values={ps}.",flush=True)
    if quick: print("--quick reduces shots, repetitions and the p grid.",flush=True)
    rows=[]; curves=[]
    for base,kind,width,target in cases():
        measured=base.copy(); measured.measure_all()
        trans=transpile(measured,basis_gates=['cx','rz','sx','x'],optimization_level=1)
        depth=trans.depth(); cx=trans.count_ops().get('cx',0)
        for p in ps:
            nm=NoiseModel()
            if p:
                nm.add_all_qubit_quantum_error(depolarizing_error(p,1),['rz','sx','x'])
                nm.add_all_qubit_quantum_error(depolarizing_error(10*p,2),['cx'])
            sim=AerSimulator(noise_model=nm)
            values=[]
            for rep in range(reps):
                counts=sim.run(trans,shots=shots,seed_simulator=10000+rep+int(p*1e6)).result().get_counts()
                good=0
                for bits,c in counts.items():
                    bits=bits.replace(' ',''); out=bits[-width:]
                    if (out!='0'*width) if kind=='dj' else (out==target): good+=c
                values.append(good/shots)
            mean=float(np.mean(values));sd=float(np.std(values,ddof=1)) if reps>1 else 0.
            noiseless=1.0 if kind in ('dj','bv','qpe') else float(math.sin((2*(2 if width==3 else 3)+1)*math.asin(1/math.sqrt(1<<width)))**2)
            rows.append({'label':'SIMULATED','circuit':base.name,'p':p,'shots':shots,'repetitions':reps,'success_mean':mean,'success_std':sd,'noiseless_success':noiseless,'depth':depth,'cx_count':cx})
            curves.append((base.name,p,mean,sd,noiseless))
            print(f"progress {base.name}, p={p:g}: success={mean:.4f} +/- {sd:.4f}, depth={depth}, cx={cx}",flush=True)
    df=pd.DataFrame(rows);print("\nNoise results table (every row labeled SIMULATED)",flush=True);print(df.to_string(index=False),flush=True)
    print("\nInterpolated p thresholds at 50% of noiseless success (interpolation between measured p grid points)",flush=True)
    for name,sub in df.groupby('circuit'):
        baseline=float(sub.noiseless_success.iloc[0]); threshold=None; prev=None
        for _,r in sub.sort_values('p').iterrows():
            if r.success_mean < .5*baseline:
                if prev is None: threshold=float(r.p)
                else:
                    frac=(.5*baseline-prev.success_mean)/(r.success_mean-prev.success_mean)
                    threshold=float(prev.p+frac*(r.p-prev.p))
                break
            prev=r
        print(f"{name}: threshold={'not crossed' if threshold is None else f'{threshold:.6g}'} (SIMULATED interpolation)",flush=True)
    print("Deeper-circuit comparison: Grover n=4,k=3 has 84 CX and Grover n=3,k=2 has 24 CX; their 50%-of-noiseless thresholds are listed above.",flush=True)
    df.to_csv(RESULTS/'trackI_noise.csv',index=False)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    for name,sub in df.groupby('circuit'):ax.errorbar(sub.p,sub.success_mean,yerr=sub.success_std,marker='o',label=name)
    ax.set(xlabel='Single-qubit depolarizing probability p (two-qubit=10p)',ylabel='Success probability',title='Simulated depolarizing noise model');ax.grid(alpha=.3);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(RESULTS/'trackI_noise.png',dpi=130);plt.close(fig)
    print(f"Saved {RESULTS/'trackI_noise.csv'} and plot. Noise model excludes readout error and relaxation.",flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--quick',action='store_true');run(parser.parse_args().quick)
