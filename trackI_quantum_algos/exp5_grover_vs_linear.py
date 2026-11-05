"""Measured query/time costs of linear search versus a simulated Grover statevector."""
import argparse,math,time
from pathlib import Path
import numpy as np
import pandas as pd
from qiskit import transpile

from trackI_quantum_algos.circuits import grover_circuit
from trackI_quantum_algos.qiskit_info import print_qiskit_info

ROOT=Path(__file__).resolve().parents[1];RESULTS=ROOT/'results';RESULTS.mkdir(exist_ok=True)


class CountingOracle:
    def __init__(self,marked):self.marked=marked;self.calls=0
    def __call__(self,x):self.calls+=1;return x==self.marked


def linear_search(values,oracle):
    for i,value in enumerate(values):
        if oracle(value):return i
    return None


def linear_average(N):
    values=list(range(N));counts=[]
    for marked in range(N):
        o=CountingOracle(marked);found=linear_search(values,o)
        if found!=marked:raise AssertionError('linear search returned wrong item')
        counts.append(o.calls)
    avg=float(np.mean(counts))
    if avg!=(N+1)/2:raise AssertionError(f'linear average {avg} != {(N+1)/2}')
    return avg,max(counts)


def random_sample_search(N,marked,rng):
    oracle=CountingOracle(marked)
    while True:
        x=int(rng.integers(N))
        if oracle(x):return oracle.calls


def grover_numpy(n,marked,k):
    N=1<<n;a=np.full(N,1/math.sqrt(N),dtype=complex);oracle=CountingOracle(marked)
    for _ in range(k):
        # Counting phase-oracle query; phase flip then inversion about the mean.
        oracle.calls+=1;a[marked]*=-1;a=2*np.mean(a)-a
    p=float(abs(a[marked])**2)
    return p,oracle.calls


def _timed(fn,repeats=1):
    t=time.perf_counter()
    for _ in range(repeats):fn()
    return (time.perf_counter()-t)/repeats


def run(quick=False):
    print_qiskit_info()
    nmax_calls=12 if quick else 14;nmax_time=14 if quick else 20
    print(f"Estimated runtime: {40 if quick else 300}s; calls n=4..{nmax_calls}, timings n=4..{nmax_time}.",flush=True)
    if quick:print("--quick reduces call-count sizes to n<=12 and wall-clock timing sizes to n<=14; the formulas and gate table still run.",flush=True)
    rng=np.random.default_rng(5);call_rows=[]
    for n in range(4,nmax_calls+1):
        N=1<<n;avg,worst=linear_average(N);random_trials=20 if quick else 100
        rand=float(np.mean([random_sample_search(N,int(rng.integers(N)),rng) for _ in range(random_trials)]))
        marked=int(rng.integers(N));k=math.floor(math.pi/4*math.sqrt(N));p,calls=grover_numpy(n,marked,k)
        theta=math.asin(1/math.sqrt(N));theory=math.sin((2*k+1)*theta)**2
        if abs(p-theory)>1e-9:raise AssertionError(f'Grover probability mismatch n={n}: {p} != {theory}')
        expected=calls/p;ratio=avg/expected
        call_rows.append({'label':'MEASURED','n':n,'N':N,'linear_avg_calls':avg,'linear_worst_calls':worst,'random_sampling_mean_calls':rand,
            'grover_calls_per_run':calls,'grover_success_probability':p,'grover_expected_calls_with_restarts':expected,'classical_over_grover_measured':ratio,
            'classical_avg_formula':'THEORETICAL: (N+1)/2','classical_worst_formula':'THEORETICAL: N',
            'grover_queries_formula':'THEORETICAL: floor(pi/4*sqrt(N))','ideal_query_reduction':'THEORETICAL: sqrt(N)/(pi/4)'})
        print(f"progress oracle counts n={n}, N={N}: linear avg={avg:.1f}; Grover expected calls={expected:.3f}",flush=True)
    calls_df=pd.DataFrame(call_rows);print("\nOracle-call comparison",flush=True);print(calls_df.to_string(index=False),flush=True)

    gate_rows=[]
    for n in range(3,9):
        N=1<<n;k=math.floor(math.pi/4*math.sqrt(N));qc=grover_circuit(n,N//3,k)
        tq=transpile(qc,basis_gates=['cx','rz','sx','x'],optimization_level=1)
        cx=int(tq.count_ops().get('cx',0));depth=tq.depth()
        gate_rows.append({'label':'MEASURED','n':n,'N':N,'k':k,'depth':depth,'cx_total':cx,'cx_per_iteration':cx/k,
            'classical_avg_comparisons':'THEORETICAL: %.1f'%(N/2),'cx_over_classical_avg':cx/(N/2)})
    gate_df=pd.DataFrame(gate_rows);print("\nHand-written Grover transpiled gate counts",flush=True);print(gate_df.to_string(index=False),flush=True)

    time_rows=[];stop_n=None
    for n in range(4,nmax_time+1):
        N=1<<n;marked=N-1;values=list(range(N));bits=np.zeros(N,dtype=bool);bits[marked]=True
        pyreps=max(1,min(1000,2**max(0,14-n)));npreps=max(1,min(1000,2**max(0,14-n)))
        pytime=_timed(lambda:linear_search(values,CountingOracle(marked)),pyreps)
        nptime=_timed(lambda:int(np.argmax(bits)),npreps)
        k=math.floor(math.pi/4*math.sqrt(N));start=time.perf_counter();p,c=grover_numpy(n,marked,k);gtime=time.perf_counter()-start
        row={'label':'MEASURED','n':n,'N':N,'python_linear_seconds':pytime,'numpy_argmax_seconds':nptime,'numpy_statevector_grover_seconds':gtime,
             'python_over_grover_time':pytime/max(gtime,1e-15),'numpy_over_grover_time':nptime/max(gtime,1e-15),'grover_probability':p}
        time_rows.append(row);print(f"progress timing n={n}: Python {pytime:.4g}s, NumPy {nptime:.4g}s, statevector Grover {gtime:.4g}s",flush=True)
        if gtime>60:stop_n=n;print(f"Stopped wall-clock sweep at n={n}: one Grover simulation exceeded 60 seconds.",flush=True);break
    time_df=pd.DataFrame(time_rows);print("\nWall-clock timings (quantum is simulated)",flush=True);print(time_df.to_string(index=False),flush=True)
    # Parallel-computing contrast: print source rows, and pair each processor count p
    # with Grover's theoretical query reduction at the same N=2^p.
    pfile=RESULTS/'trackD_parallel.csv'
    if not pfile.exists():raise FileNotFoundError(pfile)
    parallel=pd.read_csv(pfile);print(f"Inspected {pfile.name} columns: {list(parallel.columns)}",flush=True)
    source=parallel[(parallel.experiment=='part_a_strong_scaling')&(parallel.metric=='speedup')&(parallel.type.astype(str).str.lower()=='measured')].copy()
    source['label']='MEASURED';source['grover_N']='THEORETICAL: 2^p';source['grover_query_reduction']=source.procs.map(lambda p:2**(p/2)/(math.pi/4) if float(p).is_integer() and p<=20 else np.nan);source['parallel_ceiling_p']='THEORETICAL: p'
    print("\nParallel speedups with source rows and the Grover query reduction evaluated at N=2^p",flush=True);print(source[['label','experiment','procs','metric','value','type','grover_N','grover_query_reduction','parallel_ceiling_p']].to_string(index=False),flush=True)
    out=pd.concat([calls_df.assign(section='oracle_calls'),gate_df.assign(section='gate_counts'),time_df.assign(section='wall_clock')],ignore_index=True,sort=False)
    out.to_csv(RESULTS/'trackI_grover_vs_linear.csv',index=False)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,5));ax.loglog(calls_df.N,calls_df.linear_avg_calls,'o-',label='linear measured');ax.loglog(calls_df.N,calls_df.grover_expected_calls_with_restarts,'o-',label='Grover simulated');Nvals=np.array(calls_df.N,dtype=float);ax.loglog(Nvals,calls_df.linear_avg_calls.iloc[0]*(Nvals/Nvals[0]),'--',label='slope 1 reference');ax.loglog(Nvals,calls_df.grover_expected_calls_with_restarts.iloc[0]*(Nvals/Nvals[0])**.5,'--',label='slope 0.5 reference');ax.set(xlabel='N',ylabel='Oracle calls');ax.grid(True,which='both',alpha=.25);ax.legend();fig.tight_layout();fig.savefig(RESULTS/'trackI_search_calls.png',dpi=130);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5));ax.semilogy(time_df.n,time_df.python_linear_seconds,'o-',label='Python linear');ax.semilogy(time_df.n,time_df.numpy_argmax_seconds,'o-',label='NumPy argmax');ax.semilogy(time_df.n,time_df.numpy_statevector_grover_seconds,'o-',label='NumPy Grover simulator');ax.set(xlabel='n qubits (N=2^n)',ylabel='Seconds');ax.grid(True,which='both',alpha=.25);ax.legend();fig.tight_layout();fig.savefig(RESULTS/'trackI_search_time.png',dpi=130);plt.close(fig)
    print("Classical Grover simulation costs about sqrt(N) iterations times N amplitudes, so it cannot beat a linear scan. A physical speedup needs fault-tolerant hardware and an oracle cheaper than reading/loading the data (THEORETICAL).",flush=True)
    print(f"Saved {RESULTS/'trackI_grover_vs_linear.csv'} and plots.",flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--quick',action='store_true');run(ap.parse_args().quick)
