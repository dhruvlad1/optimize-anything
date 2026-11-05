"""Exact simulated quantum annealing and theoretical minor embedding counts."""
import argparse,json,math,time
from pathlib import Path
import numpy as np
import pandas as pd

from trackI_quantum_algos.annealing import chain_lower_bound, ising_maxcut, sector_gap, anneal_probability
from trackI_quantum_algos.qiskit_info import print_qiskit_info

ROOT=Path(__file__).resolve().parents[1]; RESULTS=ROOT/'results'; RESULTS.mkdir(exist_ok=True)
INSTANCE_FILE=RESULTS/'trackB_maxcut_instances.json'


def run(quick=False):
    print_qiskit_info()
    data=json.loads(INSTANCE_FILE.read_text(encoding='utf-8'))
    print(f"Inspected JSON: list[{len(data)}], record keys={sorted(data[0])}, n sizes={sorted(set(r['n'] for r in data))}.",flush=True)
    instances=[x for x in data if x['n']==10][:5]
    if quick: instances=instances[:1]
    Ts=[1,2,5,10,20,50,100,200]
    if quick: Ts=[2,10,50]
    print(f"Estimated runtime: {90 if quick else 420}s; instances={len(instances)}, 400 steps per T; --quick limits to one instance and three T values.",flush=True)
    sim_rows=[]; gap_plot=None
    for inst in instances:
        n=inst['n']; J,h,off,E,W=ising_maxcut(inst['edges'],n)
        if abs(np.min(E)+inst['optimal_cut_value'])>1e-8: raise AssertionError(f"Ising minimum {np.min(E)} != -optimal cut {-inst['optimal_cut_value']}")
        s,gaps,gmin,smin,cache=sector_gap(n,E,points=200); A,reps,Esc=cache
        if gap_plot is None: gap_plot=(s,gaps,inst['instance_id'])
        print(f"{inst['instance_id']}: minimum relevant + parity-sector gap {gmin:.7g} at s={smin:.4f} (SIMULATED)",flush=True)
        optimal=np.flatnonzero(np.isclose(E,np.min(E),atol=1e-9));
        print(f"  optimal ground assignments={len(optimal)}; cost minimum verified as {-inst['optimal_cut_value']}.",flush=True)
        for T in Ts:
            tic=time.perf_counter(); prob=anneal_probability(A,reps,Esc,n,T,400); elapsed=time.perf_counter()-tic
            gap_scale=("infinite (zero finite-grid gap)" if gmin<=1e-12 else f"THEORETICAL: {1/(gmin*gmin):.6g}")
            sim_rows.append({"label":"SIMULATED","instance_id":inst['instance_id'],"n":n,"T":T,"steps":400,"success_probability":prob,
                "gap_min":gmin,"s_at_gap":smin,"simulation_seconds":elapsed,"theory_gap_scale":gap_scale})
            print(f"  progress {inst['instance_id']} T={T}: p_ground={prob:.5f} ({elapsed:.2f}s)",flush=True)
        if not quick:
            for T in (10,100):
                p400=anneal_probability(A,reps,Esc,n,T,400); p800=anneal_probability(A,reps,Esc,n,T,800); diff=abs(p400-p800)
                print(f"  convergence T={T}: 400 steps={p400:.7f}, 800 steps={p800:.7f}, difference={diff:.2e}",flush=True)
                if diff>1e-3: raise AssertionError(f"annealing step convergence failed for {inst['instance_id']} T={T}")
    df=pd.DataFrame(sim_rows)
    print("\nAnnealing dynamics table (each row labeled)",flush=True); print(df.to_string(index=False),flush=True)
    if len(df):
        for ident,sub in df.groupby('instance_id'):
            good=sub[sub.success_probability>=.9]
            g=float(sub.gap_min.iloc[0]);scale="infinite because gap_min=0" if g<=1e-12 else f"{1/(g*g):.4g}"
            print(f"{ident}: smallest sampled T with success >=0.9: {int(good['T'].min()) if len(good) else 'none in sweep'}; compare THEORETICAL 1/gap_min^2={scale}.",flush=True)
    # Lower-bound embedding counts for all 15 Track B graphs.
    embed=[]
    for inst in data:
        n=inst['n']; deg=[0]*n
        for i,j,w in inst['edges']: deg[i]+=1;deg[j]+=1
        chim=sum(chain_lower_bound(d,'chimera') for d in deg); peg=sum(chain_lower_bound(d,'pegasus') for d in deg)
        embed.append({"label":"THEORETICAL","instance_id":inst['instance_id'],"n":n,"logical_degrees":','.join(map(str,deg)),
                      "chimera_physical_qubit_lower_bound":chim,"pegasus_physical_qubit_lower_bound":peg})
    edf=pd.DataFrame(embed);print("\nEmbedding lower bounds (degree counts only; actual embeddings need more; no annealing hardware)",flush=True);print(edf.to_string(index=False),flush=True)
    df.to_csv(RESULTS/'trackI_annealing.csv',index=False)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    if gap_plot:
        fig,ax=plt.subplots();ax.plot(gap_plot[0],gap_plot[1]);ax.set(xlabel='s',ylabel='Gap in P=+1 sector',title=gap_plot[2]);ax.grid(alpha=.3);fig.tight_layout();fig.savefig(RESULTS/'trackI_annealing_gap.png',dpi=120);plt.close(fig)
    if len(df):
        fig,ax=plt.subplots()
        for ident,sub in df.groupby('instance_id'):ax.plot(sub['T'],sub.success_probability,marker='o',label=ident)
        ax.set_xscale('log');ax.set(xlabel='Anneal time T',ylabel='Ground-manifold probability');ax.grid(alpha=.3);ax.legend(fontsize=7);fig.tight_layout();fig.savefig(RESULTS/'trackI_annealing_success.png',dpi=120);plt.close(fig)
    # Keep the theoretical embedding table beside the requested dynamics output.
    edf.to_csv(RESULTS/'trackI_embedding_bounds.csv',index=False)
    print(f"Saved {RESULTS/'trackI_annealing.csv'} and plots.",flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--quick',action='store_true');run(p.parse_args().quick)
