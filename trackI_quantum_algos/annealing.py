"""Sparse exact dynamics in the global-spin-flip-even Max-Cut sector."""
import math
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh, expm_multiply


def chain_lower_bound(degree, topology="chimera"):
    if degree < 0: raise ValueError("degree must be non-negative")
    cap=6 if topology.lower()=="chimera" else 15 if topology.lower()=="pegasus" else None
    if cap is None: raise ValueError("topology must be chimera or pegasus")
    return max(1, math.ceil((int(degree)-2)/ (cap-2)))


def sector_driver(n):
    """Return the projected sum_i X_i and the even-parity orbit representatives."""
    if n<1: raise ValueError("n must be positive")
    dim=1<<n; reps=[x for x in range(dim) if x < (dim-1-x)]; where={x:i for i,x in enumerate(reps)}
    rows=[]; cols=[]; vals=[]
    for col,r in enumerate(reps):
        for state in (r,dim-1-r):
            for q in range(n):
                nxt=state^(1<<q); rep=min(nxt,dim-1-nxt)
                rows.append(where[rep]); cols.append(col); vals.append(0.5)
    A=coo_matrix((vals,(rows,cols)),shape=(len(reps),len(reps))).tocsr()
    return A,np.asarray(reps,dtype=int)


def sector_gap(n, energies, points=200):
    energies=np.asarray(energies,dtype=float); dim=1<<n
    if energies.shape!=(dim,) or not np.allclose(energies,energies[::-1]): raise ValueError("energies must be globally spin-flip symmetric")
    A,reps=sector_driver(n); E=energies[reps]; scale=max(abs(E)) or 1.0; E=E/scale
    ss=np.linspace(0,1,points); gaps=[]
    for s in ss:
        H=-(1-s)*A+diags(s*E,format="csr")
        if np.isclose(s,1.0):
            vals=np.partition(E,1)[:2].copy()
        elif H.shape[0] <= 2:
            vals=np.linalg.eigvalsh(H.toarray())[:2]
        else:
            vals=eigsh(H,k=2,which="SA",return_eigenvectors=False,tol=1e-9)
        vals.sort(); gaps.append(vals[1]-vals[0])
    idx=int(np.argmin(gaps)); return ss,np.asarray(gaps),float(gaps[idx]),float(ss[idx]),(A,reps,E)


def anneal_probability(A,reps,E_scaled,n,T,steps):
    if T<=0 or steps<1: raise ValueError("T and steps must be positive")
    dim=1<<n; psi=np.full(len(reps),1/math.sqrt(len(reps)),dtype=complex)
    # The final optimal assignments form complement pairs in this sector.
    emin=float(np.min(E_scaled)); ground=np.isclose(E_scaled,emin,atol=1e-10)
    dt=T/steps
    for j in range(steps):
        s=(j+0.5)/steps
        H=-(1-s)*A+diags(s*E_scaled,format="csr")
        psi=expm_multiply((-1j*dt)*H,psi)
    return float(np.sum(np.abs(psi[ground])**2))


def energies_from_ising(J,h,offset):
    n=len(h); E=np.empty(1<<n)
    for x in range(1<<n):
        spin=np.array([1 if (x>>i)&1 else -1 for i in range(n)])
        E[x]=sum(J[i,j]*spin[i]*spin[j] for i in range(n) for j in range(i+1,n))+np.dot(h,spin)+offset
    return E


def ising_maxcut(edges,n):
    from trackB_discrete.qubo import maxcut_to_qubo,qubo_to_ising
    W=np.zeros((n,n),dtype=float)
    for i,j,w in edges: W[i,j]=W[j,i]=w
    Q,off=maxcut_to_qubo(W); J,h,ioff=qubo_to_ising(Q,off)
    return J,h,ioff,energies_from_ising(J,h,ioff),W
