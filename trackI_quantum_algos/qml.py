"""Batched NumPy statevector tools for small quantum-kernel and VQC studies.

Qubit q[0] is the least-significant basis bit. Only cross-check helpers use
Qiskit; all training and kernel computation below are pure NumPy.
"""
from __future__ import annotations

import math
import numpy as np


def _features(X):
    x=np.asarray(X,dtype=float)
    if x.ndim==1:x=x[None,:]
    if x.ndim!=2 or x.shape[1]<1 or not np.all(np.isfinite(x)):
        raise ValueError("features must be a finite (samples, qubits) array")
    return x


def _h(states,q):
    step=1<<q; lo=states.reshape(states.shape[0],-1,2*step); a=lo[:,:,:step].copy(); b=lo[:,:,step:].copy()
    lo[:,:,:step]=(a+b)/math.sqrt(2);lo[:,:,step:]=(a-b)/math.sqrt(2)


def _phase_z(states,q,theta):
    idx=np.arange(states.shape[1]); z=1-2*((idx>>q)&1)
    states*=np.exp(-0.5j*theta*z)[None,:]


def _phase_zz(states,q1,q2,theta):
    idx=np.arange(states.shape[1]);z1=1-2*((idx>>q1)&1);z2=1-2*((idx>>q2)&1)
    states*=np.exp(-0.5j*theta*z1*z2)[None,:]


def apply_feature_map(state,x,inverse=False):
    """Apply the two-repeat H/RZ/RZZ ZZ feature map to one or batched states."""
    x=np.asarray(x,dtype=float); one=np.asarray(state).ndim==1
    states=np.asarray(state,dtype=complex).reshape(1,-1).copy() if one else np.asarray(state,dtype=complex).copy()
    n=x.size
    if states.ndim!=2 or states.shape[1] != 1<<n: raise ValueError("state dimension must be 2**len(x)")
    if not inverse:
        for _ in range(2):
            for q in range(n):_h(states,q)
            for q in range(n):_phase_z(states,q,x[q])
            for q in range(n):
                for r in range(q+1,n):_phase_zz(states,q,r,x[q]*x[r])
    else:
        for _ in range(2):
            for q in reversed(range(n)):
                for r in reversed(range(q+1,n)):_phase_zz(states,q,r,-x[q]*x[r])
            for q in reversed(range(n)):_phase_z(states,q,-x[q])
            for q in reversed(range(n)):_h(states,q)
    return states[0] if one else states


def feature_map_state(x):
    x=np.asarray(x,dtype=float)
    return apply_feature_map(np.eye(1,1<<x.size,0,dtype=complex)[0],x)


def feature_map_states(X):
    x=_features(X);states=np.zeros((len(x),1<<x.shape[1]),dtype=complex);states[:,0]=1
    return apply_feature_map(states,x[0]) if np.all(x==x[0]) and len(x)==1 else _batched_feature_map(states,x)


def _batched_feature_map(states,X):
    # Encoding gates differ by sample, so phase application is row-specific.
    n=X.shape[1]
    for _ in range(2):
        for q in range(n):_h(states,q)
        idx=np.arange(states.shape[1])
        for q in range(n):states*=np.exp(-.5j*X[:,q,None]*(1-2*((idx>>q)&1))[None,:])
        for q in range(n):
            for r in range(q+1,n):
                zz=(1-2*((idx>>q)&1))*(1-2*((idx>>r)&1))
                states*=np.exp(-.5j*(X[:,q]*X[:,r])[:,None]*zz[None,:])
    return states


def kernel_matrix(X,Y=None,shots=None,rng=None,project_psd=False):
    """Exact overlap kernel or binomial overlap estimates for unique pairs.

    Shot estimates use the inverse-feature-map overlap test. Their sampled
    matrix may be non-PSD; ``project_psd=True`` clips negative eigenvalues and
    renormalizes the diagonal before returning it.
    """
    x=_features(X);same=Y is None
    y=x if same else _features(Y)
    if x.shape[1]!=y.shape[1]:raise ValueError("feature dimensions must match")
    if shots is None:
        sx=feature_map_states(x);sy=sx if same else feature_map_states(y)
        K=np.abs(sx.conj()@sy.T)**2
    else:
        if not isinstance(shots,(int,np.integer)) or shots<1:raise ValueError("shots must be a positive integer")
        gen=np.random.default_rng(rng);K=np.empty((len(x),len(y)))
        sx=feature_map_states(x);sy=sx if same else feature_map_states(y)
        probs=np.clip(np.abs(sx.conj()@sy.T)**2,0,1)
        if same:
            K.fill(0)
            for i in range(len(x)):
                K[i,i]=1.
                for j in range(i):K[i,j]=K[j,i]=gen.binomial(shots,probs[i,j])/shots
        else:
            K=gen.binomial(shots,probs)/shots
    if project_psd:
        if K.shape[0]!=K.shape[1] or not same:raise ValueError("PSD projection requires a symmetric self-kernel")
        K=project_kernel_psd(K)
    return K


def project_kernel_psd(K):
    """Symmetrize, clip negative eigenvalues to zero, then unit-normalize diagonal."""
    a=np.asarray(K,dtype=float)
    if a.ndim!=2 or a.shape[0]!=a.shape[1]:raise ValueError("K must be square")
    vals,vecs=np.linalg.eigh((a+a.T)/2);out=(vecs*np.maximum(vals,0))@vecs.T
    d=np.sqrt(np.maximum(np.diag(out),1e-15));out=out/d[:,None]/d[None,:]
    return (out+out.T)/2


def kernel_pair_count(n,symmetric=True):
    if not isinstance(n,(int,np.integer)) or n<0:raise ValueError("n must be a non-negative integer")
    return int(n*(n-1)//2 if symmetric else n*n)


def _ry(states,q,theta):
    step=1<<q;v=states.reshape(states.shape[0],-1,2*step);a=v[:,:,:step].copy();b=v[:,:,step:].copy()
    c=np.cos(theta/2);s=np.sin(theta/2);v[:,:,:step]=c*a-1j*s*b;v[:,:,step:]=-1j*s*a+c*b


def _rz_ansatz(states,q,theta):_phase_z(states,q,theta)


def _cx(states,control,target):
    idx=np.arange(states.shape[1]);perm=idx^((1<<target)*(((idx>>control)&1)))
    states[:]=states[:,perm]


def variational_output(X,params):
    x=_features(X);theta=np.asarray(params,dtype=float)
    if theta.ndim!=3 or theta.shape[1:]!=(x.shape[1],2) or theta.shape[0]<1:raise ValueError("params must have shape (layers, qubits, 2)")
    states=_batched_feature_map(_zero_states(len(x),x.shape[1]),x)
    for layer in theta:
        for q in range(x.shape[1]):
            _ry(states,q,layer[q,0]);_rz_ansatz(states,q,layer[q,1])
        for q in range(x.shape[1]-1):_cx(states,q,q+1)
    idx=np.arange(states.shape[1]);z=1-2*(idx&1)
    return np.sum(np.abs(states)**2*z[None,:],axis=1).real


def _zero_states(batch,n):
    out=np.zeros((batch,1<<n),dtype=complex);out[:,0]=1;return out


def variational_loss_and_grad(X,y,params):
    x=_features(X);labels=np.asarray(y,dtype=float);theta=np.asarray(params,dtype=float)
    if labels.shape!=(len(x),) or np.any(~np.isin(labels,[-1,1])):raise ValueError("labels must be a +/-1 vector matching X")
    pred=variational_output(x,theta);err=pred-labels;loss=float(np.mean(err**2));grad=np.zeros_like(theta);evals=len(x)
    for l in range(theta.shape[0]):
        for q in range(theta.shape[1]):
            for r in range(2):
                plus=theta.copy();minus=theta.copy();plus[l,q,r]+=np.pi/2;minus[l,q,r]-=np.pi/2
                deriv=(variational_output(x,plus)-variational_output(x,minus))/2
                grad[l,q,r]=2*np.mean(err*deriv);evals+=2*len(x)
    return loss,grad,evals


def train_variational(X,y,layers=1,lr=.03,iterations=150,seed=0):
    x=_features(X);rng=np.random.default_rng(seed);theta=rng.normal(0,.15,size=(layers,x.shape[1],2));m=np.zeros_like(theta);v=np.zeros_like(theta);history=[];evals=0
    for t in range(1,iterations+1):
        loss,g,count=variational_loss_and_grad(x,y,theta);evals+=count;history.append(loss)
        m=.9*m+.1*g;v=.999*v+.001*g*g
        theta-=lr*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8)
    return theta,np.asarray(history),evals


def qiskit_crosscheck(seed=23):
    """Cross-check 3 random input-pair states and kernels against Qiskit."""
    import inspect
    import qiskit,qiskit_aer
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    # Pin inspected API facts beside the cross-check (Qiskit 2.x signatures).
    print(f"Qiskit {qiskit.__version__}; Aer {qiskit_aer.__version__}; rzz{inspect.signature(QuantumCircuit.rzz)}; Statevector.from_instruction{inspect.signature(Statevector.from_instruction)}",flush=True)
    rng=np.random.default_rng(seed);max_state=max_kernel=0.
    def qstate(x):
        n=len(x);qc=QuantumCircuit(n)
        for _ in range(2):
            for q in range(n):qc.h(q)
            for q in range(n):qc.rz(float(x[q]),q)
            for q in range(n):
                for r in range(q+1,n):qc.rzz(float(x[q]*x[r]),q,r)
        return Statevector.from_instruction(qc).data
    for _ in range(3):
        x=rng.uniform(0,np.pi,3);z=rng.uniform(0,np.pi,3);a=feature_map_state(x);b=feature_map_state(z);qa=qstate(x);qb=qstate(z)
        max_state=max(max_state,float(np.max(abs(a-qa))),float(np.max(abs(b-qb))))
        max_kernel=max(max_kernel,abs(abs(np.vdot(a,b))**2-abs(np.vdot(qa,qb))**2))
    if max(max_state,max_kernel)>=1e-9:raise AssertionError(f"NumPy/Qiskit feature-map cross-check failed: {max_state}, {max_kernel}")
    return max_state,max_kernel
