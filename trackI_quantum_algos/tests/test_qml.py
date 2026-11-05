import numpy as np
from trackI_quantum_algos.qml import (
    feature_map_state, feature_map_states, kernel_matrix, kernel_pair_count,
    qiskit_crosscheck, train_variational, variational_loss_and_grad,
)
from trackI_quantum_algos.exp5_grover_vs_linear import linear_average, grover_numpy


def test_qiskit_state_and_kernel_crosscheck_three_pairs():
    a,b=qiskit_crosscheck(123)
    assert a<1e-9 and b<1e-9


def test_exact_kernel_properties_and_orthogonal_hand_pair():
    X=np.random.default_rng(4).uniform(0,np.pi,(8,3));K=kernel_matrix(X)
    assert np.max(abs(K-K.T))<1e-12
    assert np.max(abs(np.diag(K)-1))<1e-12
    assert np.linalg.eigvalsh(K).min()>-1e-10
    # For one qubit this two-repeat encoding maps x=0 and x=pi to orthogonal states.
    a=feature_map_state(np.array([0.]));b=feature_map_state(np.array([np.pi]))
    assert abs(np.vdot(a,b))**2<1e-12
    assert kernel_pair_count(5)==5*(5-1)//2==10


def test_parameter_shift_matches_finite_difference():
    rng=np.random.default_rng(8);X=rng.uniform(0,np.pi,(5,3));y=np.array([-1,1,-1,1,1]);p=rng.normal(0,.2,(2,3,2))
    loss,g,_=variational_loss_and_grad(X,y,p);fd=np.zeros_like(p);eps=1e-6
    from trackI_quantum_algos.qml import variational_output
    for ix in np.ndindex(p.shape):
        pp=p.copy();pm=p.copy();pp[ix]+=eps;pm[ix]-=eps
        lp=np.mean((variational_output(X,pp)-y)**2);lm=np.mean((variational_output(X,pm)-y)**2);fd[ix]=(lp-lm)/(2*eps)
    rel=np.linalg.norm(g-fd)/(np.linalg.norm(g)+1e-12)
    assert rel<1e-6


def test_variational_training_reduces_tiny_separable_loss():
    X=np.array([[0.05],[0.2],[2.9],[3.1]])
    y=np.array([-1.,-1.,1.,1.])
    _,hist,_=train_variational(X,y,layers=1,lr=.08,iterations=50,seed=3)
    assert hist[-1]<hist[0]


def test_search_oracle_counts_and_grover_success():
    average,worst=linear_average(16)
    assert average==(16+1)/2 and worst==16
    probability,calls=grover_numpy(4,7,3)
    assert calls==3
    assert abs(probability-0.961)<1e-3
