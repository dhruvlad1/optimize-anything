"""Quantum kernels and variational classification versus classical baselines."""
import argparse,math,time,warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_moons,make_circles,load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV,StratifiedKFold,train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from trackI_quantum_algos.qml import (feature_map_states,kernel_matrix,kernel_pair_count,
    project_kernel_psd,qiskit_crosscheck,train_variational,variational_loss_and_grad,variational_output)
from trackI_quantum_algos.qiskit_info import print_qiskit_info

ROOT=Path(__file__).resolve().parents[1];RESULTS=ROOT/'results';RESULTS.mkdir(exist_ok=True)
LR_GRID=[.02,.06]
LAYERS=[1,2,3]
CV_C=[.1,1.,10.]


def _scale(train,test):
    sc=MinMaxScaler(feature_range=(0,np.pi));a=sc.fit_transform(train);b=np.clip(sc.transform(test),0,np.pi)
    return a,b


def dataset(name,seed):
    if name=='moons':
        X,y=make_moons(n_samples=200,noise=.2,random_state=seed)
        a,b,ya,yb=train_test_split(X,y,test_size=100,random_state=seed,stratify=y)
    elif name=='circles':
        X,y=make_circles(n_samples=200,noise=.1,factor=.5,random_state=seed)
        a,b,ya,yb=train_test_split(X,y,test_size=100,random_state=seed,stratify=y)
    else:
        data=load_breast_cancer();X,y=data.data,data.target
        ix,_,iy,_=train_test_split(np.arange(len(y)),y,train_size=300,random_state=seed,stratify=y)
        X,y=X[ix],y[ix]
        a,b,ya,yb=train_test_split(X,y,test_size=100,train_size=200,random_state=seed,stratify=y)
        std=StandardScaler().fit(a);a=std.transform(a);b=std.transform(b)
        pca=PCA(n_components=4,random_state=seed).fit(a);a=pca.transform(a);b=pca.transform(b)
    a,b=_scale(a,b)
    # Convert sklearn labels {0,1} to the requested {-1,+1} convention.
    return a,b,np.where(ya==1,1.,-1.),np.where(yb==1,1.,-1.)


def _cv(n_splits,seed):return StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)


def _grid(est,params,X,y,cv):
    gs=GridSearchCV(est,params,cv=cv,scoring='accuracy',n_jobs=1,refit=True,error_score='raise')
    gs.fit(X,y);return gs


def _mlp_fit(X,y,Xte,yte,hidden,seed,cv):
    est=MLPClassifier(hidden_layer_sizes=(hidden,),max_iter=250,random_state=seed,solver='adam',early_stopping=False)
    start=time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',ConvergenceWarning)
        gs=_grid(est,{'alpha':[1e-4,1e-2],'learning_rate_init':[.001,.01]},X,y,cv)
    model=gs.best_estimator_;return accuracy_score(y,model.predict(X)),accuracy_score(yte,model.predict(Xte)),time.perf_counter()-start,model


def _vqc_cv(X,y,seed,folds,cv_iters):
    folds_iter=list(_cv(folds,seed).split(X,y));records=[];evals=0;params=np.zeros((1,X.shape[1],2))
    for L in LAYERS:
        for lr in LR_GRID:
            scores=[]
            for fold,(tr,va) in enumerate(folds_iter):
                _,hist,c=train_variational(X[tr],y[tr],layers=L,lr=lr,iterations=cv_iters,seed=seed+fold*11+L)
                evals+=c;pred=variational_output(X[va],params)
                evals+=len(va);scores.append(accuracy_score(y[va],np.where(pred>=0,1.,-1.)))
            records.append((float(np.mean(scores)),L,lr))
    return max(records),evals


def run(quick=False):
    print("Quantum kernels/VQC are SIMULATED with batched NumPy statevectors; times are classical-simulator wall-clock timings.",flush=True)
    print_qiskit_info()
    full_est=900
    if quick:
        seeds=[11,29];folds=3;cv_iters=3;final_iters=30;print(f"Estimated quick runtime: 90s; --quick uses 2 seeds, 3 CV folds, 3 candidate VQC iterations and 30 final iterations.",flush=True)
    else:
        # A 150-iteration, six-setting, five-fold VQC sweep is estimated above ten minutes.
        # Keep all five seeds, datasets, folds and final refits; shorten candidate CV fits only.
        seeds=[11,29,47,71,101];folds=5;cv_iters=10;final_iters=150
        print(f"Estimated unreduced runtime: >{full_est}s (>10 min). Estimated reduced full runtime: about 300-400s. Reduced VQC hyperparameter-candidate fits from 150 to {cv_iters} Adam iterations; final selected models retain 150 iterations. Five seeds, all data, and five-fold CV remain.",flush=True)
    st,ke=qiskit_crosscheck(314);print(f"Qiskit feature-map cross-check: max state error={st:.3e}, max kernel error={ke:.3e} (limit 1e-9).",flush=True)
    # Parameter-shift derivative self-check on independent deterministic inputs.
    rng=np.random.default_rng(72);Xc=rng.uniform(0,np.pi,(5,2));yc=np.array([-1.,1.,-1.,1.,1.]);pc=rng.normal(0,.1,(1,2,2))
    _,gc,_=variational_loss_and_grad(Xc,yc,pc);fd=np.zeros_like(pc);eps=1e-6
    for ix in np.ndindex(pc.shape):
        p1=pc.copy();p0=pc.copy();p1[ix]+=eps;p0[ix]-=eps
        fd[ix]=(np.mean((variational_output(Xc,p1)-yc)**2)-np.mean((variational_output(Xc,p0)-yc)**2))/(2*eps)
    grad_rel=float(np.linalg.norm(gc-fd)/(np.linalg.norm(gc)+1e-12))
    print(f"Parameter-shift finite-difference relative error={grad_rel:.3e} (limit 1e-6).",flush=True)
    if grad_rel>=1e-6:raise AssertionError("parameter-shift finite-difference cross-check failed")

    raw=[];paired=[];total=time.perf_counter()
    for name in ('moons','circles','breast_cancer'):
        for seed in seeds:
            print(f"progress dataset={name}, seed={seed}: preparing train-only transforms and CV",flush=True)
            X,Xte,y,yte=dataset(name,seed);cv=_cv(folds,seed);nq=X.shape[1];yd=y.astype(int)
            classical={}
            # Logistic regression, RBF-SVM and two MLP baselines.
            start=time.perf_counter();gl=_grid(LogisticRegression(max_iter=1000),{'C':[.01,.1,1,10]},X,yd,cv);tl=time.perf_counter()-start
            for model_key,model,elapsed,pcount,ptype in [
                ('logistic',gl.best_estimator_,tl,nq+1,'trainable_parameters')]:
                tracc=accuracy_score(yd,model.predict(X));teacc=accuracy_score(yte.astype(int),model.predict(Xte));classical[model_key]=teacc
                raw.append({'label':'MEASURED','dataset':name,'seed':seed,'model':model_key,'train_accuracy':tracc,'test_accuracy':teacc,'training_seconds':elapsed,'trainable_parameters':pcount,'support_vectors':np.nan,'parameter_count_type':ptype,'circuit_evaluations':0,'shots_per_pair':0,'total_shots':0})
            start=time.perf_counter();gr=_grid(SVC(kernel='rbf'),{'C':CV_C,'gamma':['scale',.1,1.]},X,yd,cv);tr=time.perf_counter()-start;mr=gr.best_estimator_
            ra=accuracy_score(yd,mr.predict(X));ea=accuracy_score(yte.astype(int),mr.predict(Xte));classical['rbf_svm']=ea
            raw.append({'label':'MEASURED','dataset':name,'seed':seed,'model':'rbf_svm','train_accuracy':ra,'test_accuracy':ea,'training_seconds':tr,'trainable_parameters':np.nan,'support_vectors':len(mr.support_),'parameter_count_type':'support vectors (not comparable)','circuit_evaluations':0,'shots_per_pair':0,'total_shots':0})
            for tag,hidden in [('mlp_16',16)]:
                mt,me,tm,mm=_mlp_fit(X,yd,Xte,yte.astype(int),hidden,seed,cv);classical[tag]=me
                count=(nq+2)*hidden+1
                raw.append({'label':'MEASURED','dataset':name,'seed':seed,'model':tag,'train_accuracy':mt,'test_accuracy':me,'training_seconds':tm,'trainable_parameters':count,'support_vectors':np.nan,'parameter_count_type':'trainable_parameters','circuit_evaluations':0,'shots_per_pair':0,'total_shots':0})

            # Select the VQC settings exclusively with folds of the training split.
            start=time.perf_counter();(cvscore,Lbest,lrbest),cv_evals=_vqc_cv(X,y,seed,folds,cv_iters)
            theta,hist,final_evals=train_variational(X,y,layers=Lbest,lr=lrbest,iterations=final_iters,seed=seed)
            vtrain=variational_output(X,theta);vtest=variational_output(Xte,theta)
            vacc=accuracy_score(y,np.where(vtrain>=0,1.,-1.));veacc=accuracy_score(yte,np.where(vtest>=0,1.,-1.));evals=cv_evals+final_evals+len(X)+len(Xte)
            qtime=time.perf_counter()-start
            qparams=2*Lbest*nq;classical['variational_classifier']=veacc
            raw.append({'label':'SIMULATED','dataset':name,'seed':seed,'model':'variational_classifier','train_accuracy':vacc,'test_accuracy':veacc,'training_seconds':qtime,'trainable_parameters':qparams,'support_vectors':np.nan,'parameter_count_type':'trainable_parameters','circuit_evaluations':evals,'shots_per_pair':0,'total_shots':0,'selected_layers':Lbest,'selected_lr':lrbest,'cv_accuracy':cvscore,'final_iterations':final_iters})
            matched=max(1,min(32,min(range(1,33),key=lambda h:abs(((nq+2)*h+1)-qparams))))
            print(f"  Parameter matching: VQC has {qparams} trainable rotations; MLP matched hidden={matched} has {(nq+2)*matched+1} parameters.",flush=True)
            mt,me,tm,mm=_mlp_fit(X,yd,Xte,yte.astype(int),matched,seed,cv);classical['mlp_matched']=me
            raw.append({'label':'MEASURED','dataset':name,'seed':seed,'model':'mlp_matched','train_accuracy':mt,'test_accuracy':me,'training_seconds':tm,'trainable_parameters':(nq+2)*matched+1,'support_vectors':np.nan,'parameter_count_type':'trainable_parameters','circuit_evaluations':0,'shots_per_pair':0,'total_shots':0,'matched_hidden':matched,'matched_to_quantum_parameters':qparams})

            # Exact and sampled quantum-kernel SVMs; one full train/test kernel per representation.
            pair_count=kernel_pair_count(len(X))+len(X)*len(Xte)
            for shots in (None,100,1000):
                start=time.perf_counter()
                if shots is None:
                    Ktr=kernel_matrix(X);Kte=kernel_matrix(Xte,X)
                else:
                    Ktr=kernel_matrix(X,shots=shots,rng=seed+shots);Kte=kernel_matrix(Xte,X,shots=shots,rng=seed+shots+1)
                    Ktr=project_kernel_psd(Ktr)
                gs=_grid(SVC(kernel='precomputed'),{'C':CV_C},Ktr,yd,cv);sv=gs.best_estimator_
                tra=accuracy_score(yd,sv.predict(Ktr));tea=accuracy_score(yte.astype(int),sv.predict(Kte));elapsed=time.perf_counter()-start
                model='quantum_kernel_exact' if shots is None else f'quantum_kernel_{shots}_shots';classical[model]=tea
                raw.append({'label':'SIMULATED','dataset':name,'seed':seed,'model':model,'train_accuracy':tra,'test_accuracy':tea,'training_seconds':elapsed,'trainable_parameters':np.nan,'support_vectors':len(sv.support_),'parameter_count_type':'support vectors (not comparable)','circuit_evaluations':pair_count,'shots_per_pair':0 if shots is None else shots,'total_shots':0 if shots is None else pair_count*shots,'psd_clipped':shots is not None})

            classical_best=max(v for k,v in classical.items() if not k.startswith('quantum_') and k!='variational_classifier')
            for qm in ('quantum_kernel_exact','quantum_kernel_100_shots','quantum_kernel_1000_shots','variational_classifier'):
                qa=classical.get(qm) if qm!='variational_classifier' else veacc
                paired.append({'label':'MEASURED','dataset':name,'seed':seed,'quantum_model':qm,'quantum_test_accuracy':qa,'best_classical_test_accuracy':classical_best,
                    'outcome':'win' if qa>classical_best else 'tie' if qa==classical_best else 'loss'})
            print(f"  completed {name} seed={seed}: VQC L={Lbest}, lr={lrbest:g}; best classical test acc={classical_best:.3f}; VQC={veacc:.3f}",flush=True)

    # Learning-curve ablation on a 260-point moons sample (160 training, 100 test).
    print("Progress: moons training-size and exact-versus-shot ablation",flush=True)
    Xa,ya=make_moons(n_samples=260,noise=.2,random_state=909);xa,xt,ya,yt=train_test_split(Xa,ya,test_size=100,train_size=160,random_state=909,stratify=ya);xa,xt=_scale(xa,xt);ya=np.where(ya==1,1,-1);yt=np.where(yt==1,1,-1)
    ab=[];rng=np.random.default_rng(909)
    for size in (20,40,80,160):
        inds=rng.choice(len(xa),size=size,replace=False);xx=xa[inds];yy=ya[inds];cv=_cv(5,909+size);pairs=kernel_pair_count(size)+size*len(xt)
        for shots in ('rbf',None,100,1000):
            start=time.perf_counter()
            if shots=='rbf':
                gs=_grid(SVC(kernel='rbf'),{'C':CV_C,'gamma':['scale',.1,1.]},xx,yy.astype(int),cv);est=gs.best_estimator_;tr=est.predict(xx);te=est.predict(xt);model='rbf_svm';sv=len(est.support_);pcount=np.nan;ev=0;totalshots=0
            else:
                K=kernel_matrix(xx) if shots is None else kernel_matrix(xx,shots=shots,rng=909+size+shots)
                KT=kernel_matrix(xt,xx) if shots is None else kernel_matrix(xt,xx,shots=shots,rng=1000+size+shots)
                if shots is not None:K=project_kernel_psd(K)
                gs=_grid(SVC(kernel='precomputed'),{'C':CV_C},K,yy.astype(int),cv);est=gs.best_estimator_;tr=est.predict(K);te=est.predict(KT);model='quantum_kernel_exact' if shots is None else f'quantum_kernel_{shots}_shots';sv=len(est.support_);pcount=np.nan;ev=pairs;totalshots=0 if shots is None else pairs*shots
            ab.append({'label':'SIMULATED' if shots!='rbf' else 'MEASURED','dataset':'moons_ablation','seed':909,'model':model,'train_size':size,'train_accuracy':accuracy_score(yy,np.where(np.asarray(tr)>0,1,-1)),'test_accuracy':accuracy_score(yt,np.where(np.asarray(te)>0,1,-1)),
                'training_seconds':time.perf_counter()-start,'trainable_parameters':pcount,'support_vectors':sv,'parameter_count_type':'support vectors (not comparable)' if shots!='rbf' else 'support vectors (not comparable)',
                'circuit_evaluations':ev,'shots_per_pair':0 if shots in ('rbf',None) else shots,'total_shots':totalshots})
            print(f"  ablation n_train={size} model={model}: test accuracy={ab[-1]['test_accuracy']:.3f}",flush=True)
    df=pd.DataFrame(raw+ab);df.to_csv(RESULTS/'trackI_qml.csv',index=False);pdf=pd.DataFrame(paired);pdf.to_csv(RESULTS/'trackI_qml_paired.csv',index=False)
    summ=df[df.dataset.isin(['moons','circles','breast_cancer'])].groupby(['dataset','model']).agg(train_accuracy_mean=('train_accuracy','mean'),train_accuracy_sd=('train_accuracy','std'),test_accuracy_mean=('test_accuracy','mean'),test_accuracy_sd=('test_accuracy','std'),training_seconds_mean=('training_seconds','mean'),training_seconds_sd=('training_seconds','std')).reset_index();summ.insert(0,'label',summ.model.apply(lambda m:'SIMULATED' if m.startswith('quantum_') or m=='variational_classifier' else 'MEASURED'))
    print("\nPer-seed QML/classical results table (train and test splits; times are MEASURED wall-clock, quantum on a classical simulator)",flush=True);print(df.to_string(index=False),flush=True)
    print("\nMean and standard deviation over seeds",flush=True);print(summ.to_string(index=False),flush=True)
    print("\nPaired quantum-versus-best-classical outcomes",flush=True)
    ps=pdf.groupby(['dataset','quantum_model','outcome']).size().unstack(fill_value=0).reset_index();ps.insert(0,'label','MEASURED');print(ps.to_string(index=False),flush=True)
    print("\nMoons training-size/exact-versus-shot ablation",flush=True);print(pd.DataFrame(ab).to_string(index=False),flush=True)
    classical_rows=pdf.groupby('quantum_model').outcome.value_counts().unstack(fill_value=0)
    print("All quantum timings are classical-simulator timings. Small simulated datasets do not support a claim of quantum advantage.",flush=True)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,5));models=list(summ.model.unique());datasets=list(summ.dataset.unique());width=.8/max(len(models),1);x=np.arange(len(datasets))
    for i,m in enumerate(models):
        sub=summ[summ.model==m].set_index('dataset').reindex(datasets);ax.bar(x+i*width,sub.test_accuracy_mean,width,yerr=sub.test_accuracy_sd.fillna(0),capsize=2,label=m)
    ax.set_xticks(x+width*len(models)/2,datasets);ax.set(ylabel='Test accuracy',ylim=(0,1.05));ax.legend(fontsize=7,ncol=3);ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(RESULTS/'trackI_qml_accuracy.png',dpi=130);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));
    for model,sub in pd.DataFrame(ab).groupby('model'):ax.plot(sub.train_size,sub.test_accuracy,marker='o',label=model)
    ax.set(xlabel='Moons training samples',ylabel='Test accuracy',ylim=(0,1.05));ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(RESULTS/'trackI_qml_learning.png',dpi=130);plt.close(fig)
    print(f"Total elapsed {time.perf_counter()-total:.1f}s. Saved {RESULTS/'trackI_qml.csv'} and plots.",flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--quick',action='store_true');run(parser.parse_args().quick)
