import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_recall_fscore_support, cohen_kappa_score, confusion_matrix, log_loss

def entropy(p): return -(p*np.log(p.clip(1e-12))).sum(1)

def metrics(y,p,groups=None):
    y=np.asarray(y);p=np.asarray(p); pred=p.argmax(1)+1
    if p.shape!=(len(y),4) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-6): raise ValueError('Invalid probabilities')
    pr,re,f,s=precision_recall_fscore_support(y,pred,labels=[1,2,3,4],zero_division=0)
    conf=p.max(1);correct=pred==y;ece=0.
    for i in range(10):
        mask=(conf>=i/10)&((conf<(i+1)/10) if i<9 else (conf<=1))
        if mask.any(): ece+=mask.mean()*abs(correct[mask].mean()-conf[mask].mean())
    order=np.argsort(entropy(p),kind='stable')
    cov=np.linspace(.1,1,10);risk=[float((pred[order[:max(1,int(c*len(y)))]]!=y[order[:max(1,int(c*len(y)))]]).mean()) for c in cov]
    out={'macro_f1':float(f1_score(y,pred,labels=[1,2,3,4],average='macro',zero_division=0)),
         'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
         'ordinal_mae':float(np.abs(y-pred).mean()),'quadratic_kappa':float(cohen_kappa_score(y,pred,labels=[1,2,3,4],weights='quadratic')),
         'brier':float(((p-np.eye(4)[y.astype(int)-1])**2).sum(1).mean()),
         'nll':float(log_loss(y,p,labels=[1,2,3,4])),'ece_10':float(ece),
         'per_class':{str(i+1):{'precision':float(pr[i]),'recall':float(re[i]),'f1':float(f[i]),'support':int(s[i])} for i in range(4)},
         'confusion_matrix':confusion_matrix(y,pred,labels=[1,2,3,4]).tolist(),
         'risk_coverage':{'coverage':cov.tolist(),'error_rate':risk}}
    if groups is not None:
        groups=np.asarray(groups)
        gs={str(g):{'n':int((groups==g).sum()),'macro_f1':float(f1_score(y[groups==g],pred[groups==g],labels=[1,2,3,4],average='macro',zero_division=0)),
                    'ordinal_mae':float(np.abs(y[groups==g]-pred[groups==g]).mean())} for g in np.unique(groups)}
        out['groups']=gs;out['worst_group_macro_f1']=min(v['macro_f1'] for v in gs.values())
        out['worst_group_ordinal_mae']=max(v['ordinal_mae'] for v in gs.values())
    return out

def bootstrap_f1(y,p,seed=42,repeats=100):
    """IID row bootstrap descriptive CI; spatial dependence can make it optimistic."""
    rng=np.random.default_rng(seed);y=np.asarray(y);pred=p.argmax(1)+1;v=[]
    for _ in range(repeats):
        idx=rng.integers(0,len(y),len(y));v.append(f1_score(y[idx],pred[idx],labels=[1,2,3,4],average='macro',zero_division=0))
    return np.quantile(v,[.025,.975]).tolist()
