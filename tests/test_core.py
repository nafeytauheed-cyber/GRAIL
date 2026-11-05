import numpy as np
import pandas as pd
import pytest
import torch
from grail import GRAILClassifier
from grail.models.ordinal import OrdinalNet,ordinal_targets
from grail.losses.groups import GroupWeights
from grail.data.synthetic import synthetic
from grail.data.features import features,FORBIDDEN
from grail.data.splits import split_data,deduplicate
from grail.evaluation.calibration import TemperatureCalibrator
from grail.data.preprocessing import preprocessor,clean_frame

def test_targets():
    assert ordinal_targets(torch.tensor([1,2,3,4])).tolist()==[[0,0,0],[1,0,0],[1,1,0],[1,1,1]]
    with pytest.raises(ValueError):ordinal_targets(torch.tensor([0,4]))

@pytest.mark.parametrize('head',['ordinal','coral','corn','ce'])
def test_probabilities_and_gradients(head):
    torch.manual_seed(1);m=OrdinalNet(5,head=head);x=torch.randn(20,5);z=m(x);p=m.probabilities(z)
    assert torch.isfinite(p).all() and (p>=0).all()
    assert torch.allclose(p.sum(1),torch.ones(20),atol=1e-6)
    if head in ['ordinal','coral']:
        assert torch.all(m.thresholds().diff()>0)
        assert torch.all(z.sigmoid().diff(dim=1)<=0)
    loss=m.losses(z,torch.tensor([1,2,3,4]*5)).mean();assert torch.isfinite(loss)
    loss.backward();assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in m.parameters())
    assert m.encoder[0].weight.grad.abs().sum()>0

@pytest.mark.parametrize('head',['ordinal','corn','ce'])
def test_extremes(head):
    m=OrdinalNet(2,head=head);z=m(torch.ones(4,2)*1e6)
    assert torch.isfinite(m.losses(z,torch.arange(1,5))).all()
    p=m.probabilities(z);assert torch.isfinite(p).all();assert (p>=0).all()

@pytest.mark.parametrize('mode',['erm','fixed','dro','adaptive'])
def test_groups(mode):
    g=GroupWeights([1,1000],mode=mode)
    for _ in range(100):w=g.update([10.,.1]);assert np.isfinite(w).all() and np.isclose(w.sum(),1) and (w>=0).all()
    if mode=='adaptive': assert w.min()>=.05

def test_small_group_shrinkage():
    small=GroupWeights([1,1000],tau=50);large=GroupWeights([1000,1000],tau=50)
    assert small.update([10,1])[0]<large.update([10,1])[0]

@pytest.fixture
def fitted():
    torch.set_num_threads(1);d=synthetic(400);X=features(d)
    m=GRAILClassifier(epochs=2,hidden=8,batch_size=64).fit(X.iloc[:300],d.Severity.iloc[:300],d.State.iloc[:300])
    return m,X,d

def test_serialization_deterministic(fitted,tmp_path):
    m,X,d=fitted;p=m.predict_proba(X.iloc[300:]);assert np.array_equal(p,m.predict_proba(X.iloc[300:]))
    m.save(tmp_path/'grail.pt');loaded=GRAILClassifier.load(tmp_path/'grail.pt')
    np.testing.assert_array_equal(p,loaded.predict_proba(X.iloc[300:]))
    assert set(m.predict(X)).issubset({1,2,3,4})

def test_missing_unseen(fitted):
    m,X,d=fitted;xx=X.iloc[300:].copy();xx['State']='UNSEEN';xx['Temperature(F)']=np.nan
    assert np.isfinite(m.predict_proba(xx)).all()
    with pytest.raises(ValueError):m.predict_proba(xx.drop(columns=['State']))

def test_calibration_separation(fitted):
    m,X,d=fitted
    with pytest.raises(ValueError):m.calibrate(X.iloc[:10],d.Severity.iloc[:10])
    m.calibrate(X.iloc[300:],d.Severity.iloc[300:]);assert m.temperature_>0

def test_train_only_imputation():
    tr=clean_frame(pd.DataFrame({'n':[1.,2.,np.nan],'c':['a','b',None]}));pp=preprocessor(tr).fit(tr)
    before=pp.named_transformers_['numeric'][0].statistics_.copy()
    pp.transform(clean_frame(pd.DataFrame({'n':[1e9],'c':['unseen']})))
    np.testing.assert_array_equal(before,pp.named_transformers_['numeric'][0].statistics_)
    assert before[0]==1.5

def test_feature_leakage():
    d=synthetic(40);X=features(d)
    assert not set(FORBIDDEN)&set(X.columns)
    d.loc[0,'Weather_Timestamp']='2030-01-01'
    assert pd.isna(features(d).loc[0,'Temperature(F)'])
    assert 'Source' not in X and 'Year' not in X

def test_splits():
    d=synthetic(3000);p=split_data(d)
    names=list(p)
    for a,b in zip(names,names[1:]):assert p[a].Start_Time.max()<p[b].Start_Time.min()
    geo=split_data(d,'geographic')
    assert not set(geo['train'].State)&set(geo['test'].State)
    assert not set(geo['train'].State)&set(geo['validation'].State)
    assert not set(geo['calibration'].State)&set(geo['test'].State)
    for a in names:
        for b in names:
            if a!=b: assert not set(p[a].ID)&set(p[b].ID)

def test_duplicate_conflicts():
    d=synthetic(20);du=pd.concat([d,d.iloc[:1]],ignore_index=True)
    assert len(deduplicate(du))==20
    du.loc[20,'Severity']=5-du.loc[0,'Severity']
    assert len(deduplicate(du))==19

def test_calibrator():
    p=np.array([[.7,.1,.1,.1],[.1,.1,.2,.6]])
    c=TemperatureCalibrator().fit(p,np.array([1,4]));q=c.transform(p)
    assert np.isclose(q.sum(1),1).all() and (q>=0).all()
    np.testing.assert_array_equal(q.argmax(1),p.argmax(1))

def test_invalid_labels(fitted):
    m,X,d=fitted
    with pytest.raises(ValueError):m.fit(X,np.zeros(len(X)),d.State)
    with pytest.raises(ValueError):GRAILClassifier(epochs=0).fit(X,d.Severity,d.State)

def test_warehouse(tmp_path):
    from grail.data.warehouse import prepare,audit,load_development_sample
    d=synthetic(200);d.to_csv(tmp_path/'raw.csv',index=False)
    prepare(tmp_path/'raw.csv',tmp_path/'d.parquet');r=audit(tmp_path/'d.parquet',tmp_path/'audit')
    assert r['rows']==200 and r['invalid_start']==0
    sample=load_development_sample(tmp_path/'d.parquet',50);assert len(sample)==50
    assert not set(FORBIDDEN)&set(features(sample))

def test_effective_weights_match_shrunken_objective():
    g=GroupWeights([1,50,1000]);g.update([3.,2.,1.]);loss=np.array([3.,2.,1.])
    r=g.counts/(g.counts+g.tau)
    s=r*loss+(1-r)*np.dot(g.prior,loss)
    assert np.isclose(np.dot(g.weights,s),np.dot(g.effective_weights(),loss))
    assert np.isclose(g.effective_weights().sum(),1)
    assert (g.effective_weights()>=0).all()

def test_adaptive_reduces_to_dro():
    a=GroupWeights([10,100],tau=0,regularization=0,mix=0)
    b=GroupWeights([10,100],mode='dro')
    for _ in range(4):
        np.testing.assert_allclose(a.update([2.,1.]),b.update([2.,1.]))
        np.testing.assert_allclose(a.effective_weights(),b.weights)

def test_same_seed_same_fitted_predictions():
    torch.set_num_threads(1);d=synthetic(300);x=features(d)
    a=GRAILClassifier(epochs=2,seed=7,hidden=8).fit(x,d.Severity,d.State)
    b=GRAILClassifier(epochs=2,seed=7,hidden=8).fit(x,d.Severity,d.State)
    np.testing.assert_array_equal(a.predict_proba(x),b.predict_proba(x))

def test_validation_rows_cannot_calibrate():
    torch.set_num_threads(1);d=synthetic(500);x=features(d)
    m=GRAILClassifier(epochs=1,hidden=8).fit(x.iloc[:300],d.Severity.iloc[:300],d.State.iloc[:300],X_val=x.iloc[300:400],y_val=d.Severity.iloc[300:400])
    with pytest.raises(ValueError):m.calibrate(x.iloc[300:400],d.Severity.iloc[300:400])

def test_corn_conditional_mask():
    import torch.nn.functional as F
    m=OrdinalNet(2,head='corn');z=torch.tensor([[.2,-1.,3.],[.1,.3,-.2],[2.,3.,4.],[1.,2.,3.]])
    y=torch.tensor([1,2,3,4]);loss=m.losses(z,y)
    reference=[]
    for i in range(4):
        v=0.
        for k in range(1,4):
            if y[i]>k-1:v=v+F.binary_cross_entropy_with_logits(z[i,k-1],torch.tensor(float(y[i]>k)))
        reference.append(v)
    torch.testing.assert_close(loss,torch.stack(reference))

def test_balanced_sampling_cannot_double_weight():
    d=synthetic(200);x=features(d)
    with pytest.raises(ValueError):GRAILClassifier(sampling=True,robustness='dro').fit(x,d.Severity,d.State)

def test_temporal_bad_cutoffs():
    with pytest.raises(ValueError):split_data(synthetic(100),'temporal',train_end='2022-01-01',val_end='2021-01-01')

@pytest.mark.parametrize('head',['ordinal','corn','ce'])
def test_learning_reduces_loss(head):
    torch.set_num_threads(1);torch.manual_seed(10)
    x=torch.linspace(-3,3,160).reshape(-1,1);y=torch.bucketize(x[:,0],torch.tensor([-1.5,0.,1.5]))+1
    m=OrdinalNet(1,hidden=8,head=head);opt=torch.optim.Adam(m.parameters(),lr=.03)
    before=m.losses(m(x),y).mean().item()
    for _ in range(80):
        loss=m.losses(m(x),y).mean();opt.zero_grad();loss.backward();opt.step()
    assert m.losses(m(x),y).mean().item()<before*.6
