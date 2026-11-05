import numpy as np
import pandas as pd
from .metrics import entropy

def failure_report(raw,y,p):
    d=raw.copy();d['error']=(p.argmax(1)+1)!=np.asarray(y)
    d['ordinal_error']=np.abs(p.argmax(1)+1-np.asarray(y));d['predicted_level4']=p[:,3]
    d['hour']=pd.to_datetime(d.Start_Time,format='mixed').dt.hour
    rows=[]
    for col in ['State','Source','Weather_Condition','hour','Traffic_Signal','Junction']:
        if col in d:
            for value,v in d.groupby(col,dropna=False):
                rows.append({'factor':col,'value':str(value),'n':len(v),'error_rate':v.error.mean(),
                             'ordinal_mae':v.ordinal_error.mean(),'mean_predicted_level4':v.predicted_level4.mean()})
    return pd.DataFrame(rows)

def prioritize(raw,p):
    d=raw[['ID','State']].copy();d['expected_disruption']=p@np.arange(1,5)
    d['p_level4']=p[:,3];d['entropy']=entropy(p)
    # Separate risk and review ranks avoids inventing an intervention utility scale.
    d['risk_rank']=d.expected_disruption.rank(method='first',ascending=False).astype(int)
    d['review_rank']=d.entropy.rank(method='first',ascending=False).astype(int)
    return d.sort_values('risk_rank')

def perturb_weather(X,seed=42):
    d=X.copy();r=np.random.default_rng(seed)
    # Bounded sensitivity scenarios, not a calibrated sensor-error distribution.
    for col,amount,lo,hi in [('Temperature(F)',2.,-100,140),('Humidity(%)',3.,0,100),
                             ('Visibility(mi)',.5,0,140),('Wind_Speed(mph)',1.,0,200)]:
        if col in d:
            original=d[col].to_numpy(dtype=float)
            eligible=(original>=lo)&(original<=hi)
            changed=np.clip(original+r.uniform(-amount,amount,len(d)),lo,hi)
            d[col]=np.where(eligible,changed,original)
    return d

def disagreement(predictions,y):
    p=np.stack(predictions);avg=p.mean(0)
    js=entropy(avg)-np.stack([entropy(v) for v in p]).mean(0)
    labels=p.argmax(2)+1
    return pd.DataFrame({'js_divergence':js,'models_disagree':np.any(labels!=labels[:1],axis=0),
                         'any_error':np.any(labels!=np.asarray(y)[None,:],axis=0)})
