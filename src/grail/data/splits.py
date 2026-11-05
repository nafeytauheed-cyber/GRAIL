"""Splits are assigned before any learned preprocessing or sampling."""
import numpy as np
import pandas as pd

def deduplicate(df):
    d=df.copy().reset_index(drop=True)
    # Exclude every copy of conflicting incident labels; do not select by target.
    keys=['Start_Time','Start_Lat','Start_Lng']
    for subset in [['ID'],keys]:
        if all(k in d for k in subset):
            complete=d[subset].notna().all(axis=1)
            conflict=d.loc[complete].groupby(subset,dropna=False)['Severity'].transform('nunique').gt(1)
            d=d.drop(index=conflict.index[conflict])
            complete=d[subset].notna().all(axis=1)
            dup=d.loc[complete].duplicated(subset,keep='first')
            d=d.drop(index=dup.index[dup])
    return d.reset_index(drop=True)

def split_data(df, mode='temporal', train_end='2021-01-01', val_end='2022-01-01',
               cal_end='2023-01-01', validation_states=('NV','OR'), test_states=('CA','TX')):
    """Four disjoint roles: training, selection validation, calibration, final test.

    Geographic evaluation uses unseen validation states before cal_end; calibration
    uses later records from validation states. Final test states never tune anything.
    """
    d=deduplicate(df)
    t=pd.to_datetime(d.Start_Time,errors='coerce',format='mixed')
    if t.isna().any(): raise ValueError('Invalid timestamps must be quarantined')
    if not pd.Timestamp(train_end)<pd.Timestamp(val_end)<pd.Timestamp(cal_end):
        raise ValueError('Cutoffs must increase')
    if mode=='temporal':
        masks=[t<pd.Timestamp(train_end),t.between(pd.Timestamp(train_end),pd.Timestamp(val_end),inclusive='left'),
               t.between(pd.Timestamp(val_end),pd.Timestamp(cal_end),inclusive='left'),t>=pd.Timestamp(cal_end)]
    elif mode=='geographic':
        vs,ts=set(validation_states),set(test_states)
        if not vs or not ts or vs & ts: raise ValueError('Validation/test states must be nonempty and disjoint')
        masks=[~d.State.isin(vs|ts)&(t<pd.Timestamp(train_end)),
               d.State.isin(vs)&(t<pd.Timestamp(val_end)),
               d.State.isin(vs)&t.between(pd.Timestamp(val_end),pd.Timestamp(cal_end),inclusive='left'),
               d.State.isin(ts)&(t>=pd.Timestamp(cal_end))]
    else: raise ValueError('mode must be temporal or geographic')
    parts={n:d.loc[m].copy() for n,m in zip(['train','validation','calibration','test'],masks)}
    if any(len(v)==0 for v in parts.values()): raise ValueError('An evaluation split is empty')
    return parts

def split_summary(parts):
    return {k:{'n':len(v),'classes':{str(c):int((v.Severity==c).sum()) for c in range(1,5)},
               'states':sorted(v.State.unique().tolist()),'start':str(v.Start_Time.min()),'end':str(v.Start_Time.max())} for k,v in parts.items()}
