"""One runner, identical frozen splits, two validation-selected candidates per family."""
from pathlib import Path
import json,time,pickle,platform
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from ..data.features import features
from ..data.splits import split_data,split_summary
from ..models.classifier import GRAILClassifier
from ..models.baselines import baseline
from ..evaluation.metrics import metrics,bootstrap_f1
from ..evaluation.calibration import TemperatureCalibrator
from ..evaluation.investigation import failure_report,prioritize,perturb_weather,disagreement

NEURAL={'mlp':('ce','erm',None),'coral':('coral','erm',None),'corn':('corn','erm',None),
        'ordinal':('ordinal','erm',None),'ordinal_weighted':('ordinal','erm','balanced'),
        'ordinal_dro':('ordinal','dro',None),'ordinal_fixed':('ordinal','fixed',None),
        'grail':('ordinal','adaptive',None),'grail_no_shrink':('ordinal','adaptive',None),
        'grail_no_regularization':('ordinal','adaptive',None),'ordinal_sampled':('ordinal','erm',None)}
DEFAULT=['majority','logistic','forest','xgboost','catboost','mlp','coral','corn','ordinal_weighted','ordinal_dro','ordinal_fixed','grail','grail_no_shrink','grail_no_regularization','ordinal_sampled']

def run(df,out,seeds=(11,23,37),modes=('temporal','geographic'),names=None,epochs=25,trials=2,synthetic=True,feature_mode='primary'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    if trials not in [1,2]: raise ValueError('Supported candidate budget: 1 or 2')
    names=DEFAULT if names is None else names
    feat_options={'geography':feature_mode!='no_geo','provider':feature_mode=='provider','year':feature_mode=='year','weather':feature_mode!='no_weather'}
    records=[]
    for mode in modes:
        parts=split_data(df,mode)
        (out/f'{mode}_splits.json').write_text(json.dumps(split_summary(parts),indent=2))
        pd.concat([v[['ID']].assign(split=k) for k,v in parts.items()]).to_csv(out/f'{mode}_split_ids.csv',index=False)
        X={k:features(v,**feat_options) for k,v in parts.items()};y={k:v.Severity.to_numpy(dtype=int) for k,v in parts.items()}
        for seed in seeds:
            predictions={}
            for name in names:
                begin=time.perf_counter();best=None;best_loss=np.inf;selection=[]
                for trial in range(1 if name=='majority' else trials):
                    if name in NEURAL:
                        head,robust,cw=NEURAL[name]
                        model=GRAILClassifier(head=head,robustness=robust,class_weight=cw,seed=seed,epochs=epochs,
                            hidden=[32,48][trial],lr=[.002,.001][trial],sampling=name=='ordinal_sampled',
                            tau=0. if name=='grail_no_shrink' else 50.,
                            regularization=0. if name=='grail_no_regularization' else .1,
                            mix=0. if name=='grail_no_regularization' else .1)
                        model.fit(X['train'],y['train'],groups=parts['train'].State.to_numpy(),X_val=X['validation'],y_val=y['validation'])
                    else:
                        model=baseline(name,X['train'],seed,budget=trial)
                        model.fit(X['train'],y['train']-1)
                    p=model.predict_proba(X['validation'])
                    loss=float(log_loss(y['validation'],p,labels=[1,2,3,4]));selection.append(loss)
                    if loss<best_loss:best_loss=loss;best=model;best_trial=trial
                elapsed=time.perf_counter()-begin
                raw=best.predict_proba(X['test'])
                # Fit calibrator only on the fourth-role split; selection already frozen.
                cal=TemperatureCalibrator().fit(best.predict_proba(X['calibration']),y['calibration'])
                p=cal.transform(raw);predictions[name]=p
                t=time.perf_counter();best.predict_proba(X['test']);latency=(time.perf_counter()-t)/len(p)*1000
                m=metrics(y['test'],p,parts['test'].State.to_numpy());un=metrics(y['test'],raw,parts['test'].State.to_numpy())
                pert=cal.transform(best.predict_proba(perturb_weather(X['test'],seed)))
                item={'data_kind':'SYNTHETIC' if synthetic else 'US_ACCIDENTS_SUBSET','n_input':len(df),'mode':mode,
                    'feature_mode':feature_mode,'seed':seed,'model':name,'candidate_validation_nll':selection,'selected_trial':best_trial,
                    'train_seconds_all_candidates':elapsed,'inference_ms_per_row':latency,'serialized_bytes':len(pickle.dumps(best)),
                    'temperature':cal.temperature_,'calibrated':m,'uncalibrated':un,'macro_f1_iid_bootstrap_95':bootstrap_f1(y['test'],p,seed),
                    'weather_mean_total_variation':float(np.abs(pert-p).sum(1).mean()/2),
                    'weather_label_flip_rate':float((pert.argmax(1)!=p.argmax(1)).mean())}
                records.append(item)
                stem=f'{mode}_{seed}_{name}'
                pd.DataFrame(p,columns=['p1','p2','p3','p4']).assign(ID=parts['test'].ID.to_numpy(),y=y['test']).to_csv(out/(stem+'_predictions.csv'),index=False)
                failure_report(parts['test'],y['test'],p).to_csv(out/(stem+'_failures.csv'),index=False)
                if name=='grail':
                    prioritize(parts['test'],p).to_csv(out/(stem+'_priorities.csv'),index=False)
                    (out/(stem+'_history.json')).write_text(json.dumps(best.history_,indent=2))
                    old_temperature=best.temperature_;best.temperature_=cal.temperature_
                    best.save(out/(stem+'.pt'));best.temperature_=old_temperature
                    # Permutation sensitivity: associations/model reliance, no causal interpretation.
                    rng=np.random.default_rng(seed); rows=[]
                    for col in X['test']:
                        xx=X['test'].copy();xx[col]=rng.permutation(xx[col].to_numpy())
                        pp=cal.transform(best.predict_proba(xx))
                        rows.append({'feature':col,'nll_increase':float(log_loss(y['test'],pp,labels=[1,2,3,4])-m['nll'])})
                    pd.DataFrame(rows).sort_values('nll_increase',ascending=False).to_csv(out/(stem+'_permutation.csv'),index=False)
                (out/'results.json').write_text(json.dumps(records,indent=2))
                print(f'{mode} seed={seed} {name}: F1={m["macro_f1"]:.3f}, MAE={m["ordinal_mae"]:.3f}, NLL={m["nll"]:.3f}',flush=True)
            if all(n in predictions for n in ['grail','xgboost','corn']):
                dis=disagreement([predictions[n] for n in ['grail','xgboost','corn']],y['test'])
                for col in ['ID','State','Source','Weather_Condition']: 
                    if col in parts['test']:dis[col]=parts['test'][col].to_numpy()
                dis.sort_values('js_divergence',ascending=False).to_csv(out/f'{mode}_{seed}_disagreement.csv',index=False)
    manifest={'data_kind':'SYNTHETIC' if synthetic else 'US_ACCIDENTS_SUBSET','seeds':list(seeds),'epochs':epochs,'trials':trials,
              'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'device':'cpu','feature_mode':feature_mode,
              'warning':'Equal candidate count is not equal wall-clock compute; timings are reported. No superiority claim from synthetic data.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));return records

def figures(results,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    r=json.loads(Path(results).read_text());d=pd.DataFrame([{'mode':v['mode'],'model':v['model'],'seed':v['seed'],**{k:v['calibrated'][k] for k in ['macro_f1','ordinal_mae','nll','worst_group_macro_f1']}} for v in r])
    d.to_csv(out.parent/'results'/'summary.csv',index=False) if (out.parent/'results').exists() else None
    for metric in ['macro_f1','ordinal_mae','worst_group_macro_f1']:
        fig,ax=plt.subplots(figsize=(11,5))
        mean=d.groupby(['model','mode'])[metric].mean().unstack();std=d.groupby(['model','mode'])[metric].std().fillna(0).unstack()
        mean.plot.bar(yerr=std,ax=ax,rot=60,capsize=2);ax.set_title(f'{r[0]["data_kind"]}: {metric} (seed SD)');ax.set_ylabel(metric)
        fig.tight_layout();fig.savefig(out/(metric+'.png'),dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    for v in r:
        if v['mode']=='temporal' and v['seed']==r[0]['seed'] and v['model'] in ['grail','xgboost','corn','ordinal_dro']:
            rc=v['calibrated']['risk_coverage'];ax.plot(rc['coverage'],rc['error_rate'],label=v['model'])
    ax.set(xlabel='Coverage',ylabel='Error among accepted cases',title=r[0]['data_kind']+' selective prediction');ax.legend();fig.tight_layout();fig.savefig(out/'risk_coverage.png',dpi=150);plt.close(fig)
