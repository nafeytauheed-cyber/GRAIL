"""Summarize only completed artifacts, verify calibrated checkpoints, render figures."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from grail import GRAILClassifier
from grail.data.synthetic import synthetic
from grail.data.splits import split_data
from grail.data.features import features
from grail.training.experiment import figures

root=Path('reports/results/synthetic');r=json.loads((root/'results.json').read_text())
assert len(r)==90, f'Expected 15 variants x 3 seeds x 2 protocols; found {len(r)}'
figures(root/'results.json','reports/figures')
# Ensure checkpoints made by earlier runner version retain evaluated calibration.
for v in r:
    if v['model']!='grail':continue
    p=root/f'{v["mode"]}_{v["seed"]}_grail.pt'
    model=GRAILClassifier.load(p);model.temperature_=v['temperature'];model.save(p)
    test=split_data(synthetic(8000),v['mode'])['test']
    pred=model.predict_proba(features(test))
    saved=pd.read_csv(root/f'{v["mode"]}_{v["seed"]}_grail_predictions.csv')[['p1','p2','p3','p4']].to_numpy()
    np.testing.assert_allclose(pred,saved,rtol=1e-5,atol=1e-6)
rows=[]
for v in r:
    rows.append({'mode':v['mode'],'model':v['model'],'seed':v['seed'],**{k:v['calibrated'][k] for k in ['macro_f1','balanced_accuracy','ordinal_mae','nll','brier','worst_group_macro_f1']},
                 'class4_recall':v['calibrated']['per_class']['4']['recall']})
d=pd.DataFrame(rows);summary=d.groupby(['mode','model']).agg(['mean','std']);summary.to_csv('reports/results/aggregate.csv')
lines=['# Executed results — SYNTHETIC ONLY','',
       '8,000 generated rows; three model seeds (11, 23, 37); 15 variants; two protocols; 90 selected model evaluations. Each non-majority family tries two validation-selected candidates. All cumulative-link models use training-prior threshold initialization.',
       '', 'These results validate software execution and study this artificial generator. They are not evidence of US Accidents performance or road-safety conclusions.','']
for mode in ['temporal','geographic']:
    lines += ['## '+mode,'','| Model | Macro-F1 mean ± seed SD | Ordinal MAE | NLL | Worst-state F1 | Grade-4 recall |','|---|---:|---:|---:|---:|---:|']
    for name in d.model.unique():
        z=d[(d['mode']==mode)&(d.model==name)]
        lines.append(f'| {name} | {z.macro_f1.mean():.3f} ± {z.macro_f1.std():.3f} | {z.ordinal_mae.mean():.3f} | {z.nll.mean():.3f} | {z.worst_group_macro_f1.mean():.3f} | {z.class4_recall.mean():.3f} |')
    lines += ['']
lines+=['## Interpretation limits','',
        'Seed SD measures training randomness on one fixed dataset, not population confidence. Temporal test and geographic test have different populations; their score difference does not isolate a geographic causal effect. Consult split JSONs for sample sizes and all four class supports.',
        '', 'An interrupted earlier run with narrow initial threshold gaps is preserved under `reports/results/initialization_diagnostic/`. It is not part of this final comparison. Training-only prior initialization was introduced after diagnosing underfitting; this is an exploratory development sequence, not a pristine preregistered confirmatory experiment.',
        '', 'GRAIL is not assumed to win. Compare it with ordinary DRO and CORAL on every metric, especially worst-state performance, probability scores and rare-grade recall. Real-data label audit, tuning and evaluation remain necessary.']
Path('docs/results_summary.md').write_text('\n'.join(lines)+'\n')
fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,mode in zip(axes,['temporal','geographic']):
    v=next(v for v in r if v['model']=='grail' and v['seed']==11 and v['mode']==mode)
    cm=np.array(v['calibrated']['confusion_matrix']);ax.imshow(cm,cmap='Blues')
    for i in range(4):
        for j in range(4):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='red')
    ax.set(xticks=range(4),yticks=range(4),xticklabels=range(1,5),yticklabels=range(1,5),xlabel='Predicted disruption grade',ylabel='True grade',title=mode+' (synthetic)')
fig.tight_layout();fig.savefig('reports/figures/confusion_matrices.png',dpi=150);plt.close(fig)
fig,ax=plt.subplots(figsize=(6,5));ax.plot([0,1],[0,1],'--',color='gray')
for name in ['grail','xgboost','corn']:
    v=pd.read_csv(root/f'temporal_11_{name}_predictions.csv');p=v[['p1','p2','p3','p4']].to_numpy();confidence=p.max(1);correct=p.argmax(1)+1==v.y.to_numpy();xs=[];ys=[]
    for i in range(10):
        mask=(confidence>=i/10)&(confidence<((i+1)/10) if i<9 else confidence<=1)
        if mask.any():xs.append(confidence[mask].mean());ys.append(correct[mask].mean())
    ax.plot(xs,ys,'o-',label=name)
ax.set(xlabel='Mean confidence',ylabel='Observed correctness',title='SYNTHETIC: calibrated top-label reliability');ax.legend();fig.tight_layout();fig.savefig('reports/figures/reliability.png',dpi=150);plt.close(fig)
fig,ax=plt.subplots(figsize=(6,4));s=synthetic(8000).Severity.value_counts().sort_index();ax.bar(s.index,s.values);ax.set(xticks=[1,2,3,4],xlabel='Disruption grade',ylabel='Generated rows',title='SYNTHETIC fixture class imbalance');fig.tight_layout();fig.savefig('reports/figures/class_distribution.png',dpi=150);plt.close(fig)
print('Verified six calibrated checkpoints and generated seven figures.')

# Extra shortcut comparisons and compact interpretation are regenerated as well.
shortcut_rows=[]
for path in Path('reports/results').glob('shortcut_*/results.json'):
    for v in json.loads(path.read_text()):
        shortcut_rows.append({'feature_mode':v['feature_mode'],'mode':v['mode'],'model':v['model'],**{k:v['calibrated'][k] for k in ['macro_f1','ordinal_mae','nll']}})
if shortcut_rows:
    pd.DataFrame(shortcut_rows).sort_values(['mode','model','feature_mode']).to_csv('reports/results/shortcut_summary.csv',index=False)
extra=['','## Measured comparison','']
for mode in ['temporal','geographic']:
    a=d[(d['mode']==mode)&(d.model=='grail')];b=d[(d['mode']==mode)&(d.model=='ordinal_dro')]
    extra.append(f'{mode}: GRAIL / ordinary DRO mean macro-F1 {a.macro_f1.mean():.3f} / {b.macro_f1.mean():.3f}; NLL {a.nll.mean():.3f} / {b.nll.mean():.3f}; worst-state F1 {a.worst_group_macro_f1.mean():.3f} / {b.worst_group_macro_f1.mean():.3f}.')
    info=json.loads((root/f'{mode}_splits.json').read_text())['test']
    extra.append(f'Test population: {info["n"]} synthetic rows, grade-4 count {info["classes"]["4"]}.')
extra += ['',f'{len(shortcut_rows)} additional one-seed shortcut evaluations are recorded in reports/results/shortcut_summary.csv. The adaptive objective has not established superiority; tiny holdouts and optimizer variability prevent strong conclusions.']
p=Path('docs/results_summary.md');p.write_text(p.read_text()+'\n'.join(extra)+'\n')
