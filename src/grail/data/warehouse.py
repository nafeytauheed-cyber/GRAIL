"""Disk-backed CSV conversion/auditing; no full-data pandas concatenation."""
from pathlib import Path
import pandas as pd
import json
import duckdb

def connection():
    c=duckdb.connect();c.execute("SET memory_limit='2GB'");c.execute('SET threads=2');return c

def prepare(csv,out):
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    c=connection();c.read_csv(str(csv),all_varchar=True).create_view('raw')
    c.sql('SELECT * FROM raw').write_parquet(str(out),compression='zstd')
    c.close()

def audit(parquet,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    c=connection();c.read_parquet(str(parquet)).create_view('raw')
    columns=[row[0] for row in c.execute('DESCRIBE raw').fetchall()]
    c.execute('CREATE VIEW d AS SELECT *, try_cast(Start_Time AS TIMESTAMP) AS start_ts, try_cast(Severity AS INTEGER) AS label FROM raw')
    n=c.execute('SELECT count(*) FROM d').fetchone()[0]
    report={'rows':n,'invalid_start':c.execute('SELECT count(*) FROM d WHERE start_ts IS NULL').fetchone()[0],
            'invalid_target':c.execute('SELECT count(*) FROM d WHERE label IS NULL OR label NOT BETWEEN 1 AND 4').fetchone()[0],
            'missing':{},'label_change_status':'UNVERIFIED: prevalence shifts do not prove reclassification'}
    for col in columns:
        q='"'+col.replace('"','""')+'"'
        report['missing'][col]=c.execute(f"SELECT count(*) FROM d WHERE {q} IS NULL OR cast({q} AS VARCHAR)='' ").fetchone()[0]
    for name,group in [('severity','label'),('year','year(start_ts), label'),('state','State,label'),('provider','Source,label'),('year_source','year(start_ts),Source,label'),
                       ('year_source_state','year(start_ts),Source,State,label')]:
        c.execute(f'SELECT {group}, count(*) AS n FROM d GROUP BY {group} ORDER BY {group}').df().to_csv(out/(name+'.csv'),index=False)
    report['duplicate_ids']=c.execute('SELECT count(*)-count(DISTINCT ID) FROM d WHERE ID IS NOT NULL').fetchone()[0]
    report['duplicate_signatures']=c.execute('SELECT coalesce(sum(n-1),0) FROM (SELECT count(*) n FROM d WHERE start_ts IS NOT NULL AND Start_Lat IS NOT NULL AND Start_Lng IS NOT NULL GROUP BY start_ts,Start_Lat,Start_Lng)').fetchone()[0]
    if 'End_Time' in columns:
        report['negative_duration']=c.execute('SELECT count(*) FROM d WHERE try_cast(End_Time AS TIMESTAMP)<start_ts').fetchone()[0]
    if 'Weather_Timestamp' in columns:
        report['future_weather']=c.execute('SELECT count(*) FROM d WHERE try_cast(Weather_Timestamp AS TIMESTAMP)>start_ts').fetchone()[0]
        report['weather_older_than_2h']=c.execute("SELECT count(*) FROM d WHERE try_cast(Weather_Timestamp AS TIMESTAMP)<start_ts-INTERVAL '2 hours'").fetchone()[0]
    table=c.execute('SELECT year(start_ts) AS year, Source, State, label, count(*) n FROM d GROUP BY ALL').df()
    table['prevalence']=table.n/table.groupby(['year','Source','State']).n.transform('sum')
    table.to_csv(out/'conditional_label_prevalence.csv',index=False)
    # Within-stratum total variation distance between successive years, with counts.
    import numpy as np
    shifts=[]
    for (source,state),sub in table.groupby(['Source','State']):
        p=sub.pivot_table(index='year',columns='label',values='n',fill_value=0).reindex(columns=[1,2,3,4],fill_value=0).sort_index()
        for i in range(1,len(p)):
            a,b=p.iloc[i-1].to_numpy(),p.iloc[i].to_numpy()
            shifts.append({'source':source,'state':state,'year_before':int(p.index[i-1]),'year_after':int(p.index[i]),
                           'n_before':int(a.sum()),'n_after':int(b.sum()),'total_variation':float(np.abs(a/a.sum()-b/b.sum()).sum()/2)})
    import pandas as pd
    pd.DataFrame(shifts).to_csv(out/'within_stratum_shift.csv',index=False)
    # Descriptive feature drift summaries, not fitted model transformations.
    from .features import WEATHER_NUM
    for grouping,name in [('year(start_ts)','year'),('State','state')]:
        expr=[]
        for col in WEATHER_NUM:
            if col not in columns: continue
            q='"'+col.replace('"','""')+'"'
            expr += [f'avg(try_cast({q} AS DOUBLE)) AS "{col}_mean"',
                     f'stddev_pop(try_cast({q} AS DOUBLE)) AS "{col}_sd"',
                     f'avg(CASE WHEN try_cast({q} AS DOUBLE) IS NULL THEN 1.0 ELSE 0.0 END) AS "{col}_missing_rate"']
        if expr:
            c.execute(f'SELECT {grouping} AS stratum, count(*) AS n, '+','.join(expr)+f' FROM d GROUP BY {grouping}').df().to_csv(out/f'{name}_weather_shift.csv',index=False)
    # Standardize annual prevalence over source/state strata observed in all years.
    # This is descriptive and must never select features using test labels.
    joint=table.groupby(['year','Source','State']).n.sum().reset_index()
    years=joint.year.dropna().unique();present=joint.groupby(['Source','State']).year.nunique()
    common=present[present==len(years)].index
    standardized=[]
    if len(common):
        reference=joint[joint.year==min(years)].set_index(['Source','State']).loc[common,'n']
        reference=reference/reference.sum()
        for year in sorted(years):
            sub=table[table.year==year].pivot_table(index=['Source','State'],columns='label',values='n',fill_value=0).reindex(index=common,columns=[1,2,3,4],fill_value=0)
            prevalence=sub.div(sub.sum(axis=1),axis=0)
            values=prevalence.mul(reference,axis=0).sum(axis=0)
            standardized.append({'year':int(year),'common_strata':len(common),**{f'p{j}':float(values[j]) for j in range(1,5)}})
    pd.DataFrame(standardized).to_csv(out/'mixture_standardized_prevalence.csv',index=False)
    report['common_source_state_strata']=len(common)
    (out/'audit.json').write_text(json.dumps(report,indent=2));c.close();return report

def load_development_sample(parquet,limit=50000,seed=42,source=None,start=None,end=None):
    """Deterministic hash sample after optional predeclared label-compatible filtering.
    No target stratification. Full audit remains separate from this bounded sample.
    """
    c=connection();c.read_parquet(str(parquet)).create_view('raw')
    filters=['try_cast(Severity AS INTEGER) BETWEEN 1 AND 4','try_cast(Start_Time AS TIMESTAMP) IS NOT NULL'];args=[]
    for clause,value in [('Source = ?',source),('Start_Time >= ?',start),('Start_Time < ?',end)]:
        if value is not None: filters.append(clause);args.append(value)
    c.execute('CREATE TEMP TABLE sample AS SELECT * FROM raw WHERE '+' AND '.join(filters)+f' ORDER BY hash(ID, {int(seed)}) LIMIT {int(limit)}',args)
    d=c.sql('SELECT * FROM sample').df();c.close()
    from .features import WEATHER_NUM
    for col in WEATHER_NUM+['Severity','Start_Lat','Start_Lng','Distance(mi)']:
        if col in d: d[col]=pd.to_numeric(d[col],errors='coerce')
    return d
