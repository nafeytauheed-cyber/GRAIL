"""Artificial software-validation fixture, not US Accidents evidence."""
import numpy as np
import pandas as pd

def synthetic(n=8000,seed=91):
    r=np.random.default_rng(seed)
    t=pd.date_range('2016-02-01','2023-03-31',periods=n)
    states=r.choice(['AZ','FL','NY','WA','NV','OR','CA','TX'],n)
    rain=r.exponential(.03,n);wind=np.maximum(0,r.normal(8,4,n));vis=np.clip(10-60*rain+r.normal(0,1,n),.1,15)
    sig=r.random(n)<.3;source=r.choice(['Source1','Source2','Source3'],n)
    z=1.1*sig+.14*wind-.3*vis+7*rain+r.logistic(0,.65,n)
    z+=np.isin(states,['CA','TX'])*.4
    y=np.digitize(z,[-3,1.3,2.8])+1
    d=pd.DataFrame({'ID':['SYNTH-'+str(i) for i in range(n)],'Start_Time':t.astype(str),
        'End_Time':(t+pd.to_timedelta(10+30*y,unit='m')).astype(str),
        'Weather_Timestamp':(t-pd.Timedelta(minutes=30)).astype(str),'State':states,'Source':source,
        'Start_Lat':r.uniform(25,49,n),'Start_Lng':r.uniform(-124,-70,n),'Severity':y,
        'Temperature(F)':r.normal(60,15,n),'Humidity(%)':np.clip(r.normal(65,15,n),0,100),
        'Wind_Speed(mph)':wind,'Visibility(mi)':vis,'Precipitation(in)':rain,
        'Traffic_Signal':sig,'Junction':r.random(n)<.1,'Sunrise_Sunset':np.where((t.hour>=6)&(t.hour<18),'Day','Night'),
        'Weather_Condition':np.where(rain>.04,'Rain','Fair'),'Distance(mi)':y*.7+r.random(n)})
    d.loc[r.random(n)<.05,'Temperature(F)']=np.nan
    return d
