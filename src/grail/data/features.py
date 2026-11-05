"""Explicit prediction-time allowlist; no automatic inclusion of new columns."""
import numpy as np
import pandas as pd

WEATHER_NUM = ['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
               'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']
WEATHER_CAT = ['Wind_Direction', 'Weather_Condition']
ROAD = ['Amenity','Bump','Crossing','Give_Way','Junction','No_Exit','Railway',
        'Roundabout','Station','Stop','Traffic_Calming','Traffic_Signal','Turning_Loop']
LIGHT = ['Sunrise_Sunset','Civil_Twilight','Nautical_Twilight','Astronomical_Twilight']
FORBIDDEN = ['Severity','End_Time','End_Lat','End_Lng','Distance(mi)',
             'Description','Duration','Accident_Duration_Min','Severe_Accident']

def features(frame, geography=True, provider=False, year=False, weather=True):
    """Mask weather recorded after start or >2 hours earlier; not a publication-time guarantee.

    Original local timestamp strings are interpreted consistently per row. No UTC
    claim is made; cross-state cutoffs use local calendar dates.
    """
    d = frame.copy()
    start = pd.to_datetime(d['Start_Time'], errors='coerce', format='mixed')
    if start.isna().any():
        raise ValueError('Invalid Start_Time: quarantine before feature extraction')
    cols = [c for c in ROAD + LIGHT for _ in [0] if c in d]
    if weather:
        lag = (start - pd.to_datetime(d.get('Weather_Timestamp', pd.Series(pd.NaT,index=d.index)),
                                     errors='coerce',format='mixed')).dt.total_seconds()/60
        valid = lag.between(0,120)
        for c in WEATHER_NUM + WEATHER_CAT:
            if c in d:
                d.loc[~valid,c] = np.nan
                cols.append(c)
    if geography:
        cols += [c for c in ['Start_Lat','Start_Lng','State'] if c in d]
    if provider and 'Source' in d: cols += ['Source']
    out = d[cols].copy()
    for name, values, period in [('hour',start.dt.hour,24),('weekday',start.dt.dayofweek,7),('month',start.dt.month-1,12)]:
        out[name+'_sin'] = np.sin(2*np.pi*values/period)
        out[name+'_cos'] = np.cos(2*np.pi*values/period)
    if year: out['Year'] = start.dt.year
    for c in ROAD:
        if c in out:
            out[c] = out[c].astype(str).str.lower().map({'true':1.,'false':0.,'1':1.,'0':0.,'1.0':1.,'0.0':0.})
    return out
