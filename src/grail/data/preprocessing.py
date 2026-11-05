import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def preprocessor(X):
    numeric = list(X.select_dtypes(include=['number','bool']).columns)
    categorical = [c for c in X if c not in numeric]
    return ColumnTransformer([
        ('numeric',make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler()),numeric),
        ('categorical',make_pipeline(SimpleImputer(strategy='constant',fill_value='__MISSING__'),
           OneHotEncoder(handle_unknown='ignore',max_categories=32,sparse_output=False,dtype=np.float32)),categorical),
    ],sparse_threshold=0)

def clean_frame(X, columns=None):
    if not isinstance(X,pd.DataFrame):
        a=np.asarray(X)
        if a.ndim != 2: raise ValueError('X must be a 2D array or DataFrame')
        X=pd.DataFrame(a,columns=columns)
    if X.columns.duplicated().any(): raise ValueError('Duplicate feature names')
    if columns is not None and list(X.columns)!=list(columns):
        raise ValueError('Feature columns/order differ from training')
    X=X.copy().replace([np.inf,-np.inf],np.nan)
    for c in X.select_dtypes(exclude=['number','bool']):
        X[c]=X[c].map(lambda v: str(v) if pd.notna(v) else np.nan).astype(object)
    return X
