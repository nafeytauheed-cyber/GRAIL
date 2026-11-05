from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from ..data.preprocessing import preprocessor

def baseline(name,X,seed=42,weighted=False,budget=0):
    cw='balanced' if weighted else None
    if name=='majority': model=DummyClassifier(strategy='most_frequent')
    elif name=='logistic': model=LogisticRegression(C=[.1,1.][budget],max_iter=500,class_weight=cw)
    elif name=='forest': model=RandomForestClassifier(n_estimators=100,max_depth=[8,14][budget],min_samples_leaf=3,class_weight=cw,n_jobs=2,random_state=seed)
    elif name=='xgboost':
        from xgboost import XGBClassifier
        model=XGBClassifier(n_estimators=120,max_depth=[3,5][budget],learning_rate=.05,n_jobs=2,random_state=seed,objective='multi:softprob',num_class=4,tree_method='hist')
    elif name=='catboost':
        from catboost import CatBoostClassifier
        model=CatBoostClassifier(iterations=120,depth=[4,6][budget],learning_rate=.05,loss_function='MultiClass',thread_count=2,random_seed=seed,verbose=False,allow_writing_files=False)
    else: raise ValueError(name)
    return Pipeline([('preprocessor',preprocessor(X)),('model',model)])
