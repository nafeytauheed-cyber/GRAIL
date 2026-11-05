import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax

class TemperatureCalibrator:
    def fit(self,p,y):
        p=np.asarray(p); y=np.asarray(y)
        if p.ndim!=2 or p.shape!=(len(y),4) or len(y)==0 or not np.isin(y,[1,2,3,4]).all(): raise ValueError('Invalid calibration arrays')
        if not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1): raise ValueError('Invalid probabilities')
        def objective(t):
            q=softmax(np.log(p.clip(1e-12))/np.exp(t),axis=1)
            return -np.log(q[np.arange(len(y)),y.astype(int)-1].clip(1e-12)).mean()
        fit=minimize_scalar(objective,bounds=(-3,3),method='bounded')
        self.temperature_=float(np.exp(fit.x)) if fit.fun<objective(0.) else 1.
        return self
    def transform(self,p):
        return softmax(np.log(np.asarray(p).clip(1e-12))/self.temperature_,axis=1)
