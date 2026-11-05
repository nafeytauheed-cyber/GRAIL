import copy
import hashlib
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from .ordinal import OrdinalNet
from ..data.preprocessing import preprocessor,clean_frame
from ..losses.groups import GroupWeights

def row_hashes(X):
    return set(pd.util.hash_pandas_object(X,index=False).astype(str))

class GRAILClassifier(ClassifierMixin,BaseEstimator):
    """Four-class ordinal learner. Only fit on prediction-time features.

    Calibration is separate and guarded against reused rows. save/load use pickle
    internally for preprocessing: load only checkpoints from trusted sources.
    """
    def __init__(self,hidden=32,epochs=30,batch_size=256,lr=.002,weight_decay=.001,
                 head='ordinal',robustness='adaptive',class_weight=None,sampling=False,
                 eta=.2,tau=50.,regularization=.1,mix=.1,patience=6,seed=42,device='cpu'):
        self.hidden=hidden; self.epochs=epochs; self.batch_size=batch_size; self.lr=lr
        self.weight_decay=weight_decay; self.head=head; self.robustness=robustness
        self.class_weight=class_weight; self.sampling=sampling; self.eta=eta; self.tau=tau
        self.regularization=regularization; self.mix=mix; self.patience=patience
        self.seed=seed; self.device=device
    def fit(self,X,y,groups=None,X_val=None,y_val=None):
        if self.epochs<1 or self.batch_size<1 or self.hidden<1 or self.patience<1 or self.lr<=0: raise ValueError('Invalid training configuration')
        if self.class_weight not in [None,'balanced']: raise ValueError('class_weight must be None or balanced')
        if self.sampling and (self.robustness!='erm' or self.class_weight is not None):
            raise ValueError('Controlled sampling is only supported with unweighted ERM')
        X=clean_frame(X); y=np.asarray(y)
        if y.ndim!=1 or len(X)!=len(y) or not np.isin(y,[1,2,3,4]).all() or len(X)==0: raise ValueError('Require nonempty X and integer labels 1..4')
        if set(y)!={1,2,3,4}: raise ValueError('All four classes required in training; inspect subset compatibility')
        if (X_val is None)!=(y_val is None): raise ValueError('Provide both validation arrays')
        self.columns_=list(X.columns); self.classes_=np.arange(1,5); self.n_features_in_=X.shape[1]
        self.training_hashes_=row_hashes(X); self.validation_hashes_=set()
        self.preprocessor_=preprocessor(X)
        a=self.preprocessor_.fit_transform(X).astype('float32')
        if X_val is not None:
            X_val=clean_frame(X_val,self.columns_); self.validation_hashes_=row_hashes(X_val)
            if self.training_hashes_ & self.validation_hashes_: raise ValueError('Training/validation duplicate feature rows')
            y_val=np.asarray(y_val)
            if len(y_val)!=len(X_val) or not np.isin(y_val,[1,2,3,4]).all(): raise ValueError('Invalid validation labels')
            av=self.preprocessor_.transform(X_val).astype('float32')
        if groups is None:
            if self.robustness!='erm': raise ValueError('Geographic groups required for robust learning')
            groups=np.zeros(len(y),dtype=int)
        groups=np.asarray(groups)
        if len(groups)!=len(y) or pd.isna(groups).any(): raise ValueError('Invalid groups')
        self.groups_,g=np.unique(groups.astype(str),return_inverse=True)
        counts=np.bincount(g)
        gw=GroupWeights(counts,self.robustness,self.eta,self.tau,self.regularization,self.mix)
        torch.manual_seed(self.seed)
        if str(self.device).startswith('cuda') and not torch.cuda.is_available(): raise ValueError('CUDA requested but unavailable')
        self.network_=OrdinalNet(a.shape[1],self.hidden,self.head).to(self.device)
        if self.head in ['ordinal','coral']:
            # Empirical cumulative priors give wide middle-class intervals immediately.
            # Entirely training-only; shared identically by every cumulative-link variant.
            q=np.array([(y>k).mean() for k in [1,2,3]]).clip(1e-4,1-1e-4)
            thresholds=-np.log(q/(1-q));gaps=np.maximum(np.diff(thresholds)-1e-4,1e-4)
            inverse_softplus=gaps+np.log(-np.expm1(-gaps))
            with torch.no_grad():
                self.network_.base.copy_(torch.tensor(thresholds[0],device=self.device))
                self.network_.gaps.copy_(torch.as_tensor(inverse_softplus,device=self.device))
                self.network_.output.bias.zero_()

        opt=torch.optim.AdamW(self.network_.parameters(),lr=self.lr,weight_decay=self.weight_decay)
        class_counts=np.bincount(y.astype(int),minlength=5)[1:]
        cw=len(y)/(4*class_counts) if self.class_weight=='balanced' else np.ones(4)
        generator=torch.Generator().manual_seed(self.seed)
        sampler=WeightedRandomSampler(torch.tensor(1/class_counts[y.astype(int)-1]),len(y),replacement=True,generator=generator) if self.sampling else None
        loader=DataLoader(TensorDataset(torch.from_numpy(a),torch.tensor(y,dtype=torch.long),torch.tensor(g)),
                          batch_size=self.batch_size,shuffle=sampler is None,sampler=sampler,generator=generator)
        self.history_=[]; best=np.inf; stale=0; best_state=None
        # CORN original denominator is the total number of eligible binary tasks.
        corn_scale=1/np.mean(np.minimum(y,3)) if self.head=='corn' else 1.
        for epoch in range(self.epochs):
            self.network_.train()
            coeff=gw.effective_weights()/(counts/len(y))
            for xb,yb,gb in loader:
                xb,yb=xb.to(self.device),yb.to(self.device)
                losses=self.network_.losses(self.network_(xb),yb)*corn_scale
                weight=torch.as_tensor(coeff[gb.numpy()]*cw[yb.cpu().numpy()-1],dtype=torch.float32,device=self.device)
                loss=(losses*weight).mean()
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(self.network_.parameters(),5.);opt.step()
            self.network_.eval()
            per=self._loss_array(a,y)*corn_scale*cw[y.astype(int)-1]
            gl=np.bincount(g,weights=per,minlength=len(counts))/counts
            gw.update(gl)
            score=float(self._loss_array(av,y_val).mean()) if X_val is not None else float(np.dot(gw.weights,gl))
            self.history_.append({'epoch':epoch+1,'selection_loss':score,'group_loss':gl.tolist(),'weights':gw.weights.tolist()})
            if score<best-1e-6:
                best=score;stale=0;best_state=copy.deepcopy(self.network_.state_dict());self.best_epoch_=epoch+1
            else: stale+=1
            if X_val is not None and stale>=self.patience: break
        self.network_.load_state_dict(best_state);self.network_.eval()
        self.group_weights_=gw.weights;self.temperature_=1.
        return self
    def _loss_array(self,a,y):
        out=[]
        with torch.inference_mode():
            for i in range(0,len(a),self.batch_size):
                z=self.network_(torch.as_tensor(a[i:i+self.batch_size],device=self.device))
                out.append(self.network_.losses(z,torch.as_tensor(y[i:i+self.batch_size],dtype=torch.long,device=self.device)).cpu().numpy())
        return np.concatenate(out)
    def predict_proba(self,X):
        check_is_fitted(self,'network_'); X=clean_frame(X,self.columns_)
        if len(X)==0: return np.empty((0,4))
        out=[];self.network_.eval()
        with torch.inference_mode():
            for i in range(0,len(X),self.batch_size):
                a=self.preprocessor_.transform(X.iloc[i:i+self.batch_size]).astype('float32')
                z=self.network_(torch.as_tensor(a,device=self.device))
                p=self.network_.probabilities(z).cpu().numpy().clip(1e-12,1)
                logp=np.log(p)/self.temperature_;logp-=logp.max(1,keepdims=True)
                p=np.exp(logp);out.append(p/p.sum(1,keepdims=True))
        return np.concatenate(out)
    def predict(self,X): return self.classes_[self.predict_proba(X).argmax(1)]
    def calibrate(self,X,y):
        from ..evaluation.calibration import TemperatureCalibrator
        check_is_fitted(self,'network_'); X=clean_frame(X,self.columns_)
        if row_hashes(X)&(self.training_hashes_|self.validation_hashes_):
            raise ValueError('Calibration must use separate held-out rows')
        self.temperature_=1.
        self.calibrator_=TemperatureCalibrator().fit(self.predict_proba(X),y)
        self.temperature_=self.calibrator_.temperature_
        return self
    def save(self,path):
        import joblib
        check_is_fitted(self,'network_')
        payload={k:v for k,v in self.__dict__.items() if k!='network_'}
        payload['state_dict']={k:v.cpu() for k,v in self.network_.state_dict().items()}
        payload['input_dim']=self.network_.encoder[0].in_features
        joblib.dump(payload,path)
    @classmethod
    def load(cls,path,device='cpu'):
        import joblib
        payload=joblib.load(path);state=payload.pop('state_dict');d=payload.pop('input_dim')
        obj=cls();obj.__dict__.update(payload);obj.device=device
        obj.network_=OrdinalNet(d,obj.hidden,obj.head).to(device)
        obj.network_.load_state_dict(state);obj.network_.eval()
        return obj
