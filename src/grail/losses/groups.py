"""Epoch-level adversarial updates avoid treating absent minibatch groups as easy."""
import numpy as np
from scipy.special import softmax

class GroupWeights:
    def __init__(self,counts,mode='adaptive',eta=.2,tau=50.,regularization=.1,mix=.1):
        self.counts=np.asarray(counts,dtype=float)
        if self.counts.ndim!=1 or len(self.counts)==0 or np.any(self.counts<=0): raise ValueError('Positive group counts required')
        if mode not in ['erm','fixed','dro','adaptive']: raise ValueError('Unknown weighting mode')
        if eta<=0 or tau<0 or regularization<0 or not 0<=mix<1: raise ValueError('Invalid robustness hyperparameters')
        self.mode,self.eta,self.tau,self.regularization,self.mix=mode,eta,tau,regularization,mix
        self.prior=self.counts/self.counts.sum()
        self.weights=self.prior.copy() if mode=='erm' else np.ones(len(counts))/len(counts)
        self.history=[]
    def update(self,losses):
        l=np.asarray(losses,dtype=float)
        if l.shape!=self.counts.shape or not np.isfinite(l).all(): raise ValueError('One finite loss per group required')
        if self.mode=='dro':
            self.weights=softmax(np.log(self.weights.clip(1e-30))+self.eta*l)
        elif self.mode=='adaptive':
            r=self.counts/(self.counts+self.tau)
            shrunk=r*l+(1-r)*np.dot(self.prior,l)
            # KL-regularized mirror ascent with uniform prior u; uniform floor.
            score=(np.log(self.weights.clip(1e-30))+self.eta*shrunk)/(1+self.eta*self.regularization)
            self.weights=(1-self.mix)*softmax(score)+self.mix/len(l)
        self.history.append(self.weights.tolist())
        return self.weights.copy()

    def effective_weights(self):
        """Gradient weights for the same shrunken-loss objective used by adversary."""
        if self.mode != 'adaptive': return self.weights.copy()
        r=self.counts/(self.counts+self.tau)
        return self.weights*r+np.dot(self.weights,1-r)*self.prior
