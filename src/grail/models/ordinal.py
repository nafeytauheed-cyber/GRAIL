"""Independently implemented cumulative-link and conditional ordinal heads."""
import torch
from torch import nn
from torch.nn import functional as F

def ordinal_targets(y):
    if y.ndim!=1 or not torch.all((y>=1)&(y<=4)&(y==y.long())):
        raise ValueError('Severity labels must be integers 1..4')
    return (y[:,None]>torch.arange(1,4,device=y.device)).float()

def cumulative_to_probs(q):
    return torch.cat([1-q[:,:1],q[:,:-1]-q[:,1:],q[:,-1:]],dim=1)

class OrdinalNet(nn.Module):
    def __init__(self,d,hidden=32,head='ordinal'):
        super().__init__()
        if head not in ['ordinal','coral','corn','ce']: raise ValueError('Unknown head')
        self.head=head
        self.encoder=nn.Sequential(nn.Linear(d,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU())
        self.output=nn.Linear(hidden,4 if head=='ce' else 3 if head=='corn' else 1)
        if head in ['ordinal','coral']:
            self.base=nn.Parameter(torch.tensor(-1.))
            self.gaps=nn.Parameter(torch.zeros(2))
    def thresholds(self):
        return torch.cat([self.base[None],self.base+torch.cumsum(F.softplus(self.gaps)+1e-4,0)])
    def forward(self,x):
        z=self.output(self.encoder(x))
        return z-self.thresholds()[None,:] if self.head in ['ordinal','coral'] else z
    def probabilities(self,z):
        if self.head=='ce': return z.softmax(1)
        q=z.sigmoid()
        if self.head=='corn': q=q.cumprod(1)
        return cumulative_to_probs(q)
    def losses(self,z,y):
        if self.head=='ce': return F.cross_entropy(z,y-1,reduction='none')
        bce=F.binary_cross_entropy_with_logits(z,ordinal_targets(y),reduction='none')
        if self.head=='corn':
            # Conditional risk sets Y>k-1. Global normalization handled by trainer.
            mask=y[:,None]>torch.arange(0,3,device=y.device)
            return (bce*mask).sum(1)
        return bce.mean(1)
