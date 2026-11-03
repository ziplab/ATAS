"""Exact incremental voxel membership; immutable base plus merge levels."""
import numpy as np
import torch

def _keys(values,device):
    return torch.as_tensor(np.asarray(values,dtype=np.int64),device=device,dtype=torch.int64)

def unique_first(values,device):
    keys=_keys(values,device)
    unique,inverse=torch.unique(keys,sorted=True,return_inverse=True)
    first=torch.full((len(unique),),len(keys),device=device,dtype=torch.int64)
    first.scatter_reduce_(0,inverse,torch.arange(len(keys),device=device),reduce='amin',include_self=True)
    return first.cpu().numpy(),unique

def _contains(sorted_keys,queries):
    if not len(sorted_keys):return torch.zeros(len(queries),device=queries.device,dtype=torch.bool)
    position=torch.searchsorted(sorted_keys,queries)
    return (position<len(sorted_keys))&(sorted_keys[position.clamp(max=len(sorted_keys)-1)]==queries)

class IncrementalKeys:
    def __init__(self,base=(),*,device=None):
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.base=torch.unique(_keys(base,self.device),sorted=True)
        self.levels=[]
    def contains_tensor(self,values):
        queries=values if isinstance(values,torch.Tensor) else _keys(values,self.device)
        found=_contains(self.base,queries)
        for level in self.levels:
            if level is not None:found|=_contains(level,queries)
        return found
    def contains(self,values):return self.contains_tensor(values).cpu().numpy()
    def add(self,values):
        keys=values if isinstance(values,torch.Tensor) else _keys(values,self.device)
        keys=torch.unique(keys,sorted=True)
        keys=keys[~self.contains_tensor(keys)]
        if not len(keys):return
        index=0
        while index<len(self.levels) and self.levels[index] is not None:
            keys=torch.unique(torch.cat((self.levels[index],keys)),sorted=True)
            self.levels[index]=None;index+=1
        if index==len(self.levels):self.levels.append(keys)
        else:self.levels[index]=keys

class TemporalSupport:
    def __init__(self,bins=3,*,device=None):
        self.bins=[IncrementalKeys(device=device) for _ in range(bins)]
    def add(self,values,index):self.bins[index].add(values)
    def mask(self,values,min_bins=2,chunk_size=500000):
        output=[]
        for start in range(0,len(values),chunk_size):
            keys=_keys(values[start:start+chunk_size],self.bins[0].device)
            votes=torch.zeros(len(keys),device=keys.device,dtype=torch.int16)
            for index in self.bins:votes+=index.contains_tensor(keys).to(torch.int16)
            output.append((votes>=min_bins).cpu().numpy())
        return np.concatenate(output) if output else np.zeros(0,bool)
