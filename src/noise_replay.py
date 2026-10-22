"""Optional explicit initial/step-noise replay, scoped to a single planner forward."""
from contextlib import contextmanager

@contextmanager
def replay_bank(bank):
    import torch
    old_randn,old_like=torch.randn,torch.randn_like;calls=[0,0]
    def initial(*shape,**kwargs):
        requested=tuple(shape[0]) if len(shape)==1 and isinstance(shape[0],(tuple,list)) else tuple(shape)
        if calls[0] or requested!=tuple(bank.shape[1:]):raise AssertionError(('Unexpected initial RNG call',requested,calls))
        calls[0]+=1
        return torch.as_tensor(bank[0],device=kwargs.get('device','cuda'),dtype=kwargs.get('dtype',torch.float32)).clone()
    def step(value,*args,**kwargs):
        if calls[1]>=5 or tuple(value.shape)!=tuple(bank.shape[1:]):raise AssertionError(('Unexpected step RNG call',tuple(value.shape),calls))
        calls[1]+=1;return torch.as_tensor(bank[calls[1]],device=value.device,dtype=value.dtype).clone()
    torch.randn,torch.randn_like=initial,step
    try:
        yield
        if calls!=[1,5]:raise AssertionError(('Noise call count differs from original protocol',calls))
    finally:torch.randn,torch.randn_like=old_randn,old_like
