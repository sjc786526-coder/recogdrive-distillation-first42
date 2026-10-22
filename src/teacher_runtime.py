"""Verbatim first42 fits, observation gate and actual-forward capture."""
from d4_common import *
from functools import lru_cache

class FitBank:
    def __init__(self):self.data={}
    def get(self,root,arm):
        import numpy as np
        import torch
        path=Path(root)/arm['fit_path'];key=(str(path),arm['fit_sha256'])
        if key not in self.data:
            assert sha(path)==key[1]
            with np.load(path,allow_pickle=False) as z:
                self.data[key]={k:torch.as_tensor(np.array(z[k]),device='cuda',dtype=torch.float32) for k in z.files}
        return self.data[key]


@lru_cache(maxsize=8)
def gate_fit(path,hashed):
    import numpy as np
    assert sha(path)==hashed
    with np.load(path,allow_pickle=False) as z:return {k:np.asarray(z[k],dtype=np.float64) for k in ['center','basis','mh','mf','cov_inv_s','cov_inv_f']}


def observation_gate(engine,root,arm):
    import numpy as np
    from vlm_core import VISION_GROUPS,LANGUAGE_GROUPS
    gate=arm['observation_gate'];fit=gate['fit'];block=fit['block'];group=fit['group']
    matrix,_=engine.vm if block.startswith('vision') else engine.bm if block.startswith('bridge') else engine.lm
    names=VISION_GROUPS if block.startswith(('vision','bridge')) else LANGUAGE_GROUPS
    value=(matrix[names.index(group)]@engine.prefix[block].reshape(-1,engine.prefix[block].shape[-1]).float()).detach().cpu().numpy().astype(np.float64)
    z=gate_fit(str(Path(root)/fit['path']),fit['sha256']);p=(value-z['center'])@z['basis'];ds=p-z['mh'];df=p-z['mf']
    risk=.5*(ds@z['cov_inv_s']@ds-df@z['cov_inv_f']@df)
    return bool(np.isfinite(risk) and risk>gate['threshold']),float(risk)


def prediction(engine,features,hidden,j,token,interface=None,gradient=False,capture=True):
    import torch
    from common import reseed
    observed={}
    def hook(module,args,out):
        value=interface if interface is not None else out
        if capture:observed['interface']=value.detach().clone()
        return value
    handle=engine.agent.action_head.feature_encoder.register_forward_hook(hook)
    try:
        batch={k:v.unsqueeze(0).cuda().float() for k,v in features.items() if k not in ['pixel_values','last_hidden_state']}
        batch['last_hidden_state']=hidden;reseed(seed(token,j))
        with torch.set_grad_enabled(gradient),torch.autocast('cuda',enabled=False):
            value=engine.agent.forward(batch)['pred_traj'].squeeze(0)
        if capture:assert 'interface' in observed
        return value,observed.get('interface')
    finally:handle.remove()

