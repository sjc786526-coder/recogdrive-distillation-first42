"""Original per-scene RNG protocol: no dependence on rank, batching or Python hash()."""
import hashlib,random

ROOT_SEED=20261008
SEED_INDICES=[0,1,2,3,4]
TIMESTEPS=[80,60,40,20,0]
def scene_seed(token,j=0):
    return int(hashlib.sha256(f'{token}|{ROOT_SEED}|{j}'.encode('utf-8')).hexdigest()[:8],16)
def reseed(value):
    import numpy as np
    import torch
    random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)
def noise_bank(token,j=0,device='cuda'):
    """Reference generator. The planner makes these six calls in the same order.

    First [1,8,3] draw initializes actions; next five draws are step noises.
    Step noise is clipped to [-1,1] by the planner, not in this raw bank.
    Fixed RNG does not promise cross-device or cross-version bitwise identity.
    """
    import torch
    generator=torch.Generator(device=device);generator.manual_seed(scene_seed(token,j))
    return torch.stack([torch.randn((1,8,3),generator=generator,device=device,dtype=torch.float32) for _ in range(6)])
