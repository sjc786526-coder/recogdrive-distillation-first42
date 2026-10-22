"""Admit DDP's initial parameter broadcast by exact frozen-value checks."""
from d4_common import *


def frozen_hash(student):
    import torch
    result=hashlib.sha256()
    for parameter,_ in student.frozen:
        result.update(str(tuple(parameter.shape)).encode());result.update(str(parameter.dtype).encode())
        result.update(parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return result.hexdigest()


def wrap(student):
    from torch.nn.parallel import DistributedDataParallel
    before=frozen_hash(student)
    model=DistributedDataParallel(student,device_ids=[0],broadcast_buffers=False,find_unused_parameters=False)
    after=frozen_hash(student)
    assert before==after, 'DDP broadcast changed frozen parameter values'
    # Broadcast can increment version counters even when values are identical.
    # Update the ledger only after comparing every frozen parameter byte.
    changed=sum(p._version!=v for p,v in student.frozen)
    assert all(p.grad is None and not p.requires_grad for p,_ in student.frozen)
    student.frozen=[(p,p._version) for p,_ in student.frozen]
    student.verify_frozen()
    return model,dict(frozen_parameters_sha256_before=before,frozen_parameters_sha256_after=after,
        version_counters_incremented_by_initial_broadcast=changed,all_frozen_values_byte_exact=True)
