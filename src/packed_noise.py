"""Optional exact original-device RNG tensors; the unmodified sampler stays inside."""
from pathlib import Path
import os
def install(student,cohort='navtrain'):
    if os.environ.get('RECOGDRIVE_NATIVE_RNG')=='1':return False
    import numpy as np
    from portable_data import local_rows
    from noise_replay import replay_bank
    root=Path(os.environ['RECOGDRIVE_DATA']);path=root/'noise'/(cohort+'_noise.npy')
    if not path.exists():return False
    data=np.load(path,mmap_mode='r',allow_pickle=False);rows=local_rows(cohort);positions={r['token']:i for i,r in enumerate(rows)}
    assert len(rows)==data.shape[0] and data.shape[2:]==(6,1,8,3)
    original=student.sample
    def sample(features,hidden,interface,token,noise_index):
        assert noise_index<data.shape[1]
        with replay_bank(data[positions[token],noise_index]):return original(features,hidden,interface,token,noise_index)
    student.sample=sample;return True
