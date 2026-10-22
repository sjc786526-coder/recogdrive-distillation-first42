"""No-GT raw observations and immutable same-execution full-context targets."""
from d4_common_v2 import *
import io


class TeacherDataset:
    def __init__(self,indices,scene_manifest,scene_manifest_sha256,noise_epoch=0):
        assert sha(scene_manifest)==scene_manifest_sha256
        self.scenes=read(scene_manifest);self.indices=list(indices);self.noise_epoch=noise_epoch

    def __len__(self):return len(self.indices)

    def __getitem__(self,position):
        import numpy as np
        import torch
        from vlm_core import raw_features
        index=self.indices[position];row=self.scenes[index]
        record=read(ROOT/'runs/full_context_replay_v1/cases'/row['token']/'RESULT.json')
        assert record['status']=='PASS_EXACT_R2_FULL_CONTEXT_REPLAY' and record['global_index']==index
        data=retry(lambda:(ROOT/record['arrays']['path']).read_bytes())
        assert hashlib.sha256(data).hexdigest()==record['arrays']['sha256']
        with np.load(io.BytesIO(data),allow_pickle=False) as z:target={k:torch.from_numpy(np.array(z[k])) for k in z.files}
        return dict(global_index=index,token=row['token'],subset=row['subset'],pool=row['pool'],
            noise_index=(self.noise_epoch+seed(row['token'],0)%5)%5,features=raw_features(row),target=target,
            teacher_residual_squared_sum=record['teacher_residual_squared_sum'],Base_interface_squared_sum=record['Base_interface_squared_sum'])


def initialize_worker(worker):
    import torch
    setup('data_workers/'+str(os.getpid()),threads=1);compatibility();torch.set_num_threads(1)
