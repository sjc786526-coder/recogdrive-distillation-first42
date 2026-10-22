"""Read copied exact teacher NPZ payloads directly from indexed tar offsets."""
from d4_common import *
import io,threading
from functools import lru_cache
from collections import OrderedDict

@lru_cache(maxsize=8)
def local_rows(cohort):
    local=read(os.environ['RECOGDRIVE_PATHS']);data=Path(os.environ['RECOGDRIVE_DATA'])
    rows=read(data/'manifests'/(cohort+'_scenes.json'))
    for r in rows:
        r.update(raw_path=str(Path(local[cohort+'_logs'])/r['raw_rel']),sensor_root=str(Path(local[cohort+'_sensor'])/r['sensor_rel']),metric_cache=str(Path(local[cohort+'_metric'])/r['metric_rel']))
    return rows

class TeacherStore:
    def __init__(self,data,index=None):
        self.data=Path(data);values=read(index or os.environ.get('RECOGDRIVE_TEACHER_INDEX') or self.data/'manifests/TEACHER_INDEX.json')
        self.index={v['global_index']:v for v in values};assert len(self.index)==len(values);self.fds=OrderedDict();self.lock=threading.Lock()
    def get(self,index):
        import numpy as np
        r=self.index[index];path=(self.data/r.get('npz',r.get('shard',''))).resolve();assert path.is_relative_to(self.data.resolve())
        if 'npz' in r:data=path.read_bytes()
        else:
            with self.lock:
                if path not in self.fds:
                    self.fds[path]=os.open(path,os.O_RDONLY)
                    if len(self.fds)>8:os.close(self.fds.popitem(last=False)[1])
                self.fds.move_to_end(path);data=os.pread(self.fds[path],r['bytes'],r['offset'])
        assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256']
        with np.load(io.BytesIO(data),allow_pickle=False) as z:values={k:np.array(z[k]) for k in z.files}
        assert values['teacher_interface'].dtype==np.float32 and values['teacher_interface'].shape==(r['full_tokens'],384)
        assert values['teacher_trajectories'].shape==(5,8,3)
        return values,r
    def close(self):
        for fd in self.fds.values():os.close(fd)
        self.fds.clear()

class TeacherDataset:
    def __init__(self,indices,noise_epoch=0):
        self.indices=list(indices);self.noise_epoch=noise_epoch;self.scenes=local_rows('navtrain');self.store=TeacherStore(os.environ['RECOGDRIVE_DATA'])
    def __len__(self):return len(self.indices)
    def __getitem__(self,position):
        import torch
        from vlm_core import raw_features
        index=self.indices[position];row=self.scenes[index];values,record=self.store.get(index)
        assert row['token']==record['token']
        return dict(global_index=index,token=row['token'],subset=row['subset'],pool=row['pool'],noise_index=(self.noise_epoch+seed(row['token'],0)%5)%5,
            features=raw_features(row),target={k:torch.from_numpy(v) for k,v in values.items()},teacher_residual_squared_sum=record['teacher_residual_squared_sum'])
