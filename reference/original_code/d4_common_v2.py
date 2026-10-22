"""Fourth-batch private outputs and frozen earlier-batch read dependencies."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
B1=ROOT.parent/'1016-rl_representation_analysis'
B2=ROOT.parent/'1018-representation_repair_batch2'
B3=ROOT.parent/'1021-representation_repair_batch3'
sys.dont_write_bytecode=True
for path in [B1/'code/upstream',B1/'deps',B1/'code',B2/'code',B3/'code',ROOT/'code']:
    if str(path) not in sys.path:sys.path.insert(0,str(path))
from r3_common import read,retry,sha,digest,seed,stamp,compatibility


def atomic(path,value):
    path=Path(path).resolve();assert path.is_relative_to(ROOT)
    data=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    def write():
        path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
        tmp.write_text(data);tmp.replace(path)
    retry(write)


def setup(name,threads=1):
    base=ROOT/'runtime'/name;base.mkdir(parents=True,exist_ok=True)
    for key in ['HF_HOME','HF_MODULES_CACHE','TRANSFORMERS_CACHE','TORCH_HOME','XDG_CACHE_HOME','MPLCONFIGDIR',
                'NUMBA_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR','TMPDIR']:
        path=base/key.lower();path.mkdir(exist_ok=True);os.environ[key]=str(path)
    # Python tensor IPC appends Unix socket names; keep this task-private path short.
    short=ROOT/'tmp';short.mkdir(exist_ok=True)
    assert len((str(short)+'/pymp-00000000/listener-00000000').encode())<108
    os.environ['TMPDIR']=str(short)
    import tempfile
    tempfile.tempdir=str(short)
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]=str(threads)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        NUPLAN_MAPS_ROOT='/mnt/data/navsim/maps',NUPLAN_MAP_VERSION='nuplan-maps-v1.0')


def save_arrays(path,values):
    import io
    import zipfile
    import numpy as np
    path=Path(path).resolve();assert path.is_relative_to(ROOT) and not path.exists()
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for name,value in values.items():
            stream=io.BytesIO();np.lib.format.write_array(stream,np.asarray(value),allow_pickle=False)
            archive.writestr(name+'.npy',stream.getvalue())
    tmp.replace(path);return sha(path)
