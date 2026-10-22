"""Portable runtime paths only; scientific model and loss code stay verbatim."""
from pathlib import Path
import datetime as dt,errno,hashlib,json,os,sys,time,uuid
REPO=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('RECOGDRIVE_RUN_ROOT',REPO/'outputs')).resolve()
B1=REPO/'native';B2=REPO;B3=REPO
UPSTREAM=B1/'upstream'
COMPONENTS=['no_at_fault_collisions','drivable_area_compliance','ego_progress','time_to_collision_within_bound','comfort','driving_direction_compliance']
sys.dont_write_bytecode=True
for path in [B1,B1/'upstream',REPO/'src']:
    if str(path) not in sys.path:sys.path.insert(0,str(path))
def retry(fn):
    for n in range(7):
        try:return fn()
        except OSError as e:
            if e.errno not in {errno.ESTALE,errno.EIO,errno.ETIMEDOUT,errno.EAGAIN} or n==6:raise
            time.sleep(min(.2*2**n,4))
def read(p):
    p=Path(p);value=retry(lambda:json.loads(p.read_text()))
    if p.resolve()==(B1/'configs/repair_final_baseline.json').resolve():
        local=json.loads(Path(os.environ['RECOGDRIVE_PATHS']).read_text())
        value=dict(value,checkpoint=str(Path(local['checkpoint']).expanduser().resolve()),vlm=str(Path(local['vlm']).expanduser().resolve()))
    return value
def sha(p):
    def run():
        h=hashlib.sha256()
        with Path(p).open('rb') as f:
            for b in iter(lambda:f.read(8<<20),b''):h.update(b)
        return h.hexdigest()
    return retry(run)
def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def atomic(p,v):
    p=Path(p).resolve();assert p.is_relative_to(ROOT),p;p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def seed(token,j=0):return int(hashlib.sha256(f'{token}|20261008|{j}'.encode()).hexdigest()[:8],16)
def quality(value):return 'failure' if value<.7 else 'high_quality' if value>.9 else 'intermediate'
def reseed(value):
    import random,numpy as np,torch
    random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)
def setup(name,threads=1):
    base=ROOT/'runtime'/name;base.mkdir(parents=True,exist_ok=True)
    for k in ['HF_HOME','HF_MODULES_CACHE','TRANSFORMERS_CACHE','TORCH_HOME','XDG_CACHE_HOME','MPLCONFIGDIR','NUMBA_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR','TMPDIR']:
        p=base/k.lower();p.mkdir(exist_ok=True);os.environ[k]=str(p)
    for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]=str(threads)
    local=json.loads(Path(os.environ['RECOGDRIVE_PATHS']).read_text())
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',NUPLAN_MAPS_ROOT=local['maps'],NUPLAN_MAP_VERSION='nuplan-maps-v1.0')
def compatibility():
    import numpy as np
    if int(np.__version__.split('.')[0])<2:sys.modules.setdefault('numpy._core.numeric',np.core.numeric)
