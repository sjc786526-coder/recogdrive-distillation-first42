"""Read-only identity and completeness check before any new training."""
from pathlib import Path
import argparse,hashlib,json,sys
REPO=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def check_models(paths,fast=False):
    expected=json.loads((REPO/'native/configs/repair_final_baseline.json').read_text())
    assert Path(paths['checkpoint']).is_file() and Path(paths['vlm']).is_dir()
    if not fast:assert sha(paths['checkpoint'])==expected['checkpoint_sha256'],'RL Base SHA mismatch'
    checked=0
    for name,hashed in expected['vlm_sha256'].items():
        p=Path(paths['vlm'])/name
        if name in ['all_results.json','train_results.json','trainer_state.json'] and not p.exists():continue
        assert p.is_file(),p
        if not fast or p.stat().st_size<10000000:assert sha(p)==hashed,('VLM mismatch',name)
        checked+=1
    return checked
def main(a):
    paths=json.loads(Path(a.paths).read_text());data=Path(a.data) if a.data else None
    checked=check_models(paths,a.fast)
    if a.models_only:
        print(json.dumps(dict(status='PASS_MODEL_IDENTITY_CHECK',VLM_files_checked=checked,large_hashes_checked=not a.fast)),flush=True);return
    assert data is not None,'--data is required for teacher dataset checks'
    rows=json.loads((data/'manifests/navtrain_scenes.json').read_text());train=json.loads((REPO/'manifests/TRAIN_INDICES_V2.json').read_text());val=json.loads((REPO/'manifests/VALIDATION_INDICES_V1.json').read_text())
    assert len(rows)==103288 and len(train)==85339 and len(val)==17475 and not (set(train)&set(val))
    assert all(rows[i]['subset']=='exploration' and rows[i]['pool'] in ['I','P'] for i in train)
    assert all(rows[i]['subset']=='validation' for i in val)
    index=json.loads((data/'manifests/TEACHER_INDEX.json').read_text());assert len(index)==103288
    if (data/'REPLAY_IDENTITY.json').exists():
        ready=json.loads((data/'TEACHER_READY.json').read_text());assert ready['complete_population'] and ready['scenes']==103288
        assert sha(data/'manifests/TEACHER_INDEX.json')==ready['index_sha256']
        for row in index:
            p=(data/row['npz']).resolve();assert p.is_relative_to(data.resolve()) and p.is_file() and p.stat().st_size==row['bytes'],p
            if a.verify_shards:assert sha(p)==row['sha256'],p
    else:
        shards=json.loads((data/'manifests/TEACHER_SHARDS.json').read_text())['shards']
        for shard in shards:
            p=data/shard['file'];assert p.is_file() and p.stat().st_size==shard['bytes'],p
            if a.verify_shards:assert sha(p)==shard['sha256'],p
    for cohort in ['navtrain','navtest']:
        examples=json.loads((data/'manifests'/(cohort+'_scenes.json')).read_text())
        for r in examples[:2]:
            for key,rel in [('logs','raw_rel'),('sensor','sensor_rel'),('metric','metric_rel')]:assert (Path(paths[cohort+'_'+key])/r[rel]).exists(),(cohort,key,r['token'])
    print(json.dumps(dict(status='PASS_READ_ONLY_ASSET_CHECK',train_scenes=len(train),validation_scenes=len(val),teacher_scenes=len(index),VLM_files_checked=checked,
        all_teacher_shards_hashed=a.verify_shards,large_Base_and_VLM_hashes_checked=not a.fast)),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paths',required=True);p.add_argument('--data');p.add_argument('--models-only',action='store_true');p.add_argument('--fast',action='store_true');p.add_argument('--verify-shards',action='store_true');main(p.parse_args())
