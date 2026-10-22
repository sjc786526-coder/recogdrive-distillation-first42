"""CPU-only coverage audit and assembly of a locally replayed teacher dataset."""
from pathlib import Path
import argparse,hashlib,json,uuid,datetime
REPO=Path(__file__).resolve().parent
def read(path):return json.loads(Path(path).read_text())
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def main(a):
    import numpy as np
    root=Path(a.out).resolve();identity=read(root/'REPLAY_IDENTITY.json');hashed=digest(identity);rows=read(root/'manifests/navtrain_scenes.json');n=len(rows);assert n==103288
    indices=read(a.engineering_indices) if a.engineering_indices else list(range(n));assert len(indices)==len(set(indices)) and set(indices)<=set(range(n))
    if not a.engineering_indices:assert not (root/'ENGINEERING_ONLY.json').exists(),'Engineering outputs cannot become formal datasets'
    # Verify the entire requested population before publishing any ready marker.
    index=[];records=[];deltas=[];exact=0
    for i in indices:
        row=rows[i];r=read(root/'teacher_cases'/row['token']/'RESULT.json')
        assert r['status']=='COMPLETE_FIXED_R2_PLATFORM_REPLAY' and r['global_index']==i and r['token']==row['token'] and r['candidate']==row['teacher_candidate']
        assert r['identity_sha256']==hashed and r['same_execution_interface'] and r['teacher_fixed_not_reselected'] and r['GT_used'] is False
        p=(root/r['npz']).resolve();assert p.is_relative_to(root) and p.stat().st_size==r['bytes']
        if a.verify_files:assert sha(p)==r['sha256']
        index.append({k:r[k] for k in ['global_index','token','npz','sha256','bytes','candidate','full_tokens','subset','pool','teacher_residual_squared_sum','Base_interface_squared_sum']})
        records.append(r);d=r['diagnostics_against_original_platform'];deltas.append(d['teacher_PDMS_difference']);exact+=int(d['teacher_trajectory_exact'] and d['teacher_quality_exact'] and d.get('interface_exact',True))
    folder=root/'reference/navtrain';folder.mkdir(parents=True,exist_ok=True)
    arrays={
        'Base_quality':np.full((n,5,7),np.nan,dtype=np.float64),'teacher_R2_quality':np.full((n,5,7),np.nan,dtype=np.float64),
        'Base_trajectories':np.full((n,5,8,3),np.nan,dtype=np.float32),'teacher_R2_trajectories':np.full((n,5,8,3),np.nan,dtype=np.float32),
        'teacher_R2_candidate_index':np.full(n,-1,dtype=np.int64)}
    spec=read(REPO/'configs/teacher_replay_v1.json');by_id={e['id']:j for j,e in enumerate(spec['entries'])}
    for r in records:
        i=r['global_index']
        for name,key in [('Base_quality','Base_quality'),('teacher_R2_quality','teacher_quality'),('Base_trajectories','Base_trajectories'),('teacher_R2_trajectories','teacher_trajectories')]:arrays[name][i]=r[key]
        arrays['teacher_R2_candidate_index'][i]=by_id[r['candidate']]
    for name,value in arrays.items():
        path=folder/(name+'.npy')
        if path.exists():
            old=np.load(path,allow_pickle=False);assert np.array_equal(old,value,equal_nan=True),('Existing reference differs',path)
        else:
            tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
            with tmp.open('xb') as stream:np.save(stream,value,allow_pickle=False)
            tmp.replace(path)
    index_path=root/'manifests/TEACHER_INDEX.json'
    if index_path.exists():assert read(index_path)==index
    else:atomic(index_path,index)
    tq=np.asarray([r['teacher_quality'] for r in records]);bq=np.asarray([r['Base_quality'] for r in records])
    result=dict(status='PASS_ENGINEERING_REPLAY_DATASET_ONLY' if a.engineering_indices else 'PASS_COMPLETE_LOCAL_REPLAY_TEACHER_DATASET',time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scenes=len(index),expected_full_scenes=n,complete_population=not bool(a.engineering_indices),identity_sha256=hashed,index_sha256=sha(index_path),all_npz_rehashed=a.verify_files,
        teacher_PDMS=float(tq[:,:,0].mean()*100),Base_PDMS=float(bq[:,:,0].mean()*100),teacher_gain=float((tq[:,:,0]-bq[:,:,0]).mean()*100),
        original_platform_exact_cases=exact,mean_teacher_PDMS_difference_from_original=float(np.mean(deltas)),same_execution_targets=True,no_teacher_reselection=True)
    atomic(root/('ENGINEERING_ONLY.json' if a.engineering_indices else 'TEACHER_READY.json'),result)
    print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--engineering-indices');p.add_argument('--verify-files',action='store_true');main(p.parse_args())
