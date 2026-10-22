"""Replay frozen per-scene first42 R2 identities; no teacher selection or GT."""
from pathlib import Path
import argparse,hashlib,json,os,sys,uuid,shutil
REPO=Path(__file__).resolve().parent;sys.path.insert(0,str(REPO/'src'))
from portable_setup import configure

def initialize(data,out):
    data=Path(data).resolve();out=Path(out).resolve();assert data!=out,'Use a new output dataset; keep reference data intact'
    for cohort in ['navtrain','navtest']:
        for name in ['manifests/'+cohort+'_scenes.json','noise/'+cohort+'_noise.npy']:
            source=data/name;dest=out/name
            if not source.exists():continue
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert dest.stat().st_size==source.stat().st_size;continue
            temp=dest.with_name(dest.name+'.'+uuid.uuid4().hex+'.tmp');shutil.copyfile(source,temp);temp.replace(dest)
    source=data/'noise/PROTOCOL.json';dest=out/'noise/PROTOCOL.json'
    if source.exists() and not dest.exists():
        dest.parent.mkdir(parents=True,exist_ok=True);temp=dest.with_name(dest.name+'.'+uuid.uuid4().hex+'.tmp');shutil.copyfile(source,temp);temp.replace(dest)

def save_npz(path,values):
    import io,numpy as np,zipfile
    path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name,value in values.items():
            stream=io.BytesIO();np.lib.format.write_array(stream,np.asarray(value),allow_pickle=False);z.writestr(name+'.npy',stream.getvalue())
    temp.replace(path)

def main(a):
    configure(a.paths,a.data,a.out)
    from d4_common import ROOT,read,sha,atomic,stamp,setup,compatibility,digest
    setup('teacher/rank%04d'%a.rank);compatibility()
    import numpy as np,torch,time
    from portable_data import local_rows
    from vlm_core import raw_features
    from vlm_intervene_v4 import Engine
    from teacher_runtime import FitBank,observation_gate,prediction
    from noise_replay import replay_bank
    from repair_score import score_group
    from portable_eval import initialize_score,FIELDS
    from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
    import multiprocessing as mp
    assert a.world_size>0 and 0<=a.rank<a.world_size
    data=Path(a.data).resolve();out=Path(a.out).resolve();initialize(data,out)
    import fcntl
    locks=ROOT/'locks';locks.mkdir(exist_ok=True);rank_lock=(locks/('rank%04d.lock'%a.rank)).open('a')
    fcntl.flock(rank_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    spec=read(REPO/'configs/teacher_replay_v1.json');rows=local_rows('navtrain');assert len(rows)==103288
    entries={e['id']:dict(e,root=str(data/'teacher_replay_v1') if e['arm'] else None) for e in spec['entries']};assert len(entries)==42
    for asset in spec['assets']:
        path=data/'teacher_replay_v1'/asset['file'];assert path.stat().st_size==asset['bytes'] and sha(path)==asset['sha256'],path
    for name,hashed in spec['scientific_code_sha256'].items():assert sha(REPO/name)==hashed,name
    choices=np.load(data/'reference/navtrain/teacher_R2_candidate_index.npy',allow_pickle=False);assert choices.shape==(len(rows),)
    assert all(row['teacher_candidate']==spec['entries'][int(choices[i])]['id'] for i,row in enumerate(rows))
    noise_path=data/'noise/navtrain_noise.npy';assert noise_path.is_file(),'Download the original reference noise bank; do not silently switch RNG backend'
    noise=np.load(noise_path,mmap_mode='r',allow_pickle=False);assert noise.shape==(103288,5,6,1,8,3) and noise.dtype==np.float32
    protocol=read(data/'noise/PROTOCOL.json')
    assert sha(noise_path)==spec['original_noise']['sha256'],'Original reference noise bank changed'
    identity=dict(fixed_catalogue_sha256=sha(REPO/'configs/teacher_replay_v1.json'),original_PACK_sha256=spec['source_PACK_sha256'],scene_manifest_sha256=sha(data/'manifests/navtrain_scenes.json'),
        choice_index_sha256=sha(data/'reference/navtrain/teacher_R2_candidate_index.npy'),noise_sha256=sha(noise_path),noise_protocol_sha256=sha(data/'noise/PROTOCOL.json'),
        checkpoint_sha256=spec['checkpoint_sha256'],code_sha256={name:sha(REPO/name) for name in ['replay_teacher.py','src/teacher_runtime.py','native/vlm_intervene_v4.py','native/vlm_semantic.py','native/vlm_core.py']},
        teacher='fixed first42 R2 branch identity replayed on this platform',noises=list(range(5)),selection_not_repeated=True,GT_used=False)
    identity_sha=digest(identity);marker=ROOT/'REPLAY_IDENTITY.json'
    if marker.exists():assert read(marker)==identity,'Use a new output directory after any protocol change'
    else:atomic(marker,identity)
    if a.golden_only:indices=[v['global_index'] for v in spec['goldens']]
    elif a.indices:indices=read(a.indices)
    else:indices=list(range(len(rows)))
    if a.limit:indices=indices[:a.limit]
    assert len(set(indices))==len(indices) and set(indices)<=set(range(len(rows)))
    indices=indices[a.rank::a.world_size]
    rank_protocol=dict(indices=indices,world_size=a.world_size,rank=a.rank,identity_sha256=identity_sha,strict_original=a.strict_original)
    path=ROOT/('RANK_PROTOCOL_%04d.json'%a.rank)
    if path.exists():assert read(path)==rank_protocol,'Rank assignment changed; use a new output directory'
    else:atomic(path,rank_protocol)
    if a.check_only:print(json.dumps(dict(status='PASS_REPLAY_DEPENDENCY_CLOSURE',entries=42,assets=len(spec['assets']),assigned_scenes=len(indices))),flush=True);return
    sys.path.insert(0,str(REPO))
    from check_assets import check_models
    check_models(read(a.paths))
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    atomic(ROOT/('PREFLIGHT_%04d.json'%a.rank),dict(time=stamp(),device=torch.cuda.get_device_name(0),GPU_free_total=list(torch.cuda.mem_get_info()),torch=torch.__version__,rank=a.rank,world_size=a.world_size,A800_not_prevalidated=True))
    engine=Engine();assert engine.planner_runtime['checkpoint_sha256']==spec['checkpoint_sha256'];fits=FitBank()
    old_t=np.load(data/'reference/navtrain/teacher_R2_trajectories.npy',mmap_mode='r',allow_pickle=False);old_q=np.load(data/'reference/navtrain/teacher_R2_quality.npy',mmap_mode='r',allow_pickle=False)
    old_b=np.load(data/'reference/navtrain/Base_quality.npy',mmap_mode='r',allow_pickle=False)
    goldens={g['global_index']:g for g in spec.get('goldens',[])};done=[];pending=[];began=time.monotonic()
    def finish(item):
        future,index,arrays,metadata=item;row=rows[index];scored=future.result(timeout=3600);lookup={(r['arm'],r['seed_index']):r for r in scored}
        tq=np.asarray([[lookup['teacher',j]['score'][f] for f in FIELDS] for j in range(5)],dtype=np.float64)
        bq=np.asarray([[lookup['Base',j]['score'][f] for f in FIELDS] for j in range(5)],dtype=np.float64)
        arrays.update(teacher_quality=tq,Base_quality=bq)
        diagnostics=dict(teacher_trajectory_exact=bool(np.array_equal(arrays['teacher_trajectories'],old_t[index])),teacher_quality_exact=bool(np.array_equal(tq,old_q[index])),
            Base_quality_exact=bool(np.array_equal(bq,old_b[index])),trajectory_max_abs_difference=float(np.max(np.abs(arrays['teacher_trajectories']-old_t[index]))),
            teacher_PDMS_difference=float((tq[:,0]-old_q[index,:,0]).mean()*100),Base_PDMS_difference=float((bq[:,0]-old_b[index,:,0]).mean()*100))
        if index in goldens:
            golden=data/'teacher_replay_v1'/goldens[index]['file'];assert sha(golden)==goldens[index]['sha256']
            with np.load(golden,allow_pickle=False) as z:
                for name in ['input_ids','attention_mask','position_ids','valid_token_positions']:assert np.array_equal(arrays[name],z[name]),('Input preprocessing changed',index,name)
                h=arrays['teacher_interface'];g=z['teacher_interface'];diagnostics.update(interface_exact=bool(np.array_equal(h,g)),interface_max_abs_difference=float(np.max(np.abs(h-g))),interface_relative_L2=float(np.linalg.norm(h.astype(np.float64)-g)/max(np.linalg.norm(g.astype(np.float64)),1e-12)))
        if a.strict_original:assert diagnostics['teacher_trajectory_exact'] and diagnostics['teacher_quality_exact'] and diagnostics['Base_quality_exact'] and diagnostics.get('interface_exact',True),(index,diagnostics)
        folder=ROOT/'teacher_cases'/row['token'];target=folder/'full_context.npz'
        # A crash may leave an uncommitted array. Validate before replacing only
        # this run's orphan; never overwrite a completed case or source asset.
        if target.exists():
            with np.load(target,allow_pickle=False) as z:assert set(z.files)==set(arrays) and all(np.array_equal(z[k],v) for k,v in arrays.items())
        else:save_npz(target,arrays)
        record=dict(status='COMPLETE_FIXED_R2_PLATFORM_REPLAY',time=stamp(),global_index=index,token=row['token'],subset=row['subset'],pool=row['pool'],candidate=row['teacher_candidate'],
            identity_sha256=identity_sha,npz=str(target.relative_to(ROOT)),sha256=sha(target),bytes=target.stat().st_size,full_tokens=len(arrays['input_ids']),
            teacher_residual_squared_sum=metadata['teacher_residual_squared_sum'],Base_interface_squared_sum=metadata['Base_interface_squared_sum'],
            teacher_quality=tq.tolist(),Base_quality=bq.tolist(),teacher_trajectories=arrays['teacher_trajectories'].tolist(),Base_trajectories=arrays['Base_trajectories'].tolist(),
            diagnostics_against_original_platform=diagnostics,same_execution_interface=True,teacher_fixed_not_reselected=True,GT_used=False)
        atomic(folder/'RESULT.json',record);done.append(index)
        atomic(ROOT/('LIVE_rank%04d.json'%a.rank),dict(status='REPLAYING',time=stamp(),complete=len(done),assigned=len(indices),last_global_index=index,elapsed_seconds=time.monotonic()-began))
    with ProcessPoolExecutor(max_workers=a.scoring_workers,mp_context=mp.get_context('spawn'),initializer=initialize_score) as scorers,ThreadPoolExecutor(max_workers=1) as reader:
        future=reader.submit(raw_features,rows[indices[0]]) if indices else None
        for position,index in enumerate(indices):
            features=future.result(timeout=180)
            if position+1<len(indices):future=reader.submit(raw_features,rows[indices[position+1]])
            path=ROOT/'teacher_cases'/rows[index]['token']/'RESULT.json'
            if path.exists():
                previous=read(path);assert previous['identity_sha256']==identity_sha and previous['global_index']==index and sha(ROOT/previous['npz'])==previous['sha256'];done.append(index);continue
            row=rows[index];entry=entries[row['teacher_candidate']]
            with torch.no_grad(),torch.autocast('cuda',enabled=False):
                hidden=engine.forward_native(features);base_h=engine.prefix['interface_post'];condition,post=hidden,None;actual=entry['arm']
                if actual and actual['gate']=='external_observation_qda_hard':
                    active,_=observation_gate(engine,entry['root'],actual);actual=dict(actual,gate='none') if active else None
                if actual:
                    changed,_=engine.edited(actual,fits.get(entry['root'],actual));condition,post=engine.suffix(actual['block'],changed)
                target_h=None;trajectories=[];base_trajectories=[]
                for j in range(5):
                    with replay_bank(noise[index,j]):trajectory,h=prediction(engine,features,condition,j,row['token'],post)
                    if target_h is None:target_h=h
                    else:assert torch.equal(h,target_h),'Upstream target unexpectedly depends on action noise'
                    trajectories.append(trajectory)
                    if actual:
                        with replay_bank(noise[index,j]):base_trajectory,_=prediction(engine,features,hidden,j,row['token'])
                    else:base_trajectory=trajectory
                    base_trajectories.append(base_trajectory)
                arrays=dict(teacher_interface=target_h[0].float().cpu().numpy(),teacher_trajectories=torch.stack(trajectories).float().cpu().numpy(),
                    Base_trajectories=torch.stack(base_trajectories).float().cpu().numpy(),input_ids=engine.inputs['input_ids'][0].cpu().numpy(),attention_mask=engine.inputs['attention_mask'][0].cpu().numpy(),position_ids=engine.inputs['position_ids'][0].cpu().numpy())
                arrays['valid_token_positions']=np.flatnonzero(arrays['attention_mask']);assert arrays['teacher_interface'].shape==(len(arrays['input_ids']),384)
                assert all(np.isfinite(v).all() for v in arrays.values())
                metadata=dict(teacher_residual_squared_sum=float((target_h-base_h).square().sum().cpu()),Base_interface_squared_sum=float(base_h.square().sum().cpu()))
            score_records=[dict(arm=name,seed_index=j,trajectory=arrays[key][j].tolist()) for name,key in [('teacher','teacher_trajectories'),('Base','Base_trajectories')] for j in range(5)]
            scoring=scorers.submit(score_group,dict(metric_cache=row['metric_cache'],metric_sha256=row['metric_sha256'],scoring=engine.cfg['scoring'],records=score_records))
            pending.append((scoring,index,arrays,metadata))
            if len(pending)>=2:finish(pending.pop(0))
        for item in pending:finish(item)
    assert sorted(done)==sorted(indices);engine.verify_engine()
    result=dict(status='COMPLETE_FIXED_R2_REPLAY_RANK',time=stamp(),rank=a.rank,world_size=a.world_size,cases=len(done),indices=indices,identity_sha256=identity_sha,elapsed_seconds=time.monotonic()-began)
    atomic(ROOT/('COMPLETE_rank%04d.json'%a.rank),result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paths',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1)
    p.add_argument('--indices');p.add_argument('--limit',type=int,default=0);p.add_argument('--golden-only',action='store_true');p.add_argument('--strict-original',action='store_true');p.add_argument('--check-only',action='store_true');p.add_argument('--scoring-workers',type=int,default=1)
    args=p.parse_args()
    try:main(args)
    except Exception as error:
        import traceback
        out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True);(out/('FAILED_rank%04d.json'%args.rank)).write_text(json.dumps(dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc()),indent=2)+'\n');raise
