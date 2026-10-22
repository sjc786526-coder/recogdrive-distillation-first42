"""Matched-noise, no-grad development evaluation; never trains on validation."""
from d4_common_v2 import *
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


def initialize_score():setup('development_scores/'+str(os.getpid()),threads=1);compatibility()


def evaluate(student,indices,rank,world_size,output_stage,loader_workers=2):
    import numpy as np
    import torch
    import torch.distributed as dist
    from d4_prefetch_v1 import PrefetchLoader
    from d4_dataset_v2 import TeacherDataset,initialize_worker
    from repair_score import score_group
    from r3_incremental_audit_v1 import FIELDS
    manifest=ROOT/'manifests/R2_TRAINING_SCENES_V2.json';local=indices[rank::world_size]
    dataset=TeacherDataset(local,str(manifest),sha(manifest));rows=dataset.scenes
    assert all(rows[i]['subset']=='validation' for i in indices)
    kwargs=dict(batch_size=None,num_workers=loader_workers,pin_memory=True,worker_init_fn=initialize_worker)
    if loader_workers:kwargs.update(multiprocessing_context='spawn',prefetch_factor=2,timeout=120)
    loader=PrefetchLoader(dataset,workers=loader_workers,prefetch_factor=2,timeout=120);summ=read(B3/'analysis/first42_navtrain_v1/SUMMARY.json')
    base=np.load(B3/summ['artifacts']['scores.npy']['path'],mmap_mode='r');pending=[];records=[]
    student.eval()
    def finish(item):
        future,data,trajectories,mse,relative=item;scored=future.result(timeout=3600)
        lookup={s['seed_index']:s for s in scored};assert set(lookup)==set(range(5))
        q=np.asarray([[lookup[j]['score'][f] for f in FIELDS] for j in range(5)],dtype=np.float64)
        assert np.isfinite(q).all() and (q>=0).all() and (q<=1+1e-8).all()
        index=data['global_index'];row=rows[index]
        records.append(dict(global_index=index,token=row['token'],session=row['session'],pool=row['pool'],quality=q.tolist(),
            Base_quality=np.asarray(base[index,0]).tolist(),trajectories=trajectories.tolist(),interface_MSE=mse,relative_interface_error=relative))
    with ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context('spawn'),initializer=initialize_score) as pool:
        with torch.no_grad():
            for data in loader:
                hidden={}
                handle=student.agent.action_head.feature_encoder.register_forward_pre_hook(lambda module,args:hidden.update(value=args[0]))
                try:out=student(data['features'],data['token'],0,trajectory=False)
                finally:handle.remove()
                assert np.array_equal(out['input_ids'][0].cpu().numpy(),data['target']['input_ids'].numpy())
                assert np.array_equal(out['attention_mask'][0].cpu().numpy(),data['target']['attention_mask'].numpy())
                h=out['interface'];target=data['target']['teacher_interface'].cuda(non_blocking=True)
                squared=(h[0]-target).square();mse=float(squared.mean());den=data['teacher_residual_squared_sum']
                relative=float((squared.sum()/den).sqrt()) if den>0 else None
                predictions=[student.sample(data['features'],hidden['value'],h,data['token'],j)[0] for j in range(5)]
                trajectories=torch.stack(predictions).float().cpu().numpy()
                row=rows[data['global_index']];metric_sha=row.get('metric_sha256') or sha(row['metric_cache'])
                future=pool.submit(score_group,dict(metric_cache=row['metric_cache'],metric_sha256=metric_sha,scoring=student.scientific['scoring'],
                    records=[dict(arm='student',seed_index=j,seed=seed(row['token'],j),trajectory=trajectories[j].tolist()) for j in range(5)]))
                pending.append((future,data,trajectories,mse,relative))
                if len(pending)>=2:finish(pending.pop(0))
            for item in pending:finish(item)
    folder=ROOT/output_stage;path=folder/('rank%02d.json'%rank)
    proof=dict(status='PASS_DEVELOPMENT_NATIVE_INFERENCE_AND_FRESH_SCORE',time=stamp(),rank=rank,cases=records,indices=local,GT_used=False)
    atomic(path,proof);dist.barrier()
    summary=None
    if rank==0:
        combined=[]
        for r in range(world_size):
            part=read(folder/('rank%02d.json'%r));assert part['status']==proof['status'];combined.extend(part['cases'])
        assert sorted(v['global_index'] for v in combined)==sorted(indices)
        q=np.asarray([v['quality'] for v in combined]);b=np.asarray([v['Base_quality'] for v in combined]);delta=q[:,:,0]-b[:,:,0]
        safety={}
        for name in ['no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound']:
            col=FIELDS.index(name);safety[name]=dict(new=int(((q[:,:,col]<1)&(b[:,:,col]>=1)).sum()),removed=int(((q[:,:,col]>=1)&(b[:,:,col]<1)).sum()))
        summary=dict(status='PASS_DEVELOPMENT_PANEL',time=stamp(),cases=len(combined),outputs=5*len(combined),noises=[0,1,2,3,4],
            PDMS=float(q[:,:,0].mean()*100),Base_PDMS=float(b[:,:,0].mean()*100),gain=float(delta.mean()*100),
            mean_interface_MSE=float(np.mean([v['interface_MSE'] for v in combined])),safety=safety,
            source_rank_files=[dict(path=str((folder/('rank%02d.json'%r)).relative_to(ROOT)),sha256=sha(folder/('rank%02d.json'%r))) for r in range(world_size)],
            checkpoint_selection_metric='PDMS only; no extra safety eligibility',GT_used=False,student_inference_teacher_or_PDMS_input=False)
        atomic(folder/'SUMMARY.json',summary)
    box=[summary];dist.broadcast_object_list(box,src=0);student.verify_frozen();return box[0]
