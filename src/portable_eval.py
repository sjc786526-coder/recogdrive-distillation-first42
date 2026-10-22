"""Exact native Student trajectories, independent real PDMS, no teacher routing."""
from d4_common import *
from portable_data import TeacherDataset
from d4_prefetch_v1 import PrefetchLoader
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
FIELDS=['score','no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','ego_progress','comfort','driving_direction_compliance']
def initialize_score():setup('scoring/'+str(os.getpid()),threads=1);compatibility()
def evaluate(student,indices,rank,world_size,output_stage,loader_workers=2):
    import numpy as np,torch
    import torch.distributed as dist
    from repair_score import score_group
    data_root=Path(os.environ['RECOGDRIVE_DATA']);local=indices[rank::world_size];dataset=TeacherDataset(local);rows=dataset.scenes
    assert all(rows[i]['subset']=='validation' for i in indices)
    base=np.load(data_root/'reference/navtrain/Base_quality.npy',mmap_mode='r',allow_pickle=False)
    records=[];pending=[];student.eval()
    def finish(item):
        future,data,trajectories,mse,relative=item;values=future.result(timeout=3600);lookup={v['seed_index']:v for v in values}
        assert set(lookup)==set(range(5));q=np.asarray([[lookup[j]['score'][f] for f in FIELDS] for j in range(5)],dtype=np.float64)
        index=data['global_index'];row=rows[index]
        records.append(dict(global_index=index,token=row['token'],session=row['session'],pool=row['pool'],quality=q.tolist(),Base_quality=np.asarray(base[index]).tolist(),
            trajectories=trajectories.tolist(),interface_MSE=mse,relative_interface_error=relative))
    with ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context('spawn'),initializer=initialize_score) as pool:
        with torch.no_grad():
            for data in PrefetchLoader(dataset,workers=loader_workers,prefetch_factor=2,timeout=120):
                hidden={};handle=student.agent.action_head.feature_encoder.register_forward_pre_hook(lambda module,args:hidden.update(value=args[0]))
                try:out=student(data['features'],data['token'],0,trajectory=False)
                finally:handle.remove()
                assert torch.equal(out['input_ids'][0],data['target']['input_ids'].cuda()) and torch.equal(out['attention_mask'][0],data['target']['attention_mask'].cuda())
                h=out['interface'];squared=(h[0]-data['target']['teacher_interface'].cuda()).square();mse=float(squared.mean());den=data['teacher_residual_squared_sum']
                relative=float((squared.sum()/den).sqrt()) if den>0 else None
                predictions=torch.stack([student.sample(data['features'],hidden['value'],h,data['token'],j)[0] for j in range(5)]).float().cpu().numpy()
                row=rows[data['global_index']]
                future=pool.submit(score_group,dict(metric_cache=row['metric_cache'],metric_sha256=row['metric_sha256'],scoring=student.scientific['scoring'],
                    records=[dict(arm='student',seed_index=j,seed=seed(row['token'],j),trajectory=predictions[j].tolist()) for j in range(5)]))
                pending.append((future,data,predictions,mse,relative))
                if len(pending)>=2:finish(pending.pop(0))
            for item in pending:finish(item)
    folder=ROOT/output_stage;path=folder/('rank%02d.json'%rank)
    atomic(path,dict(status='PASS_PORTABLE_NATIVE_EVALUATION',time=stamp(),rank=rank,cases=records,indices=local,GT_used=False))
    dist.barrier();summary=None
    if rank==0:
        combined=[]
        for r in range(world_size):combined.extend(read(folder/('rank%02d.json'%r))['cases'])
        assert sorted(v['global_index'] for v in combined)==sorted(indices)
        q=np.asarray([v['quality'] for v in combined]);b=np.asarray([v['Base_quality'] for v in combined]);delta=q[:,:,0]-b[:,:,0]
        summary=dict(status='PASS_PORTABLE_FULLY_SCORED_VALIDATION',time=stamp(),cases=len(combined),noises=list(range(5)),PDMS=float(q[:,:,0].mean()*100),Base_PDMS=float(b[:,:,0].mean()*100),
            gain=float(delta.mean()*100),mean_interface_MSE=float(np.mean([v['interface_MSE'] for v in combined])),GT_used=False)
        atomic(folder/'SUMMARY.json',summary)
    box=[summary];dist.broadcast_object_list(box,src=0);student.verify_frozen();dataset.store.close();return box[0]
