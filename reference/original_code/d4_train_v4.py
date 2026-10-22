"""Node-local DDP distillation with matched scene/noise exposure and resumption."""
from d4_common_v2 import *
import argparse
import contextlib
import datetime
import math
import traceback


def checkpoint(student,optimizer,config,config_sha,run,epoch,next_micro,step,rank,world_size,best,exposure,pools):
    import torch
    import torch.distributed as dist
    states=[None]*world_size
    dist.all_gather_object(states,dict(rank=rank,cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(0),exposure=exposure,pools=dict(pools)))
    ref=None
    if rank==0:
        path=run/'checkpoints'/('step%06d_epoch%02d_micro%06d.pt'%(step,epoch,next_micro));path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
            torch.save(dict(status='DISTILLATION_CHECKPOINT',configuration=config,configuration_sha256=config_sha,
                trainable={n:p.detach().cpu() for n,p in student.named_parameters() if p.requires_grad},optimizer=optimizer.state_dict(),
                epoch=epoch,next_micro=next_micro,global_step=step,random_states=states,best_development=best,
                original_checkpoint_sha256=student.identity['checkpoint_sha256'],frozen_parameters_unchanged=True),temporary)
            temporary.replace(path)
        ref=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),global_step=step,epoch=epoch,next_micro=next_micro)
        atomic(run/'LATEST_CHECKPOINT.json',dict(time=stamp(),**ref))
    box=[ref];dist.broadcast_object_list(box,src=0);return box[0]


def main(a):
    setup('train/'+a.run+'/rank%02d'%a.rank);compatibility()
    import numpy as np
    import torch
    import torch.distributed as dist
    from d4_prefetch_v1 import PrefetchLoader
    from d4_student_v1 import Student,losses
    from d4_distributed_v1 import wrap
    from d4_dataset_v2 import TeacherDataset,initialize_worker
    from d4_development_eval_v3 import evaluate
    config_path=Path(a.config);config=read(config_path);config_sha=sha(config_path)
    assert config['run_id']==a.run and config['world_size']==a.world_size
    assert (ROOT/'provenance/DDP_SMOKE_ALL_NODES_ACCEPTANCE_V2.json').exists()
    assert (ROOT/'runs/full_context_replay_v1/AGENT_ACCEPTANCE.json').exists()
    assert config['GT_used'] is False
    assert torch.cuda.device_count()==1 and torch.cuda.mem_get_info()[0]>12*(1<<30)
    dist.init_process_group('nccl',rank=a.rank,world_size=a.world_size,timeout=datetime.timedelta(minutes=30))
    student=Student(config);model,broadcast=wrap(student)
    run=ROOT/'runs'/a.run;rank_folder=run/'ranks'/('rank%02d'%a.rank)
    atomic(rank_folder/'PREFLIGHT.json',dict(time=stamp(),rank=a.rank,world_size=a.world_size,device=torch.cuda.get_device_name(0),
        free_gpu_bytes=torch.cuda.mem_get_info()[0],trainable_manifest=student.module_manifest,broadcast=broadcast,configuration_sha256=config_sha))
    optimizer=torch.optim.AdamW(student.optimizer_groups(),weight_decay=config['weight_decay'],betas=tuple(config['adam_betas']),eps=config['adam_epsilon'])
    train_path=Path(config['train_indices']);assert sha(train_path)==config['train_indices_sha256'];train_indices=read(train_path)
    manifest=Path(config['scene_manifest']);assert sha(manifest)==config['scene_manifest_sha256']
    scenes=read(manifest);assert all(scenes[i]['subset']=='exploration' and scenes[i]['pool'] in ['I','P'] for i in train_indices)
    valid_path=Path(config['development_indices']);assert sha(valid_path)==config['development_indices_sha256'];validation=read(valid_path)
    assert all(scenes[i]['subset']=='validation' for i in validation)
    batch=config['global_batch_size'];assert batch%a.world_size==0;accumulation=batch//a.world_size
    padded=math.ceil(len(train_indices)/batch)*batch;steps_per_epoch=padded//batch;total_steps=steps_per_epoch*config['epochs']
    step=0;start_epoch=0;next_micro=0;best=None;exposure=0;all_pools={};began=time.monotonic()
    if a.resume:
        state=torch.load(a.resume,map_location='cpu',weights_only=False);assert state['configuration_sha256']==config_sha
        with torch.no_grad():
            for name,p in student.named_parameters():
                if p.requires_grad:p.copy_(state['trainable'][name])
        optimizer.load_state_dict(state['optimizer']);step=state['global_step'];start_epoch=state['epoch'];next_micro=state['next_micro'];best=state['best_development']
        rng=state['random_states'][a.rank];torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state(rng['cuda'],device=0)
        exposure=rng['exposure'];all_pools=rng['pools']
        if (run/'BEST_DEVELOPMENT.json').exists():
            saved_best=read(run/'BEST_DEVELOPMENT.json');assert saved_best['configuration_sha256']==config_sha
            if saved_best['global_step']<=step:best={k:v for k,v in saved_best.items() if k not in ['status','time','configuration_sha256']}
        student.verify_frozen()
    log=rank_folder/'OPTIMIZATION.jsonl';loss_sum=torch.zeros(3,device='cuda');logged=0;window=time.monotonic()
    for epoch in range(start_epoch,config['epochs']):
        order=np.asarray(train_indices,dtype=np.int64);rng=np.random.default_rng(config['data_seed']+epoch);rng.shuffle(order)
        order=np.concatenate([order,order[:padded-len(order)]]) if padded>len(order) else order
        local=order[a.rank::a.world_size].tolist();offset=next_micro if epoch==start_epoch else 0;next_micro=0
        dataset=TeacherDataset(local[offset:],str(manifest),config['scene_manifest_sha256'],noise_epoch=epoch)
        generator=torch.Generator().manual_seed(config['loader_seed']+1000*epoch+a.rank)
        kwargs=dict(batch_size=None,num_workers=config['loader_workers'],pin_memory=True,worker_init_fn=initialize_worker,generator=generator)
        if config['loader_workers']:kwargs.update(multiprocessing_context='spawn',prefetch_factor=config['prefetch_factor'],timeout=120)
        loader=PrefetchLoader(dataset,workers=config['loader_workers'],prefetch_factor=config['prefetch_factor'],timeout=120);optimizer.zero_grad(set_to_none=True);model.train()
        for position,data in enumerate(loader,start=offset):
            last=(position+1)%accumulation==0
            context=contextlib.nullcontext() if last else model.no_sync()
            with context:
                out=model(data['features'],data['token'],data['noise_index'],trajectory=config['group']!='H')
                assert torch.equal(out['input_ids'][0],data['target']['input_ids'].cuda(non_blocking=True))
                assert torch.equal(out['attention_mask'][0],data['target']['attention_mask'].cuda(non_blocking=True))
                lh,lt=losses(out,data['target'],data['noise_index'])
                objective=config['lambda_H']*lh
                if lt is not None:objective=objective+config['lambda_T']*lt
                (objective/accumulation).backward()
            loss_sum+=torch.stack([objective.detach(),lh.detach(),lh.detach().new_zeros(()) if lt is None else lt.detach()]);logged+=1
            exposure+=1;all_pools[data['pool']]=all_pools.get(data['pool'],0)+1
            if last:
                gradient_norm=torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad],config['gradient_clip_norm'],error_if_nonfinite=True)
                scheduled=min(1.,(step+1)/config['warmup_steps']) if step<config['warmup_steps'] else config['minimum_lr_ratio']+(1-config['minimum_lr_ratio'])*.5*(1+math.cos(math.pi*(step-config['warmup_steps'])/max(1,total_steps-config['warmup_steps'])))
                for group,initial in zip(optimizer.param_groups,[config['lora_learning_rate'],config['projection_learning_rate']]):group['lr']=initial*scheduled
                optimizer.step();optimizer.zero_grad(set_to_none=True);step+=1;student.verify_frozen()
                if step==1 or step%config['log_every_steps']==0:
                    mean=(loss_sum/logged).cpu().tolist();assert np.isfinite(mean).all();seconds=time.monotonic()-window
                    record=dict(time=stamp(),rank=a.rank,epoch=epoch,global_step=step,loss=mean[0],H=mean[1],T=mean[2],gradient_norm=float(gradient_norm),
                        local_exposures=exposure,pools=all_pools,local_scenes_per_second=logged/max(seconds,1e-9),learning_rates=[g['lr'] for g in optimizer.param_groups],
                        elapsed_seconds=time.monotonic()-began,actual_noise_rule=config['noise_rule'])
                    log.parent.mkdir(parents=True,exist_ok=True)
                    with log.open('a') as stream:stream.write(json.dumps(record,separators=(',',':'))+'\n')
                    atomic(rank_folder/'LIVE.json',dict(status='TRAINING',**record));loss_sum.zero_();logged=0;window=time.monotonic()
                if step%config['checkpoint_every_steps']==0:
                    checkpoint(student,optimizer,config,config_sha,run,epoch,position+1,step,a.rank,a.world_size,best,exposure,all_pools)
                if step==config['early_evaluation_step']:
                    early_ref=checkpoint(student,optimizer,config,config_sha,run,epoch,position+1,step,a.rank,a.world_size,best,exposure,all_pools)
                    early=evaluate(student,validation[:min(64,len(validation))],a.rank,a.world_size,str((run/'early_development').relative_to(ROOT)),loader_workers=config['loader_workers'])
                    if a.rank==0:atomic(run/'EARLY_PHASE_READY.json',dict(status='EARLY_METRICS_REQUIRE_AGENT_REVIEW',time=stamp(),checkpoint=early_ref,development=early,
                        training_continues=True,not_eligible_for_final_checkpoint_selection=True))
                    model.train()
            del out,lh,lt,objective
        assert (len(local)-offset)%accumulation==0
        reference=checkpoint(student,optimizer,config,config_sha,run,epoch+1,0,step,a.rank,a.world_size,best,exposure,all_pools)
        development=evaluate(student,validation,a.rank,a.world_size,str((run/'development'/('epoch%02d'%(epoch+1))).relative_to(ROOT)),loader_workers=config['loader_workers'])
        candidate=dict(checkpoint=reference,PDMS=development['PDMS'],gain=development['gain'],epoch=epoch+1,global_step=step,
            development_summary=str((run/'development'/('epoch%02d'%(epoch+1))/'SUMMARY.json').relative_to(ROOT)))
        if best is None or candidate['PDMS']>best['PDMS'] or (candidate['PDMS']==best['PDMS'] and step<best['global_step']):best=candidate
        if a.rank==0:atomic(run/'BEST_DEVELOPMENT.json',dict(status='SELECTED_ON_DEVELOPMENT_ONLY',time=stamp(),configuration_sha256=config_sha,**best))
        atomic(rank_folder/'LIVE.json',dict(status='EPOCH_COMPLETE',time=stamp(),epoch=epoch+1,global_step=step,best=best,local_exposures=exposure))
        if a.stop_after_epoch and epoch+1==a.stop_after_epoch:
            assert config['engineering_only'] and epoch+1<config['epochs']
            atomic(rank_folder/'ENGINEERING_BOUNDARY_PAUSE.json',dict(status='ENGINEERING_CHECKPOINT_BOUNDARY_PAUSE',time=stamp(),checkpoint=reference,configuration_sha256=config_sha))
            dist.barrier()
            if a.rank==0:atomic(run/'ENGINEERING_BOUNDARY_PAUSE.json',dict(status='ENGINEERING_PAUSED_FOR_REAL_RESUME_TEST',time=stamp(),checkpoint=reference,world_size=a.world_size))
            dist.destroy_process_group();return
    assert step==total_steps
    atomic(rank_folder/'TRAINING_COMPLETE.json',dict(status='TRAINING_COMPLETE_REQUIRES_FULL_EVALUATION',time=stamp(),rank=a.rank,group=config['group'],
        configuration_sha256=config_sha,optimizer_steps=step,local_exposures=exposure,pools=all_pools,best=best,final_matched_exposure_checkpoint=reference,elapsed_seconds=time.monotonic()-began,
        frozen_original_parameters_unchanged=True,GT_used=False))
    dist.barrier()
    if a.rank==0:atomic(run/'TRAINING_COMPLETE.json',dict(status='ALL_RANKS_TRAINING_COMPLETE_REQUIRES_FULL_EVALUATION',time=stamp(),best=best,final_matched_exposure_checkpoint=reference,world_size=a.world_size,configuration_sha256=config_sha))
    dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--run',required=True);p.add_argument('--rank',type=int,required=True)
    p.add_argument('--world-size',type=int,required=True);p.add_argument('--resume');p.add_argument('--stop-after-epoch',type=int,default=0);a=p.parse_args()
    try:main(a)
    except BaseException as error:
        atomic(ROOT/'runs'/a.run/'failures'/('rank%02d.json'%a.rank),dict(status='FAILED',time=stamp(),rank=a.rank,error=repr(error),traceback=traceback.format_exc()));raise
