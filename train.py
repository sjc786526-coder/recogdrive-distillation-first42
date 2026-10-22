"""Node-local DDP distillation with matched scene/noise exposure and resumption."""
from pathlib import Path
import sys,os,argparse
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from portable_setup import configure
_boot=argparse.ArgumentParser(add_help=False)
for _key in ['paths','data','out']:_boot.add_argument('--'+_key)
_args,_=_boot.parse_known_args()
if _args.paths and _args.data and _args.out:configure(_args.paths,_args.data,_args.out)

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
    from portable_data import TeacherDataset
    from portable_eval import evaluate
    config_path=REPO/'configs/original'/(a.group+'.json');config=read(config_path)
    config.update(run_id=Path(a.out).name,world_size=a.world_size,global_batch_size=a.global_batch,epochs=a.epochs,
        scene_manifest=str(Path(a.data)/'manifests/navtrain_scenes.json'),train_indices=str(REPO/'manifests/TRAIN_INDICES_V2.json'),
        development_indices=str(REPO/'manifests/DEVELOPMENT_512_V1.json'),source_original_configuration_sha256=sha(config_path))
    for field in ['scene_manifest','train_indices','development_indices']:config[field+'_sha256']=sha(config[field])
    config.pop('code_hashes',None)
    if a.smoke_indices:
        assert a.max_steps>0
        config['train_indices']=str(Path(a.smoke_indices).resolve());config['train_indices_sha256']=sha(config['train_indices']);config['engineering_only']=True
    if a.overrides:
        changes=read(a.overrides)
        allowed={'lambda_H','lambda_T','lora_learning_rate','projection_learning_rate','weight_decay','gradient_clip_norm','warmup_steps','minimum_lr_ratio'}
        assert set(changes)<=allowed,set(changes)-allowed;config.update(changes)
    bank=Path(a.data)/'noise/navtrain_noise.npy';protocol=Path(a.data)/'noise/PROTOCOL.json'
    config['explicit_original_device_noise']=bank.exists() and os.environ.get('RECOGDRIVE_NATIVE_RNG')!='1'
    config['noise_bank_protocol_sha256']=sha(protocol) if config['explicit_original_device_noise'] else None
    config['portable_code_hashes']={str(p.relative_to(REPO)):sha(p) for p in sorted((REPO/'src').glob('*.py'))}
    config_sha=digest(config);a.run=config['run_id']
    if a.rank==0:atomic(ROOT/'PORTABLE_CONFIGURATION.json',dict(config=config,sha256=config_sha))
    assert config['run_id']==a.run and config['world_size']==a.world_size
    assert config['GT_used'] is False
    assert torch.cuda.device_count()==1 and torch.cuda.mem_get_info()[0]>12*(1<<30)
    dist.init_process_group('nccl',rank=a.rank,world_size=a.world_size,timeout=datetime.timedelta(minutes=30))
    student=Student(config)
    from packed_noise import install
    explicit_noise=install(student,'navtrain')
    model,broadcast=wrap(student)
    run=ROOT;rank_folder=run/'ranks'/('rank%02d'%a.rank)
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
        dataset=TeacherDataset(local[offset:],noise_epoch=epoch)
        generator=torch.Generator().manual_seed(config['loader_seed']+1000*epoch+a.rank)
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
            if a.max_steps and last and step>=a.max_steps:
                reference=checkpoint(student,optimizer,config,config_sha,run,epoch,position+1,step,a.rank,a.world_size,best,exposure,all_pools)
                dist.barrier()
                if a.rank==0:atomic(ROOT/'SMOKE_COMPLETE.json',dict(status='ENGINEERING_ONLY_REAL_TRAINING_STEPS_COMPLETE',steps=step,checkpoint=reference))
                dist.destroy_process_group();return
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
    p=argparse.ArgumentParser();p.add_argument('--paths',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--group',choices=['H','T','HT'],required=True);p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1)
    p.add_argument('--global-batch',type=int,default=64);p.add_argument('--epochs',type=int,default=5);p.add_argument('--max-steps',type=int,default=0)
    p.add_argument('--overrides');p.add_argument('--resume');p.add_argument('--smoke-indices');p.add_argument('--stop-after-epoch',type=int,default=0);a=p.parse_args();a.run=Path(a.out).name
    try:main(a)
    except BaseException as error:
        atomic(ROOT/'failures'/('rank%02d.json'%a.rank),dict(status='FAILED',time=stamp(),rank=a.rank,error=repr(error),traceback=traceback.format_exc()));raise
