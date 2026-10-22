"""Frozen RL planner inference, per-scene RNG, asynchronous independent scoring."""
from common import *
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import importlib.metadata
import io
import multiprocessing as mp
import pickle
import subprocess
import traceback


def population():
    config = read(ROOT / 'configs/baseline.json')
    assert sha(ROOT / 'manifests/navtrain.csv') == config['manifest_sha256']
    rows = list(csv.DictReader((ROOT / 'manifests/navtrain.csv').open()))
    assert len(rows) == config['samples']
    return rows


def load_features(row, config):
    import torch
    folder = Path(row['feature_dir'])
    identity = read(folder / 'complete.json')
    assert identity['token'] == row['token']
    assert identity['vlm_sha256'] == config['vlm_sha256']
    assert identity['protocol_sha256'] == config['cache_protocol_sha256']
    assert identity['cache_policy_sha256'] == config['cache_policy_sha256']
    data = retry(lambda: (folder / 'internvl_feature.gz').read_bytes())
    feature_hash = hashlib.sha256(data).hexdigest()
    features = pickle.loads(gzip.decompress(data))
    assert features['last_hidden_state'].ndim == 2 and features['last_hidden_state'].shape[-1] == 1536
    assert tuple(features['history_trajectory'].shape) == (4,3)
    assert tuple(features['status_feature'].shape) == (8,)
    assert tuple(features['high_command_one_hot'].shape) == (4,)
    assert all(isinstance(v,torch.Tensor) and v.dtype == torch.float32 and torch.isfinite(v).all() for v in features.values())
    target = retry(lambda: (folder / 'trajectory_target.gz').read_bytes())
    gt = pickle.loads(gzip.decompress(target))['trajectory']
    assert tuple(gt.shape) == (8,3) and torch.isfinite(gt).all()
    return features, gt, dict(feature_sha256=feature_hash, target_sha256=hashlib.sha256(target).hexdigest(),
                              feature_identity_sha256=digest(identity), feature_dir=str(folder))


def load_model(config):
    import torch
    from model.recogdrive_agent import ReCogDriveAgent
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    assert sha(config['checkpoint']) == config['checkpoint_sha256']
    agent = ReCogDriveAgent(TrajectorySampling(time_horizon=4, interval_length=.5), checkpoint_path=None,
                            vlm_path=None, cache_hidden_state=True, cache_mode=False, dit_type='small',
                            vlm_size='small', sampling_method='ddim', grpo=False, train_backbone=False)
    checkpoint = torch.load(config['checkpoint'], map_location='cpu', weights_only=False)
    state = {k.removeprefix('agent.'):v for k,v in checkpoint['state_dict'].items()}
    # Released RL checkpoints also serialize GRPO's training-only old policy.
    # Require every active tensor exactly; omit only this explicit namespace.
    excluded = {k:v for k,v in state.items() if k.startswith('action_head.old_policy.')}
    active = {k:v for k,v in state.items() if not k.startswith('action_head.old_policy.')}
    assert set(active) == set(agent.state_dict()),dict(missing=sorted(set(agent.state_dict())-set(active)),unexpected=sorted(set(active)-set(agent.state_dict())))
    result = agent.load_state_dict(active, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    metadata = {k:checkpoint[k] for k in ['epoch','global_step'] if k in checkpoint}
    del checkpoint
    agent.float().eval().requires_grad_(False)
    head = agent.action_head
    assert head.ddim_t.tolist() == config['expected_timesteps']
    assert head.ddim_steps == config['denoising_steps']
    eta = float(head.eta(torch.zeros((1,8,3),device='cuda'))[0,0])
    assert eta == 1.
    runtime = dict(time=stamp(), python=sys.executable, gpu=torch.cuda.get_device_name(0),
                   memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                   planner_dtype='float32', autocast=False, batch_size=1, mode='eval', deterministic=False,
                   effective_eta=eta, ddim_t=head.ddim_t.tolist(), num_train_timesteps=head.ddpm_num_train_timesteps,
                   checkpoint_sha256=config['checkpoint_sha256'], checkpoint_metadata=metadata,
                   strict_loaded_tensors=len(agent.state_dict()),
                   excluded_training_old_policy_tensors=len(excluded),
                   clipping={k:float(getattr(head,k,default)) for k,default in [('denoised_clip_value',1.),
                     ('final_action_clip_value',1.),('eval_randn_clip_value',1.),('eval_min_sampling_denoising_std',.0001)]},
                   ddim_buffers_sha256={k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()
                     for k,v in head.named_buffers() if k.startswith('ddim_')},
                   versions={n:importlib.metadata.version(n) for n in ['torch','torchvision','numpy','transformers','timm','diffusers','scipy','nuplan-devkit']},
                   tf32=dict(matmul=torch.backends.cuda.matmul.allow_tf32,cudnn=torch.backends.cudnn.allow_tf32))
    assert all(p.dtype == torch.float32 and not p.requires_grad for p in agent.parameters())
    return agent, runtime


def predict(agent, features, value):
    import numpy as np
    import torch
    batch = {k:v.unsqueeze(0).cuda().float() for k,v in features.items()}
    reseed(value)
    with torch.no_grad(), torch.autocast('cuda',enabled=False):
        trajectory = agent.forward(batch)['pred_traj'].float().cpu().squeeze(0).numpy()
    assert trajectory.shape == (8,3) and np.isfinite(trajectory).all()
    return trajectory


def output_dir(run, phase, rank):
    return ROOT / 'runs' / run / phase / f'rank{rank:03d}'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--phase',choices=['dsw_smoke','smoke','formal'],required=True)
    parser.add_argument('--rank',type=int,default=0)
    parser.add_argument('--world-size',type=int,default=1)
    parser.add_argument('--chunk-size',type=int,default=32)
    parser.add_argument('--scoring-workers',type=int,default=2)
    args=parser.parse_args()
    setup(f'{args.run}/{args.phase}/rank{args.rank:03d}')
    compatibility()
    import torch
    import numpy as np
    from score import init_scorer,score_one
    output=output_dir(args.run,args.phase,args.rank)
    output.mkdir(parents=True,exist_ok=True)
    try:
        config=read(ROOT/'configs/baseline.json')
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        atomic(output/'status.json',dict(status='LOADING_MODEL',rank=args.rank,time=stamp()))
        agent,runtime=load_model(config)
        atomic(output/'runtime.json',runtime)
        parameter_versions={k:v._version for k,v in agent.named_parameters()}
        rows=population()
        if args.phase == 'formal':
            panel=rows[args.rank::args.world_size]
        else:
            panel=rows[:2]+rows[2+2*args.rank:4+2*args.rank]
        began=time.monotonic()
        completed=0
        all_smoke=[]
        with ProcessPoolExecutor(max_workers=args.scoring_workers,mp_context=mp.get_context('spawn'),initializer=init_scorer) as pool:
            for offset in range(0,len(panel),args.chunk_size):
                group=panel[offset:offset+args.chunk_size]
                destination=output/'chunks'/f'{offset:06d}.json'
                if destination.exists():
                    cached=read(destination)
                    assert cached['identity'] == config['identity']
                    assert [r['token'] for r in cached['records']] == [r['token'] for r in group]
                    assert all(r['valid'] and r['seed']==seed(r['token']) for r in cached['records'])
                    completed+=len(group)
                    continue
                pending=[]
                tasks=[]
                for row in group:
                    features,gt,meta=load_features(row,config)
                    value=seed(row['token'])
                    assert value == int(row['seed'])
                    trajectory=predict(agent,features,value)
                    repeated=None
                    if args.phase != 'formal':
                        repeated=predict(agent,features,value)
                        assert np.array_equal(trajectory,repeated),row['token']
                    task=dict(token=row['token'],trajectory=trajectory.tolist(),metric_cache=row['metric_cache'],
                              metric_sha256=sha(row['metric_cache']),scoring=config['scoring'])
                    future=pool.submit(score_one,task)
                    record=dict(token=row['token'],log_name=row['log_name'],raw_path=row['raw_path'],
                                frame_index=int(row['frame_index']),seed=value,seed_index=0,
                                checkpoint_sha256=config['checkpoint_sha256'],identity=config['identity'],
                                rank=args.rank,trajectory=trajectory.tolist(),metric_cache=row['metric_cache'],**meta)
                    pending.append((future,record))
                    tasks.append(task)
                    if args.phase != 'formal':
                        all_smoke.append(dict(task,trajectory=gt.float().numpy().tolist(),source='GT'))
                records=[]
                for future,record in pending:
                    record.update(future.result(timeout=300))
                    records.append(record)
                atomic(destination,dict(identity=config['identity'],records=records))
                all_smoke.extend(tasks)
                completed+=len(records)
                atomic(output/'status.json',dict(status='RUNNING',rank=args.rank,completed=completed,requested=len(panel),
                        seconds=time.monotonic()-began,last_chunk=str(destination),time=stamp(),identity=config['identity']))
                print(json.dumps(dict(rank=args.rank,phase=args.phase,completed=completed,total=len(panel),seconds=time.monotonic()-began)),flush=True)
        assert {k:v._version for k,v in agent.named_parameters()} == parameter_versions
        assert all(p.grad is None for p in agent.parameters())
        if args.phase != 'formal':
            request=output/'score_request.json'
            atomic(request,all_smoke)
            for name,extra in [('local',[]),('original',['--original-source',config['shared_reference_source']])]:
                subprocess.run([sys.executable,'-B',str(ROOT/'code/score.py'),str(request),str(output/f'{name}_scores.json'),*extra],
                               check=True,cwd=ROOT,timeout=600)
            local=read(output/'local_scores.json');original=read(output/'original_scores.json')
            assert local['results'] == original['results']
            by_token={r['token']:r for r in records}
            for task,result in zip(all_smoke,local['results']):
                if task.get('source') != 'GT':
                    assert by_token[task['token']]['score'] == result['score']
            atomic(output/'SCORING_EQUIVALENCE.json',dict(status='PASS',samples=len(all_smoke),max_score_difference=0,
                   local_source=local['import_source'],original_source=original['import_source'],python=sys.executable,
                   numpy_compatibility='numpy._core.numeric -> numpy.core.numeric only',test_cache_access=False))
        result=dict(status='COMPLETE',rank=args.rank,phase=args.phase,completed=completed,requested=len(panel),
                    identity=config['identity'],seconds=time.monotonic()-began,time=stamp(),all_parameters_fp32=True,
                    no_gradients=True,parameter_versions_unchanged=True)
        atomic(output/'COMPLETE.json',result)
        atomic(output/'status.json',result)
    except BaseException as error:
        atomic(output/'FAILED.json',dict(status='FAILED',rank=args.rank,phase=args.phase,error=repr(error),
                                        traceback=traceback.format_exc(),time=stamp()))
        raise


if __name__ == '__main__':
    main()
