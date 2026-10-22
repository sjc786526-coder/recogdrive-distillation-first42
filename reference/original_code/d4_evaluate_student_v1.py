"""Private trained student inference, matched real scoring, no online selection."""
from d4_common_v2 import *
from functools import lru_cache
import io

PRIVATE=None
PRIVATE_KEY=None
PRIVATE_PARAMETERS=None


@lru_cache(maxsize=8)
def document(path,expected):
    assert sha(path)==expected
    return read(path)


@lru_cache(maxsize=2)
def population(cohort):
    import numpy as np
    summary=read(B3/'analysis'/('first42_'+cohort+'_v1')/'SUMMARY.json')
    assert summary['status']=='PASS_COMPLETE_AUDITED_FIRST42' and summary['cases']==(103288 if cohort=='navtrain' else 12146)
    rows=read(ROOT/'manifests/R2_TRAINING_SCENES_V2.json') if cohort=='navtrain' else read(summary['rows'])
    arrays={name:np.load(B3/summary['artifacts'][name+'.npy']['path'],mmap_mode='r',allow_pickle=False) for name in ['scores','trajectories','selected']}
    assert len(rows)==summary['cases'] and arrays['scores'].shape==(len(rows),42,5,7)
    return rows,arrays,summary


def load_student(task):
    import torch
    from d4_student_v1 import Student
    global PRIVATE,PRIVATE_KEY,PRIVATE_PARAMETERS
    key=(task['checkpoint'],task['checkpoint_sha256'],task['training_config_sha256'])
    if key==PRIVATE_KEY:return PRIVATE
    config=document(task['training_config'],task['training_config_sha256'])
    for name,hashed in config['code_hashes'].items():assert sha(ROOT/'code'/name)==hashed
    assert sha(task['checkpoint'])==task['checkpoint_sha256']
    state=torch.load(task['checkpoint'],map_location='cpu',weights_only=False)
    assert state['configuration_sha256']==task['training_config_sha256']
    assert state['original_checkpoint_sha256']==read(B3/'teachers/first42_v1/PACK.json')['checkpoint_sha256']
    if PRIVATE is None:
        PRIVATE=Student(config);PRIVATE_PARAMETERS={n:p for n,p in PRIVATE.named_parameters() if p.requires_grad}
    assert PRIVATE.config['lora_rank']==config['lora_rank'] and PRIVATE.config['lora_alpha']==config['lora_alpha']
    assert set(state['trainable'])==set(PRIVATE_PARAMETERS)
    with torch.no_grad():
        for name,p in PRIVATE_PARAMETERS.items():
            value=state['trainable'][name];assert value.shape==p.shape and value.dtype==torch.float32 and torch.isfinite(value).all()
            p.copy_(value)
    PRIVATE.eval();PRIVATE.verify_frozen();PRIVATE_KEY=key
    del state
    return PRIVATE


def release_private_student():
    import torch
    global PRIVATE,PRIVATE_KEY,PRIVATE_PARAMETERS
    PRIVATE=None;PRIVATE_KEY=None;PRIVATE_PARAMETERS=None
    torch.cuda.empty_cache()


def fetch(cohort,index,valid_only=False):
    import numpy as np
    from vlm_core import raw_features
    rows,tables,summary=population(cohort);row=rows[index]
    if cohort=='navtrain' and not valid_only:
        record=read(ROOT/'runs/full_context_replay_v1/cases'/row['token']/'RESULT.json')
        assert record['status']=='PASS_EXACT_R2_FULL_CONTEXT_REPLAY' and record['global_index']==index
        data=retry(lambda:(ROOT/record['arrays']['path']).read_bytes());assert hashlib.sha256(data).hexdigest()==record['arrays']['sha256']
        with np.load(io.BytesIO(data),allow_pickle=False) as z:
            target=dict(hidden=np.array(z['teacher_interface']),positions=np.arange(len(z['input_ids'])),input_ids=np.array(z['input_ids']),
                attention_mask=np.array(z['attention_mask']),residual_squared_sum=record['teacher_residual_squared_sum'],scope='full_context')
    else:
        old_path=B3/'runs'/summary['stage']/'cases'/row['token']/'RESULT.json';old=read(old_path)
        assert old['status']=='COMPLETE' and old['global_index']==index
        ident=old['selection']['R2'][0]['arm'];reference=old['teacher_records'][ident+'_S0']
        path=B3/old['interfaces']['path'];assert sha(path)==old['interfaces']['sha256']
        with np.load(path,allow_pickle=False) as z:
            ids=np.array(z['input_ids']);positions=np.array(z['valid_token_positions']);attention=np.zeros(len(ids),dtype=np.int64);attention[positions]=1
            target=dict(hidden=np.array(z[reference['interface_array']]),positions=positions,input_ids=ids,attention_mask=attention,
                residual_squared_sum=None,scope='valid_tokens_only_for_test_diagnostic')
    metric_sha=row.get('metric_sha256') or sha(row['metric_cache'])
    teacher_index=int(tables['selected'][index,5])
    return dict(row=row,features=raw_features(row),target=target,metric_sha256=metric_sha,index=index,
        old_Base_quality=np.array(tables['scores'][index,0]),old_Base_trajectories=np.array(tables['trajectories'][index,0]),
        old_R2_quality=np.array(tables['scores'][index,teacher_index]),old_R2_trajectories=np.array(tables['trajectories'][index,teacher_index]))


def run(engine,task,pool):
    import numpy as np
    import torch
    from r3_runtime import prediction
    from r3_incremental_audit_v1 import FIELDS
    from repair_score import score_group
    from d4_prefetch_v1 import PrefetchLoader
    student=load_student(task);noises=task['noises'];assert len(noises)==len(set(noises)) and set(noises)<=set(range(10))
    assert not task.get('valid_only_engineering',False) or task.get('engineering',False)
    assert task['cohort'] in ['navtrain','navtest']
    if task['cohort']=='navtest' and not task.get('engineering',False):
        lock=document(task['test_lock'],task['test_lock_sha256']);assert lock['status']=='FROZEN_MODEL_BEFORE_TEST'
        assert lock['checkpoint_sha256']==task['checkpoint_sha256'] and lock['training_config_sha256']==task['training_config_sha256']
    completed=[];pending=[];began=time.monotonic()
    class Inputs:
        def __len__(self):return len(task['indices'])
        def __getitem__(self,offset):return fetch(task['cohort'],task['indices'][offset],task.get('valid_only_engineering',False))
    def finish(item):
        future,data,student_traj,base_traj,hmeta=item;scored=future.result(timeout=3600)
        lookup={(r['arm'],r['seed_index']):r for r in scored}
        q=np.asarray([[lookup['student',j]['score'][f] for f in FIELDS] for j in noises],dtype=np.float64)
        b=np.asarray([data['old_Base_quality'][j] if j<5 else [lookup['Base_new',j]['score'][f] for f in FIELDS] for j in noises],dtype=np.float64)
        if any(j>=5 for j in noises):assert [lookup['Base_replay_check',0]['score'][f] for f in FIELDS]==data['old_Base_quality'][0].tolist()
        assert q.shape==(len(noises),7) and np.isfinite(q).all() and np.isfinite(b).all()
        for values in [q,b]:
            assert (values>=0).all() and (values<=1+1e-8).all()
            assert np.max(np.abs(values[:,1]*values[:,2]*(5*values[:,4]+5*values[:,3]+2*values[:,5])/12-values[:,0]))<1e-8
        row=data['row'];output=ROOT/task['output_stage']/'cases'/row['token'];numeric=output/'EVALUATION.npz'
        hashed=save_arrays(numeric,dict(student_quality=q,Base_quality=b,student_trajectories=student_traj,Base_trajectories=base_traj,
            noise_indices=np.asarray(noises,dtype=np.int64),teacher_R2_reference_S0S4=data['old_R2_quality'],
            input_ids=data['target']['input_ids'],attention_mask=data['target']['attention_mask']))
        path=output/'RESULT.json'
        record=dict(status='COMPLETE_STUDENT_NATIVE_EVALUATION',time=stamp(),task_id=task['id'],global_index=data['index'],token=row['token'],session=row['session'],
            subset=row.get('subset',task['cohort']),pool=row.get('pool','test_not_training'),cohort=task['cohort'],noises=noises,fields=FIELDS,
            checkpoint_sha256=task['checkpoint_sha256'],training_config_sha256=task['training_config_sha256'],model_id=task['model_id'],
            arrays=dict(path=str(numeric.relative_to(ROOT)),sha256=hashed),interface=hmeta,student_inference_uses_teacher_or_PDMS=False,
            GT_used=False,all_student_scores_fresh=True,old_Base_outputs_from_immutable_first42=True,new_Base_scores_fresh=True,
            original_Base_S0_replayed_exactly_when_new_noises=any(j>=5 for j in noises),original_Base_checkpoint_sha256=engine.planner_runtime['checkpoint_sha256'],
            teacher_reference_noise_indices=[0,1,2,3,4],teacher_reference_not_for_new_noise_absorption=True)
        atomic(path,record);completed.append(dict(global_index=data['index'],token=row['token'],path=str(path.relative_to(ROOT)),sha256=sha(path),outputs=len(noises)))
    with torch.no_grad():
        for data in PrefetchLoader(Inputs(),workers=1,prefetch_factor=2,timeout=120):
            row=data['row'];path=ROOT/task['output_stage']/'cases'/row['token']/'RESULT.json'
            if path.exists():
                old=read(path);assert old['checkpoint_sha256']==task['checkpoint_sha256'] and old['noises']==noises and old['status']=='COMPLETE_STUDENT_NATIVE_EVALUATION'
                assert sha(ROOT/old['arrays']['path'])==old['arrays']['sha256']
                completed.append(dict(global_index=data['index'],token=row['token'],path=str(path.relative_to(ROOT)),sha256=sha(path),outputs=len(noises),resumed=True));continue
            hidden={};handle=student.agent.action_head.feature_encoder.register_forward_pre_hook(lambda module,args:hidden.update(value=args[0]))
            try:out=student(data['features'],row['token'],noises[0],trajectory=False)
            finally:handle.remove()
            assert np.array_equal(out['input_ids'][0].cpu().numpy(),data['target']['input_ids'])
            assert np.array_equal(out['attention_mask'][0].cpu().numpy(),data['target']['attention_mask'])
            h=out['interface'];full=h[0].float().cpu().numpy();target=data['target'];difference=full[target['positions']].astype(np.float64)-target['hidden'].astype(np.float64)
            assert np.isfinite(full).all() and np.isfinite(difference).all()
            mse=float(np.square(difference).mean());den=target['residual_squared_sum']
            trajectories=torch.stack([student.sample(data['features'],hidden['value'],h,row['token'],j)[0] for j in noises]).float().cpu().numpy()
            assert trajectories.shape==(len(noises),8,3) and np.isfinite(trajectories).all()
            raw=[dict(arm='student',seed_index=j,seed=seed(row['token'],j),trajectory=trajectories[k].tolist()) for k,j in enumerate(noises)]
            base_new={}
            if any(j>=5 for j in noises):
                base_hidden=engine.forward_native(data['features'])
                check,_=prediction(engine,data['features'],base_hidden,0,row['token'],capture=False)
                check=check.float().cpu().numpy();assert np.array_equal(check,data['old_Base_trajectories'][0])
                raw.append(dict(arm='Base_replay_check',seed_index=0,seed=seed(row['token'],0),trajectory=check.tolist()))
                if den is None:
                    base_h=engine.prefix['interface_post'][0].float().cpu().numpy()[target['positions']]
                    den=float(np.square(base_h.astype(np.float64)-target['hidden'].astype(np.float64)).sum())
                for j in noises:
                    if j<5:continue
                    value,_=prediction(engine,data['features'],base_hidden,j,row['token'],capture=False);base_new[j]=value.float().cpu().numpy()
                    raw.append(dict(arm='Base_new',seed_index=j,seed=seed(row['token'],j),trajectory=base_new[j].tolist()))
            baseline=np.asarray([data['old_Base_trajectories'][j] if j<5 else base_new[j] for j in noises],dtype=np.float32)
            hmeta=dict(sha256=hashlib.sha256(full.tobytes()).hexdigest(),shape=list(full.shape),teacher_comparison_scope=target['scope'],
                MSE=mse,relative_error=float(np.sqrt(np.square(difference).sum()/den)) if den is not None and den>0 else None,
                same_actual_forward_for_all_student_trajectories=True,full_interface_replayable_from_checkpoint=True)
            future=pool.submit(score_group,dict(metric_cache=row['metric_cache'],metric_sha256=data['metric_sha256'],scoring=engine.cfg['scoring'],records=raw))
            pending.append((future,data,trajectories,baseline,hmeta))
            if len(pending)>=2:finish(pending.pop(0))
        for item in pending:finish(item)
    engine.verify_engine();student.verify_frozen()
    result=ROOT/task['output_stage']/'results'/(task['id']+'.json')
    atomic(result,dict(status='COMPLETE_STUDENT_EVALUATION_CHUNK',time=stamp(),task=task,cases=completed,seconds=time.monotonic()-began))
    return dict(path=str(result),sha256=sha(result),cases=len(completed),outputs=len(completed)*len(noises),seconds=time.monotonic()-began)
