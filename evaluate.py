"""Independent configurable inference and real scoring, without teacher input."""
from pathlib import Path
import argparse,os,sys,json
ROOT_REPO=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT_REPO/'src'))
from portable_setup import configure
def main(a):
    if a.subset is None:a.subset='all' if a.cohort=='navtest' else 'validation'
    if a.cohort=='navtest':assert a.subset=='all','navtest uses --subset all'
    configure(a.paths,a.data,a.out)
    from d4_common import ROOT,REPO,read,sha,atomic,seed,setup,compatibility,stamp
    setup('evaluation');compatibility()
    import numpy as np,torch
    from d4_student_v1 import Student
    from portable_data import local_rows
    from vlm_core import raw_features
    from repair_score import score_group
    from portable_eval import FIELDS
    config=read(REPO/'configs/original'/(a.group+'.json'));student=Student(config);identity='RL_Base_zero_LoRA'
    from packed_noise import install
    explicit_noise=install(student,a.cohort) if not a.noise_bank else False
    if a.checkpoint:
        state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        assert state['original_checkpoint_sha256']==student.identity['checkpoint_sha256']
        parameters={n:p for n,p in student.named_parameters() if p.requires_grad};assert set(parameters)==set(state['trainable'])
        with torch.no_grad():
            for name,p in parameters.items():
                value=state['trainable'][name];assert value.dtype==torch.float32 and value.shape==p.shape and torch.isfinite(value).all();p.copy_(value)
        identity=sha(a.checkpoint);del state
    rows=local_rows(a.cohort)
    indices=list(range(len(rows))) if a.subset=='all' else [i for i,r in enumerate(rows) if r['subset']==a.subset]
    if a.tokens:
        allowed=set(a.tokens.split(','));indices=[i for i in indices if rows[i]['token'] in allowed]
    if a.limit:indices=indices[:a.limit]
    indices=indices[a.rank::a.world_size];noises=[int(x) for x in a.noises.split(',')];assert noises and set(noises)<=set(range(10))
    protocol=dict(checkpoint=identity,cohort=a.cohort,subset=a.subset,noises=noises,rank=a.rank,world_size=a.world_size,explicit_original_device_noise=explicit_noise)
    marker=ROOT/('PROTOCOL_rank%02d.json'%a.rank)
    if marker.exists():assert read(marker)==protocol
    else:atomic(marker,protocol)
    bank=None
    if a.noise_bank:
        from noise_replay import replay_bank
        bank=np.load(a.noise_bank,allow_pickle=False)
    done=[];student.eval()
    with torch.no_grad():
        for index in indices:
            row=rows[index];path=ROOT/'cases'/(row['token']+'.json')
            if path.exists():
                old=read(path);assert old['checkpoint']==identity and old['noises']==noises;done.append(old);continue
            features=raw_features(row);hidden={};handle=student.agent.action_head.feature_encoder.register_forward_pre_hook(lambda module,args:hidden.update(value=args[0]))
            try:out=student(features,row['token'],0,trajectory=False)
            finally:handle.remove()
            predictions=[]
            for j in noises:
                if bank is None:value=student.sample(features,hidden['value'],out['interface'],row['token'],j)[0]
                else:
                    with replay_bank(bank[row['token']+'_s'+str(j)]):value=student.sample(features,hidden['value'],out['interface'],row['token'],j)[0]
                predictions.append(value.float().cpu().numpy())
            trajectories=np.stack(predictions)
            values=score_group(dict(metric_cache=row['metric_cache'],metric_sha256=row['metric_sha256'],scoring=student.scientific['scoring'],
                records=[dict(arm='student',seed_index=j,seed=seed(row['token'],j),trajectory=trajectories[k].tolist()) for k,j in enumerate(noises)]))
            lookup={v['seed_index']:v for v in values};q=np.asarray([[lookup[j]['score'][f] for f in FIELDS] for j in noises])
            record=dict(status='COMPLETE_PORTABLE_STUDENT',time=stamp(),checkpoint=identity,global_index=index,token=row['token'],session=row['session'],subset=row['subset'],noises=noises,
                trajectories=trajectories.tolist(),quality=q.tolist(),GT_used=False,teacher_input_to_student=False)
            atomic(path,record);done.append(record)
    student.verify_frozen();q=np.asarray([r['quality'] for r in done]);assert len(done)==len(indices)
    atomic(ROOT/('COMPLETE_rank%02d.json'%a.rank),dict(status='COMPLETE_PORTABLE_NATIVE_SCORING',cases=len(done),PDMS=float(q[:,:,0].mean()*100),protocol=protocol))
    print(json.dumps(dict(cases=len(done),PDMS=float(q[:,:,0].mean()*100))),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paths',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--group',choices=['H','T','HT'],default='HT');p.add_argument('--checkpoint')
    p.add_argument('--cohort',choices=['navtrain','navtest'],default='navtrain');p.add_argument('--subset',choices=['all','exploration','validation'],default=None);p.add_argument('--tokens');p.add_argument('--limit',type=int,default=0)
    p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1);p.add_argument('--noises',default='0,1,2,3,4');p.add_argument('--noise-bank');main(p.parse_args())
