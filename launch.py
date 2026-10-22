"""Configurable cards/nodes; one visible GPU per native worker, no cloud SDK."""
from pathlib import Path
import argparse,os,subprocess,sys,time,json
ROOT=Path(__file__).resolve().parent
def main(a):
    devices=a.devices.split(',');world=len(devices)*a.nodes;assert world>0 and a.global_batch%world==0
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    config=dict(group=a.group,devices=devices,nodes=a.nodes,node_rank=a.node_rank,world_size=world,global_batch=a.global_batch)
    marker=out/('LAUNCH_NODE%02d.json'%a.node_rank);assert not marker.exists();marker.write_text(json.dumps(config,indent=2)+'\n')
    processes=[];streams=[]
    for offset,device in enumerate(devices):
        rank=a.node_rank*len(devices)+offset;env=dict(os.environ,CUDA_VISIBLE_DEVICES=device,MASTER_ADDR=a.master_addr,MASTER_PORT=str(a.master_port),RANK=str(rank),WORLD_SIZE=str(world),LOCAL_RANK='0',PYTHONDONTWRITEBYTECODE='1')
        log=(out/('rank%02d.log'%rank)).open('x');streams.append(log)
        cmd=[sys.executable,'-B',str(ROOT/'train.py'),'--paths',str(Path(a.paths).resolve()),'--data',str(Path(a.data).resolve()),'--out',str(out),'--group',a.group,'--rank',str(rank),'--world-size',str(world),'--global-batch',str(a.global_batch),'--epochs',str(a.epochs)]
        if a.overrides:cmd+=['--overrides',str(Path(a.overrides).resolve())]
        if a.max_steps:cmd+=['--max-steps',str(a.max_steps)]
        if a.resume:cmd+=['--resume',str(Path(a.resume).resolve())]
        processes.append(subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT))
    while any(p.poll() is None for p in processes):
        failed=[p.returncode for p in processes if p.poll() is not None and p.returncode!=0]
        if failed:
            # Only Popen handles spawned by THIS launcher are terminated.
            for p in processes:
                if p.poll() is None:p.terminate()
            raise RuntimeError('One owned rank failed; inspect rank logs: '+str(failed))
        time.sleep(5)
    if any(p.returncode for p in processes):raise RuntimeError('Training rank failed')
    for f in streams:f.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paths',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--group',choices=['H','T','HT'],required=True)
    p.add_argument('--devices',default='0');p.add_argument('--nodes',type=int,default=1);p.add_argument('--node-rank',type=int,default=0);p.add_argument('--master-addr',default='127.0.0.1');p.add_argument('--master-port',type=int,default=29622)
    p.add_argument('--global-batch',type=int,default=64);p.add_argument('--epochs',type=int,default=5);p.add_argument('--overrides');p.add_argument('--max-steps',type=int,default=0);p.add_argument('--resume');main(p.parse_args())
