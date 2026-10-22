"""Configurable independent replay ranks; no DLC dependency or distributed init."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
REPO=Path(__file__).resolve().parent
def main(a):
    devices=a.devices.split(',');world=len(devices)*a.nodes;assert devices and a.nodes>0 and 0<=a.node_rank<a.nodes
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True);streams=[];children=[]
    marker=out/('TEACHER_LAUNCH_NODE%02d.json'%a.node_rank)
    config=dict(devices=devices,nodes=a.nodes,node_rank=a.node_rank,world_size=world)
    if marker.exists():assert json.loads(marker.read_text())==config
    else:marker.write_text(json.dumps(config,indent=2)+'\n')
    try:
        for offset,device in enumerate(devices):
            rank=a.node_rank*len(devices)+offset;env=dict(os.environ,CUDA_VISIBLE_DEVICES=device,PYTHONDONTWRITEBYTECODE='1')
            stream=(out/('teacher_rank%04d.log'%rank)).open('a');streams.append(stream)
            cmd=[sys.executable,'-B',str(REPO/'replay_teacher.py'),'--paths',str(Path(a.paths).resolve()),'--data',str(Path(a.data).resolve()),'--out',str(out),'--rank',str(rank),'--world-size',str(world),'--scoring-workers',str(a.scoring_workers)]
            if a.indices:cmd+=['--indices',str(Path(a.indices).resolve())]
            if a.limit:cmd+=['--limit',str(a.limit)]
            if a.golden_only:cmd+=['--golden-only']
            if a.strict_original:cmd+=['--strict-original']
            children.append(subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in children):
            if any(p.poll() is not None and p.returncode for p in children):raise RuntimeError('A replay rank failed; inspect its dedicated log')
            time.sleep(5)
        assert all(p.returncode==0 for p in children)
    finally:
        for p in children:
            if p.poll() is None:p.terminate()
        for p in children:
            if p.poll() is None:
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:p.kill();p.wait()
        for stream in streams:stream.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paths',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--devices',default='0');p.add_argument('--nodes',type=int,default=1);p.add_argument('--node-rank',type=int,default=0)
    p.add_argument('--scoring-workers',type=int,default=1);p.add_argument('--indices');p.add_argument('--limit',type=int,default=0);p.add_argument('--golden-only',action='store_true');p.add_argument('--strict-original',action='store_true');main(p.parse_args())
