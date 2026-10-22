"""Analyze copied per-scene scores without model loading or original filesystem."""
from pathlib import Path
import argparse,json,math,csv
def main(a):
    import numpy as np
    root=Path(a.data);output=Path(a.out);output.mkdir(parents=True,exist_ok=True);results=[]
    for folder in sorted((root/'results').iterdir()):
        if not folder.is_dir() or not (folder/'SUMMARY.source.json').exists():continue
        source=json.loads((folder/'SUMMARY.source.json').read_text());q=np.load(folder/'student_quality.npy',mmap_mode='r');b=np.load(folder/'Base_quality.npy',mmap_mode='r')
        assert q.shape==b.shape;noise=source['noises'];rows=json.loads((folder/'CASE_INDEX.json').read_text())
        for name,allowed in [('S0S4',range(5)),('S5S9',range(5,10))]:
            js=[k for k,j in enumerate(noise) if j in allowed]
            if not js:continue
            groups={'all':np.arange(len(rows))}
            for pool in ['I','P','E']:
                ix=np.asarray([i for i,r in enumerate(rows) if r['reference_pool']==pool],dtype=np.int64)
                if len(ix):groups[pool]=ix
            for group,indices in groups.items():
                x=np.asarray(q[indices][:,js]);y=np.asarray(b[indices][:,js])
                pdms=math.fsum(float(v) for v in x[:,:,0].ravel())/x.shape[0]/len(js)*100
                base=math.fsum(float(v) for v in y[:,:,0].ravel())/y.shape[0]/len(js)*100
                results.append(dict(model_id=source['model_id'],scope=folder.name,noise=name,pool=group,scenes=len(indices),PDMS=pdms,Base_PDMS=base,gain=pdms-base,
                    zero_outputs=int((x[:,:,0]==0).sum()),failures_below70=int((x[:,:,0]<.7).sum()),Base_high_to_failure=int(((y[:,:,0]>.9)&(x[:,:,0]<.7)).sum())))
    assert results
    path=output/'scores.csv'
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    print(json.dumps(dict(status='ANALYZED_COPIED_ACTUAL_RESULTS',rows=len(results),output=str(path))),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);main(p.parse_args())
