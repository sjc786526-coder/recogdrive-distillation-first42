"""Verify and unpack only the frozen small teacher-replay parameter closure."""
from pathlib import Path
import argparse,hashlib,json,os,tarfile,uuid
REPO=Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def prepare(data,archive=None):
    data=Path(data).resolve();spec=json.loads((REPO/'configs/teacher_replay_v1.json').read_text())
    root=data/'teacher_replay_v1';root.mkdir(parents=True,exist_ok=True)
    assert root.resolve().is_relative_to(data),'Replay assets may not redirect outside the dataset'
    bundle=Path(archive).resolve() if archive else data/spec['bundle']['file']
    assert bundle.is_file(),'Download only '+spec['bundle']['file']+' from the HF dataset'
    assert bundle.stat().st_size==spec['bundle']['bytes'] and sha(bundle)==spec['bundle']['sha256'],'Replay bundle identity mismatch'
    expected={v['file']:v for v in spec['assets']+spec.get('goldens',[])};seen=set()
    with tarfile.open(bundle,'r:gz') as tar:
        for member in tar:
            assert member.isfile() and member.name in expected and member.name not in seen,member.name
            value=expected[member.name];assert member.size==value['bytes'];dest=(root/member.name).resolve()
            assert dest.is_relative_to(root.resolve());dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert dest.stat().st_size==value['bytes'] and sha(dest)==value['sha256'];seen.add(member.name);continue
            tmp=dest.with_name(dest.name+'.'+uuid.uuid4().hex+'.tmp');h=hashlib.sha256()
            with tar.extractfile(member) as src,tmp.open('xb') as out:
                for block in iter(lambda:src.read(8<<20),b''):h.update(block);out.write(block)
            assert h.hexdigest()==value['sha256'];tmp.replace(dest);seen.add(member.name)
    assert seen==set(expected)
    for name,hashed in spec['scientific_code_sha256'].items():assert sha(REPO/name)==hashed,('Scientific code changed',name)
    result=dict(status='PASS_ALL_REPLAY_PARAMETERS_AND_CODE',candidate_entries=42,unique_parameter_files=len(spec['assets']),golden_scenes=len(spec.get('goldens',[])),bundle_sha256=spec['bundle']['sha256'])
    print(json.dumps(result),flush=True);return spec
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--archive');a=p.parse_args();prepare(a.data,a.archive)
