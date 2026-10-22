"""Download exactly the small replay closure, never the 414GB teacher shards."""
from pathlib import Path
import argparse,hashlib,json
REPO=Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def main(a):
    spec=json.loads((REPO/'configs/teacher_replay_downloads_v1.json').read_text());root=Path(a.data).resolve();checked=0
    for value in spec['files']:
        path=root/value['file']
        if not path.exists():
            assert not a.check_only,'Missing '+value['file']
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id=spec['repo'],repo_type=spec['repo_type'],filename=value['file'],revision=a.revision,local_dir=str(root))
        assert path.stat().st_size==value['bytes'] and sha(path)==value['sha256'],('File identity mismatch',value['file'])
        checked+=1
    print(json.dumps(dict(status='PASS_MINIMAL_REPLAY_DOWNLOAD',files=checked,bytes=spec['total_download_bytes'],teacher_tar_shards_downloaded=False)),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--revision');p.add_argument('--check-only',action='store_true');main(p.parse_args())
