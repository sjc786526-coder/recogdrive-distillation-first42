from pathlib import Path
import os,sys
REPO=Path(__file__).resolve().parents[1]
def configure(paths,data,out):
    os.environ.update(RECOGDRIVE_PATHS=str(Path(paths).resolve()),RECOGDRIVE_DATA=str(Path(data).resolve()),RECOGDRIVE_RUN_ROOT=str(Path(out).resolve()),PYTHONDONTWRITEBYTECODE='1')
    for p in [REPO/'native/upstream',REPO/'native',REPO/'src']:
        if str(p) not in sys.path:sys.path.insert(0,str(p))
