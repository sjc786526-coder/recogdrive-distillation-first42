"""Original PDM computation, immutable per-scene cache reuse within a scoring process."""
from common import *
from functools import lru_cache
import dataclasses
import lzma
import pickle

@lru_cache(maxsize=4)
def cache_at(path,hashed):
    raw=retry(lambda:Path(path).read_bytes());assert hashlib.sha256(raw).hexdigest()==hashed
    return pickle.loads(lzma.decompress(raw))

def score_group(task):
    import numpy as np
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer,PDMScorerConfig
    cache=cache_at(task['metric_cache'],task['metric_sha256']);sampling=TrajectorySampling(**task['scoring']['proposal_sampling'])
    records=[];scored={}
    for r in task['records']:
        poses=np.asarray(r['trajectory'],dtype=np.float32);assert poses.shape==(8,3) and np.isfinite(poses).all()
        key=poses.tobytes()
        if key in scored:values=scored[key]
        else:
            values=pdm_score(cache,Trajectory(poses,TrajectorySampling(num_poses=8,interval_length=.5)),sampling,
                PDMSimulator(sampling),PDMScorer(sampling,PDMScorerConfig(**task['scoring']['scorer'])))
            values={k:float(v) for k,v in dataclasses.asdict(values).items()};scored[key]=values
        assert all(np.isfinite(v) and 0<=v<=1+1e-8 for v in values.values())
        check=values['no_at_fault_collisions']*values['drivable_area_compliance']*(5*values['ego_progress']+5*values['time_to_collision_within_bound']+2*values['comfort'])/12
        assert abs(check-values['score'])<1e-8
        records.append(dict(r,score=values,quality=quality(values['score'])))
    return records
