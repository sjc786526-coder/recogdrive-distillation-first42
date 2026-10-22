"""Fresh official PDM scoring; standalone process supports original-source comparison."""
from common import *
import argparse
import dataclasses
import lzma
import pickle


def init_scorer():
    setup(f'scorer/{os.getpid()}')
    compatibility()


def score_one(task):
    import numpy as np
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score, get_trajectory_as_array
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    metric = Path(task['metric_cache'])
    def load():
        data = metric.read_bytes()
        assert hashlib.sha256(data).hexdigest() == task['metric_sha256']
        return pickle.loads(lzma.decompress(data))
    cache = retry(load)
    poses = np.asarray(task['trajectory'], dtype=np.float32)
    assert poses.shape == (8,3) and np.isfinite(poses).all()
    sampling = TrajectorySampling(**task['scoring']['proposal_sampling'])
    reference = get_trajectory_as_array(cache.trajectory, sampling, cache.ego_state.time_point)
    result = pdm_score(cache, Trajectory(poses, TrajectorySampling(num_poses=8, interval_length=.5)),
                       sampling, PDMSimulator(sampling), PDMScorer(sampling, PDMScorerConfig(**task['scoring']['scorer'])))
    values = {k: float(v) for k,v in dataclasses.asdict(result).items()}
    assert all(np.isfinite(v) and 0 <= v <= 1+1e-8 for v in values.values())
    check = values['no_at_fault_collisions'] * values['drivable_area_compliance'] * (
        5*values['ego_progress'] + 5*values['time_to_collision_within_bound'] + 2*values['comfort']) / 12
    assert abs(values['score'] - check) < 1e-8
    return dict(token=task['token'], valid=True, score=values, quality=quality(values['score']),
                metric_sha256=task['metric_sha256'], reference_sha256=hashlib.sha256(reference.tobytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('request', type=Path)
    parser.add_argument('response', type=Path)
    parser.add_argument('--original-source', type=Path)
    args = parser.parse_args()
    init_scorer()
    if args.original_source:
        sys.path.insert(0, str(args.original_source))
    tasks = read(args.request)
    result = [score_one(task) for task in tasks]
    import navsim.evaluate.pdm_score
    import importlib
    origin = importlib.import_module('navsim.evaluate.pdm_score').__file__
    expected = args.original_source if args.original_source else UPSTREAM
    assert Path(origin).resolve().is_relative_to(expected.resolve())
    atomic(args.response, dict(status='PASS', python=sys.executable, import_source=origin,
                             numpy_version=__import__('numpy').__version__, results=result))


if __name__ == '__main__':
    main()
