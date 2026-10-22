from typing import Any, Dict
from pathlib import Path
import logging
import os
import pickle
import traceback

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm
import torch
import torch.distributed as dist

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import Trajectory, SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_create_submission_pickle"


class InferenceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, size: int):
        self._size = int(size)
        assert size > 0
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size: int, world_size: int, rank: int):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]
        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[: rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def _broadcast_object(obj: Any, device: torch.device, src: int = 0) -> Any:
    if dist.get_rank() == src:
        buffer = pickle.dumps(obj)
        tensor = torch.ByteTensor(list(buffer)).to(device)
        size_tensor = torch.tensor(len(tensor), device=device)
    else:
        tensor = torch.ByteTensor([]).to(device)
        size_tensor = torch.tensor(0, device=device)

    dist.broadcast(size_tensor, src=src)
    if dist.get_rank() != src:
        tensor = torch.empty(size_tensor.item(), dtype=torch.uint8, device=device)
    dist.broadcast(tensor, src=src)
    return pickle.loads(tensor.cpu().numpy().tobytes())


def _gather_object(obj: Any, device: torch.device):
    buffer = pickle.dumps(obj)
    tensor = torch.ByteTensor(list(buffer)).to(device)
    local_size = torch.tensor(len(tensor), device=device)
    sizes = [torch.tensor(0, device=device) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, local_size)
    max_size = max(size.item() for size in sizes)
    if len(tensor) < max_size:
        tensor = torch.cat([tensor, torch.zeros(max_size - len(tensor), dtype=torch.uint8, device=device)])
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return [pickle.loads(buf[: size.item()].cpu().numpy().tobytes()) for buf, size in zip(gathered, sizes)]


def run_local_submission(
    agent: AbstractAgent,
    scene_filter: SceneFilter,
    data_path: Path,
    sensor_blobs_path: Path,
    tokens_to_evaluate,
) -> Dict[str, Trajectory]:
    if agent.requires_scene:
        raise ValueError("Submission generation requires an agent that does not need annotated Scene access.")

    scene_filter.tokens = tokens_to_evaluate
    input_loader = SceneLoader(
        data_path=data_path,
        scene_filter=scene_filter,
        sensor_blobs_path=sensor_blobs_path,
        sensor_config=agent.get_sensor_config(),
    )

    output: Dict[str, Trajectory] = {}
    for token in tqdm(tokens_to_evaluate, desc=f"Rank {dist.get_rank()} submission", disable=dist.get_rank() != 0):
        try:
            agent_input = input_loader.get_agent_input_from_token(token)
            output[token] = agent.compute_trajectory(agent_input)
        except Exception:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
    return output


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    rank = int(os.getenv("RANK", 0))
    visible_device_count = torch.cuda.device_count()
    default_backend = "nccl" if visible_device_count > 0 else "gloo"
    dist_backend = os.getenv("DIST_BACKEND", default_backend)
    dist.init_process_group(backend=dist_backend, world_size=world_size, rank=rank)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank % visible_device_count)
        comm_device = torch.device(f"cuda:{local_rank % visible_device_count}") if dist_backend == "nccl" else torch.device("cpu")
    else:
        comm_device = torch.device("cpu")

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)
    save_path = Path(cfg.output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        no_sensor_loader = SceneLoader(
            data_path=data_path,
            scene_filter=instantiate(cfg.train_test_split.scene_filter),
            sensor_blobs_path=sensor_blobs_path,
            sensor_config=SensorConfig.build_no_sensors(),
        )
        tokens_to_evaluate = sorted(no_sensor_loader.tokens)
        logger.info(f"Starting DDP submission generation for {len(tokens_to_evaluate)} scenes")
    else:
        tokens_to_evaluate = []

    tokens_to_evaluate = _broadcast_object(tokens_to_evaluate, comm_device, src=0)
    sampler = InferenceSampler(len(tokens_to_evaluate))
    local_tokens = [tokens_to_evaluate[idx] for idx in sampler]

    agent = instantiate(cfg.agent)
    agent.initialize()
    local_output = run_local_submission(
        agent=agent,
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        data_path=data_path,
        sensor_blobs_path=sensor_blobs_path,
        tokens_to_evaluate=local_tokens,
    )

    gathered_outputs = _gather_object(local_output, comm_device)
    if rank == 0:
        output: Dict[str, Trajectory] = {}
        for shard in gathered_outputs:
            output.update(shard)
        output = {token: output[token] for token in tokens_to_evaluate if token in output}
        submission = {
            "team_name": cfg.team_name,
            "authors": cfg.authors,
            "email": cfg.email,
            "institution": cfg.institution,
            "country / region": cfg.country,
            "predictions": [output],
        }
        filename = save_path / "submission.pkl"
        with filename.open("wb") as file:
            pickle.dump(submission, file)
        logger.info(f"Saved {len(output)} predictions to {filename}")
        print(f"Saved {len(output)} predictions to {filename}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
