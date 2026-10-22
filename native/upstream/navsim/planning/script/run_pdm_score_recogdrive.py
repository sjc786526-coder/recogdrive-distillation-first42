from typing import Any, Dict, List, Union
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import traceback
import logging
import lzma
import pickle
import os
import uuid
import torch
import torch.distributed as dist
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd

from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.metric_caching.metric_cache import MetricCache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Inline low-score plot helpers
# ---------------------------------------------------------------------------

# All camera attribute names in a navsim CameraFrame
_CAM_ATTRS = [
    "cam_l0", "cam_f0", "cam_r0",
    "cam_l1", "cam_r1",
    "cam_l2", "cam_b0", "cam_r2",
]


def _decode_camera_images(scene) -> None:
    """
    When ``SceneLoader.load_image_path=True``, every ``camera.image`` is a
    ``pathlib.Path`` (not a decoded numpy array).  Walk all frames and
    decode any path-based images in-place so matplotlib can render them.
    """
    import cv2
    from pathlib import Path

    for frame in scene.frames:
        for attr in _CAM_ATTRS:
            cam = getattr(frame.cameras, attr, None)
            if cam is None:
                continue
            # load_image_path=True → image is a Path, not an ndarray
            if isinstance(cam.image, (str, Path)):
                img = cv2.imread(str(cam.image))
                if img is not None:
                    cam.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _plot_low_score_inline(
    token: str,
    score_row: Dict[str, Any],
    cfg: DictConfig,
    scene_loader: SceneLoader,
    agent_input,
    trajectory,
    scorer: PDMScorer,
) -> None:
    """
    Load full scene, extract diagnostic from the already-run scorer, and
    save a trajectory-comparison plot — all inline during evaluation.
    """
    from navsim.visualization.low_score_diagnostic import (
        extract_diagnostic_from_scorer,
        plot_single_low_score_scene,
    )

    output_dir = Path(cfg.get("low_score_plot_dir", "./low_score_plots/RL"))

    # Load full scene — when load_image_path=True, camera images are
    # stored as file-path strings; decode them for matplotlib rendering.
    scene = scene_loader.get_scene_from_token(token)
    _decode_camera_images(scene)
    human_trajectory = scene.get_future_trajectory()
    frame_idx = scene.scene_metadata.num_history_frames - 1

    # Extract collision / off-road detail from the scorer (no re-computation)
    diagnostic = extract_diagnostic_from_scorer(scorer, pred_idx=1)

    # Metric columns to display (exclude housekeeping fields)
    metric_cols = [
        k for k in score_row.keys()
        if k not in ("token", "valid", "rank")
    ]

    output_path = plot_single_low_score_scene(
        scene=scene,
        frame_idx=frame_idx,
        human_trajectory=human_trajectory,
        agent_trajectory=trajectory,
        token=token,
        score_row=score_row,
        metric_cols=metric_cols,
        agent_input=agent_input,
        diagnostic=diagnostic,
        output_dir=output_dir,
    )

    logger.info(f"Low-score plot saved: {output_path}")

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"

class InferenceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> bytes:
    """
    Helper function to run PDMS evaluation in.
    :param args: input arguments
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=True
    )

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    tokens_to_evaluate = sorted(tokens_to_evaluate) 
    
    save_trajectories = bool(cfg.get("save_trajectories", False))
    pdm_results: List[Dict[str, Any]] = []
    trajectory_results: Dict[str, Any] = {}
    for idx, (token) in enumerate(tokens_to_evaluate):
        if dist.get_rank() == 0:
            logger.info(f"Rank {dist.get_rank()} processing scenario {idx+1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}")

        score_row: Dict[str, Any] = {"token": token, "valid": True}
        try:
            metric_cache_path = metric_cache_loader.metric_cache_paths[token]
            with lzma.open(metric_cache_path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)

            requires_scene = False
            agent_input = scene_loader.get_agent_input_from_token(token)
            if requires_scene:
                scene = scene_loader.get_scene_from_token(token)
                trajectory = agent.compute_trajectory(agent_input, scene)
            else:
                trajectory = agent.compute_trajectory(agent_input)
            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            if save_trajectories:
                trajectory_results[token] = trajectory
            score_row.update(asdict(pdm_result))
            score_row['rank'] = dist.get_rank()

            # --- Inline low-score plotting ---
            low_score_threshold = cfg.get("low_score_plot_threshold", None)
            if low_score_threshold is not None and score_row.get("score", 1.0) < low_score_threshold:
                try:
                    _plot_low_score_inline(
                        token=token,
                        score_row=score_row,
                        cfg=cfg,
                        scene_loader=scene_loader,
                        agent_input=agent_input,
                        trajectory=trajectory,
                        scorer=scorer,
                    )
                except Exception as plot_err:
                    logger.warning(
                        f"Low-score plot failed for {token}: {type(plot_err).__name__}: {plot_err}"
                    )
        except Exception as e:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row["valid"] = False

        pdm_results.append(score_row)
    serialized_payload = pickle.dumps(
        {
            "score_rows": pdm_results,
            "trajectories": trajectory_results,
        }
    )
    return serialized_payload

def broadcast_object(obj: Any, device: torch.device, src: int = 0) -> Any:
    """
    Helper function to broadcast an object from the source rank to all other processes.
    :param obj: Object to broadcast.
    :param device: Device to use for tensor operations.
    :param src: Source rank.
    :return: Broadcasted object.
    """
    if dist.get_rank() == src:
        buffer = pickle.dumps(obj)
        tensor = torch.ByteTensor(list(buffer)).to(device)
        size_tensor = torch.tensor(len(tensor)).to(device)
        dist.broadcast(size_tensor, src=src)
        dist.broadcast(tensor, src=src)
    else:
        size_tensor = torch.tensor(0).to(device)
        dist.broadcast(size_tensor, src=src)
        tensor = torch.ByteTensor(size_tensor.item()).to(device)
        dist.broadcast(tensor, src=src)
        buffer = tensor.cpu().numpy().tobytes()
        obj = pickle.loads(buffer)
    return obj


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    local_world_size = int(os.getenv('LOCAL_WORLD_SIZE', world_size))
    rank = int(os.getenv('RANK', 0))

    visible_device_count = torch.cuda.device_count()
    default_backend = "nccl" if visible_device_count > 0 and local_world_size <= visible_device_count else "gloo"
    dist_backend = os.getenv("DIST_BACKEND", default_backend)

    dist.init_process_group(
        backend=dist_backend,
        world_size=world_size,
        rank=rank,
    )

    if torch.cuda.is_available():
        cuda_device_index = local_rank % visible_device_count
        torch.cuda.set_device(cuda_device_index)
        model_device = torch.device(f"cuda:{cuda_device_index}")
    else:
        cuda_device_index = -1
        model_device = torch.device("cpu")

    comm_device = torch.device("cpu") if dist_backend == "gloo" else model_device

    if rank == 0:
        build_logger(cfg)
    dist.barrier()


    scene_loader = SceneLoader(
            sensor_blobs_path=None,
            data_path=Path(cfg.navsim_log_path),
            scene_filter=instantiate(cfg.train_test_split.scene_filter),
            sensor_config=SensorConfig.build_no_sensors(),
        )
    if rank == 0:
        metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
        tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
        tokens_to_evaluate = sorted(tokens_to_evaluate)  
        num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
        num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
        if num_missing_metric_cache_tokens > 0:
            logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
        if num_unused_metric_cache_tokens > 0:
            logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    else:
        tokens_to_evaluate = []

    tokens_to_evaluate = broadcast_object(tokens_to_evaluate, device=comm_device, src=0)


    logger.info("Starting pdm scoring of %s scenarios...", str(len(tokens_to_evaluate)))

    sampler = InferenceSampler(len(tokens_to_evaluate))

    data_points = []
    for idx in sampler:
        token = tokens_to_evaluate[idx]
        log_file = scene_loader.token_to_log_file[token] 
        data_points.append({
            "cfg": cfg,
            "log_file": log_file,
            "tokens": [token],
        })

    if data_points:
        serialized_score_rows = run_pdm_score(data_points)
    else:
        serialized_score_rows = pickle.dumps([])


    serialized_tensor = torch.ByteTensor(list(serialized_score_rows)).to(comm_device)

    local_size = len(serialized_tensor)
    size_list = [torch.tensor(local_size).to(comm_device) for _ in range(dist.get_world_size())]
    dist.all_gather(size_list, torch.tensor(local_size).to(comm_device))

    max_size = max(size_list).item() 

    if local_size < max_size:
        padded_tensor = torch.cat([serialized_tensor, torch.zeros(max_size - local_size, dtype=torch.uint8).to(comm_device)])
    else:
        padded_tensor = serialized_tensor

    gathered_results = [torch.empty_like(padded_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_results, padded_tensor)

    if dist.get_rank() == 0:
        final_results = []
        final_trajectories = {}
        for gathered_tensor, size_tensor in zip(gathered_results, size_list):
            gathered_tensor = gathered_tensor[: size_tensor.item()]
            serialized_data = gathered_tensor.cpu().numpy().tobytes()
            payload = pickle.loads(serialized_data)
            if isinstance(payload, dict) and "score_rows" in payload:
                final_results.extend(payload["score_rows"])
                final_trajectories.update(payload.get("trajectories", {}))
            else:
                final_results.extend(payload)  # backward compatibility
    
        pdm_score_df = pd.DataFrame(final_results)

        num_sucessful_scenarios = pdm_score_df["valid"].sum()
        num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
        average_row = pdm_score_df.drop(columns=["token", "valid",'rank']).mean(skipna=True)
        average_row["token"] = "average"
        average_row["valid"] = pdm_score_df["valid"].all()
        average_row["rank"] = "0"
        pdm_score_df.loc[len(pdm_score_df)] = average_row

        save_path = Path(cfg.output_dir)
        timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        pdm_score_df.to_csv(save_path / f"{timestamp}.csv")

        if bool(cfg.get("save_trajectories", False)):
            output_filename = cfg.get("trajectory_output_filename", "trajectories.pkl")
            trajectory_path = save_path / output_filename
            ordered_trajectories = {
                token: final_trajectories[token]
                for token in tokens_to_evaluate
                if token in final_trajectories
            }
            submission = {
                "team_name": cfg.get("team_name", "recogdrive"),
                "authors": cfg.get("authors", "recogdrive"),
                "email": cfg.get("email", ""),
                "institution": cfg.get("institution", ""),
                "country / region": cfg.get("country", ""),
                "predictions": [ordered_trajectories],
            }
            with trajectory_path.open("wb") as f:
                pickle.dump(submission, f)
            logger.info(f"Saved {len(ordered_trajectories)} trajectories to {trajectory_path}")
            print(f"Saved {len(ordered_trajectories)} trajectories to {trajectory_path}", flush=True)

        logger.info(
            f"""
            Finished running evaluation.
                Number of successful scenarios: {num_sucessful_scenarios}.
                Number of failed scenarios: {num_failed_scenarios}.
                Final average score of valid results: {pdm_score_df['score'].mean()}.
                Results are stored in: {save_path / f"{timestamp}.csv"}.
            """
        )


if __name__ == "__main__":
    main()
