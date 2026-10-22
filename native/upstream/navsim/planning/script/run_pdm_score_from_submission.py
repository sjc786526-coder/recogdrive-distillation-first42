from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path
from dataclasses import asdict
import logging
import traceback
import pickle

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from tqdm import tqdm

from nuplan.planning.script.builders.logging_builder import build_logger

from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.common.dataclasses import Trajectory

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_from_submission"

_WORKER_AGENT_OUTPUT: Optional[Dict[str, Trajectory]] = None
_WORKER_METRIC_CACHE_LOADER: Optional[MetricCacheLoader] = None
_WORKER_SIMULATOR: Optional[PDMSimulator] = None
_WORKER_SCORER: Optional[PDMScorer] = None


def _load_submission(submission_file_path: Path) -> Dict[str, Trajectory]:
    with open(submission_file_path, "rb") as f:
        agent_output_list: List[Dict[str, Trajectory]] = pickle.load(f)["predictions"]

    assert len(agent_output_list) == 1, "Multi-seed evaluation currently not supported in run_pdm_score!"
    return agent_output_list[0]


def _score_token(
    token: str,
    agent_output: Dict[str, Trajectory],
    metric_cache_loader: MetricCacheLoader,
    simulator: PDMSimulator,
    scorer: PDMScorer,
) -> Dict[str, Any]:
    score_row: Dict[str, Any] = {"token": token, "valid": True}
    try:
        metric_cache = metric_cache_loader.get_from_token(token)
        trajectory = agent_output[token]
        pdm_result = pdm_score(
            metric_cache=metric_cache,
            model_trajectory=trajectory,
            future_sampling=simulator.proposal_sampling,
            simulator=simulator,
            scorer=scorer,
        )
        score_row.update(asdict(pdm_result))
    except Exception:
        logger.warning(f"----------- Agent failed for token {token}:")
        traceback.print_exc()
        score_row["valid"] = False
    return score_row


def _init_score_worker(
    submission_file_path: str,
    metric_cache_path: str,
    simulator_cfg: Dict[str, Any],
    scorer_cfg: Dict[str, Any],
) -> None:
    global _WORKER_AGENT_OUTPUT
    global _WORKER_METRIC_CACHE_LOADER
    global _WORKER_SIMULATOR
    global _WORKER_SCORER

    _WORKER_AGENT_OUTPUT = _load_submission(Path(submission_file_path))
    _WORKER_METRIC_CACHE_LOADER = MetricCacheLoader(Path(metric_cache_path))
    _WORKER_SIMULATOR = instantiate(OmegaConf.create(simulator_cfg))
    _WORKER_SCORER = instantiate(OmegaConf.create(scorer_cfg))
    assert (
        _WORKER_SIMULATOR.proposal_sampling == _WORKER_SCORER.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"


def _score_token_in_worker(token: str) -> Dict[str, Any]:
    assert _WORKER_AGENT_OUTPUT is not None
    assert _WORKER_METRIC_CACHE_LOADER is not None
    assert _WORKER_SIMULATOR is not None
    assert _WORKER_SCORER is not None
    return _score_token(
        token=token,
        agent_output=_WORKER_AGENT_OUTPUT,
        metric_cache_loader=_WORKER_METRIC_CACHE_LOADER,
        simulator=_WORKER_SIMULATOR,
        scorer=_WORKER_SCORER,
    )


def run_pdm_score(
    submission_file_path: Path,
    simulator: PDMSimulator,
    scorer: PDMScorer,
    metric_cache_path: Path,
) -> pd.DataFrame:
    """
    Evaluate submission file with PDM score.
    :param submission_file_path: path to submission pickle file
    :param simulator: internal simulator object of PDM
    :param scorer: internal scoring objected in PDM
    :param metric_cache_path: path to metric cache
    :return: pandas data frame with PDMS results.
    """
    logger.info("Building SceneLoader")
    metric_cache_loader = MetricCacheLoader(metric_cache_path)
    agent_output = _load_submission(submission_file_path)

    score_rows: List[Dict[str, Any]] = []
    for token in tqdm(metric_cache_loader.tokens, desc="Compute PDM-Score"):
        score_rows.append(
            _score_token(
                token=token,
                agent_output=agent_output,
                metric_cache_loader=metric_cache_loader,
                simulator=simulator,
                scorer=scorer,
            )
        )

    pdm_score_df = pd.DataFrame(score_rows)
    if not pdm_score_df["valid"].all():
        logger.warning("Evaluation for some tokens failed. Check log for details")
    return pdm_score_df


def run_pdm_score_parallel(
    submission_file_path: Path,
    simulator_cfg: DictConfig,
    scorer_cfg: DictConfig,
    metric_cache_path: Path,
    num_workers: int,
    worker_chunk_size: int,
) -> pd.DataFrame:
    """Evaluate submission file with PDM score using multiple CPU processes."""
    metric_cache_loader = MetricCacheLoader(metric_cache_path)
    tokens = list(metric_cache_loader.tokens)
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_score_worker,
        initargs=(
            str(submission_file_path),
            str(metric_cache_path),
            OmegaConf.to_container(simulator_cfg, resolve=True),
            OmegaConf.to_container(scorer_cfg, resolve=True),
        ),
    ) as executor:
        score_rows = list(
            tqdm(
                executor.map(_score_token_in_worker, tokens, chunksize=worker_chunk_size),
                total=len(tokens),
                desc=f"Compute PDM-Score ({num_workers} workers)",
            )
        )

    pdm_score_df = pd.DataFrame(score_rows)
    if not pdm_score_df["valid"].all():
        logger.warning("Evaluation for some tokens failed. Check log for details")
    return pdm_score_df


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS from submission pickle.
    :param cfg: omegaconf dictionary
    """
    submission_file_path = Path(cfg.submission_file_path)
    metric_cache_path = Path(cfg.metric_cache_path)
    build_logger(cfg)
    num_workers = max(1, int(cfg.get("num_workers", 1)))
    worker_chunk_size = max(1, int(cfg.get("worker_chunk_size", 16)))
    if num_workers > 1:
        pdm_score_df = run_pdm_score_parallel(
            submission_file_path=submission_file_path,
            simulator_cfg=cfg.simulator,
            scorer_cfg=cfg.scorer,
            metric_cache_path=metric_cache_path,
            num_workers=num_workers,
            worker_chunk_size=worker_chunk_size,
        )
    else:
        simulator: PDMSimulator = instantiate(cfg.simulator)
        scorer: PDMScorer = instantiate(cfg.scorer)
        assert (
            simulator.proposal_sampling == scorer.proposal_sampling
        ), "Simulator and scorer proposal sampling has to be identical"
        pdm_score_df = run_pdm_score(
            submission_file_path=submission_file_path,
            simulator=simulator,
            scorer=scorer,
            metric_cache_path=metric_cache_path,
        )

    average_row = pdm_score_df.drop(columns=["token", "valid"]).mean(skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = pdm_score_df["valid"].all()
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    save_path = Path(cfg.output_dir)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(save_path / f"{timestamp}.csv")


if __name__ == "__main__":
    main()
