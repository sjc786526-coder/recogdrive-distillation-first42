from typing import Tuple
from pathlib import Path
import logging
import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.distributed as dist
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.role_mixed_dataset import (
    RoleBalancedBatchSampler,
    RoleDatasetSpec,
    RoleMixedCacheOnlyDataset,
)
from navsim.planning.training.adaptive_replay import AdaptiveReplayController, AdaptiveRoleBatchSampler
import torch
import torch.nn.utils.rnn as rnn_utils
from typing import List, Dict

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def _normalize_dataloader_params(params):
    dataloader_params = dict(params)
    num_workers = int(dataloader_params.get("num_workers", 0) or 0)
    dataloader_params["num_workers"] = num_workers

    pin_memory = dataloader_params.get("pin_memory")
    if isinstance(pin_memory, str):
        dataloader_params["pin_memory"] = pin_memory.strip().lower() in {"1", "true", "yes", "y"}

    if num_workers <= 0:
        dataloader_params.pop("prefetch_factor", None)
        dataloader_params.pop("persistent_workers", None)
    else:
        prefetch_factor = dataloader_params.get("prefetch_factor")
        if prefetch_factor is not None:
            dataloader_params["prefetch_factor"] = int(prefetch_factor)

    return dataloader_params


def _disable_dataloader_prefetch(params, reason: str):
    dataloader_params = dict(params)
    if int(dataloader_params.get("num_workers", 0) or 0) > 0:
        logger.warning("%s; forcing dataloader num_workers=0 and disabling prefetch.", reason)
    dataloader_params["num_workers"] = 0
    dataloader_params.pop("prefetch_factor", None)
    dataloader_params.pop("persistent_workers", None)
    return dataloader_params



def _clean_optional_path(value):
    if value is None:
        return None
    cleaned = str(value).strip().strip("\'\"")
    if cleaned.lower() in {"", "none", "null"}:
        return None
    return cleaned

def _load_token_list(path: str | None) -> List[str] | None:
    cleaned = _clean_optional_path(path)
    if not cleaned:
        return None
    token_path = Path(cleaned)
    tokens = [line.strip() for line in token_path.read_text().splitlines()]
    tokens = [token for token in tokens if token and not token.startswith("#")]
    logger.info("Loaded %d tokens from %s", len(tokens), token_path)
    return tokens




def custom_collate_fn(
    batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    features_list, targets_list, sample_meta = zip(*batch)

    history_trajectory = torch.stack([features['history_trajectory'] for features in features_list], dim=0).cpu()
    high_command_one_hot = torch.stack([features['high_command_one_hot'] for features in features_list], dim=0).cpu()
    status_feature = torch.stack([features['status_feature'] for features in features_list], dim=0).cpu()

    trajectory = torch.stack([targets['trajectory'] for targets in targets_list], dim=0).cpu()

    features = {
        'history_trajectory': history_trajectory,
        'high_command_one_hot': high_command_one_hot,
        'status_feature': status_feature,
    }

    first_features = features_list[0]
    if 'last_hidden_state' in first_features:
        last_hidden_state_lengths = torch.tensor(
            [features['last_hidden_state'].shape[0] for features in features_list],
            dtype=torch.long,
        )
        last_hidden_state = rnn_utils.pad_sequence(
            [features['last_hidden_state'] for features in features_list],
            batch_first=True,
            padding_value=0.0,
        ).clone().detach()
        features['last_hidden_state'] = last_hidden_state
        features['last_hidden_state_lengths'] = last_hidden_state_lengths
    elif 'image_path_tensor' in first_features:
        image_path_tensor = rnn_utils.pad_sequence(
            [features['image_path_tensor'].long() for features in features_list],
            batch_first=True,
            padding_value=0,
        ).clone().detach()
        features['image_path_tensor'] = image_path_tensor
    elif 'pixel_values' in first_features:
        pixel_values_list = []
        num_patches_list = []
        for feature_item in features_list:
            pixel_values = feature_item['pixel_values']
            if pixel_values.ndim == 3:
                pixel_values = pixel_values.unsqueeze(0)
            elif pixel_values.ndim == 5 and pixel_values.shape[0] == 1:
                pixel_values = pixel_values.squeeze(0)
            pixel_values_list.append(pixel_values)
            num_patches_list.append(int(pixel_values.shape[0]))
        features['pixel_values'] = torch.cat(pixel_values_list, dim=0).clone().detach()
        features['num_patches_list'] = torch.tensor(num_patches_list, dtype=torch.long)
    else:
        available = ', '.join(sorted(first_features.keys()))
        raise KeyError(f"Unsupported ReCogDrive feature cache keys: {available}")

    targets = {
        'trajectory': trajectory
    }

    if sample_meta and isinstance(sample_meta[0], dict):
        metadata = {
            'uids': [str(item.get('uid') or item.get('token', '')) for item in sample_meta],
            'tokens': [str(item.get('token', '')) for item in sample_meta],
            'scene_ids': [str(item.get('scene_id') or item.get('token', '')) for item in sample_meta],
            'roles': [str(item.get('role', '')) for item in sample_meta],
            'cache_paths': [item.get('cache_path') for item in sample_meta],
            'log_names': [str(item.get('log_name') or '') for item in sample_meta],
            'source_logs': [str(item.get('source_log') or item.get('log_name') or '') for item in sample_meta],
            'start_indices': [int(item.get('start_idx') or 0) for item in sample_meta],
            'teacher_cache_paths': [item.get('teacher_cache_path') for item in sample_meta],
            'target_override_paths': [item.get('target_override_path') for item in sample_meta],
            'teacher_policies': [str(item.get('teacher_policy') or '') for item in sample_meta],
            'teacher_sources': [str(item.get('teacher_source') or '') for item in sample_meta],
            'replay_streams': [item.get('replay_stream') for item in sample_meta],
        }
    else:
        metadata = sample_meta

    return features, targets, metadata

def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    train_tokens = _load_token_list(cfg.get("train_tokens_path"))
    val_tokens = _load_token_list(cfg.get("val_tokens_path")) or train_tokens
    if train_tokens is not None:
        train_scene_filter.tokens = train_tokens
    if train_tokens is None:
        if train_scene_filter.log_names is not None:
            train_scene_filter.log_names = [
                log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
            ]
        else:
            train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_tokens is not None:
        val_scene_filter.tokens = val_tokens
    if val_tokens is None:
        if val_scene_filter.log_names is not None:
            val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
        else:
            val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    feature_builders = agent.get_feature_builders()
    target_builders = agent.get_target_builders()

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=feature_builders,
        target_builders=target_builders,
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=feature_builders,
        target_builders=target_builders,
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


def _build_role_specs(role_cfg) -> List[RoleDatasetSpec]:
    specs: List[RoleDatasetSpec] = []
    for item in role_cfg:
        log_names = item.get("log_names")
        specs.append(
            RoleDatasetSpec(
                name=str(item["name"]),
                cache_path=str(_clean_optional_path(item["cache_path"])),
                tokens_path=_clean_optional_path(item.get("tokens_path")),
                log_names=list(log_names) if log_names is not None else None,
                teacher_cache_path=_clean_optional_path(item.get("teacher_cache_path")),
                target_override_path=_clean_optional_path(item.get("target_override_path")),
                manifest_path=_clean_optional_path(item.get("manifest_path")),
            )
        )
    return specs


def _build_role_counts(counts_cfg) -> Dict[str, int]:
    return {str(role): int(count) for role, count in counts_cfg.items() if int(count) > 0}


def _optional_int(value):
    if value is None or str(value).lower() in {"", "none", "null"}:
        return None
    return int(value)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    rank = int(os.getenv('RANK', 0))

    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(local_rank)
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")
    train_tokens = _load_token_list(cfg.get("train_tokens_path"))
    val_tokens = _load_token_list(cfg.get("val_tokens_path")) or train_tokens

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    role_sampling_cfg = cfg.get("role_sampling")
    use_role_sampling = bool(role_sampling_cfg and role_sampling_cfg.get("enabled", False))
    gd_ema_cfg = cfg.get("gd_ema")
    gd_ema_enabled = bool(gd_ema_cfg and gd_ema_cfg.get("enabled", False))
    if gd_ema_enabled and not use_role_sampling:
        raise ValueError("gd_ema.enabled requires role_sampling.enabled=true")
    if gd_ema_enabled:
        lightning_module.configure_gd_ema(gd_ema_cfg)

    if use_role_sampling:
        logger.info("Using role-mixed cached data")
        assert (
            cfg.use_cache_without_dataset
        ), "role_sampling currently requires use_cache_without_dataset=True"
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False with role_sampling"

        role_specs = _build_role_specs(role_sampling_cfg.roles)
        train_data = RoleMixedCacheOnlyDataset(
            specs=role_specs,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
        )

        val_cache_path = _clean_optional_path(role_sampling_cfg.get("val_cache_path")) or role_specs[0].cache_path
        val_tokens_path = _clean_optional_path(role_sampling_cfg.get("val_tokens_path")) or _clean_optional_path(cfg.get("val_tokens_path")) or role_specs[0].tokens_path
        val_data = CacheOnlyDataset(
            cache_path=val_cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=None,
            tokens=_load_token_list(val_tokens_path),
        )

        batch_size = int(cfg.dataloader.params.batch_size)
        adaptive_replay_cfg = cfg.get("adaptive_replay")
        adaptive_enabled = bool(adaptive_replay_cfg and adaptive_replay_cfg.get("enabled", False))
        if adaptive_enabled:
            logger.info("Using adaptive replay half-priority / half-uniform sampler")
            replay_items = train_data.replay_items()
            initial_priority_tokens = _load_token_list(adaptive_replay_cfg.get("initial_priority_tokens_path"))
            max_priority_per_log = adaptive_replay_cfg.get("max_priority_per_log")
            role_counts = _build_role_counts(adaptive_replay_cfg.counts)
            require_full_pass = bool(adaptive_replay_cfg.get("require_full_pass", True))
            configured_refresh_interval = int(adaptive_replay_cfg.get("refresh_interval", 1024))
            refresh_interval = configured_refresh_interval
            replay_uniform_count = int(role_counts.get("replay_uniform", 0))
            if require_full_pass:
                if replay_uniform_count <= 0:
                    raise ValueError("Adaptive replay require_full_pass requires replay_uniform > 0.")
                observations_per_step = replay_uniform_count * world_size
                if len(replay_items) % observations_per_step != 0:
                    raise ValueError(
                        "Adaptive replay full uniform pass must align with optimizer-step boundaries: "
                        f"pool={len(replay_items)} observations_per_step={observations_per_step}."
                    )
                full_pass_interval = len(replay_items) // observations_per_step
                if configured_refresh_interval != full_pass_interval:
                    logger.warning(
                        "Adaptive replay refresh_interval=%s does not match full uniform pass interval=%s "
                        "for pool=%s world_size=%s replay_uniform=%s; using %s.",
                        configured_refresh_interval,
                        full_pass_interval,
                        len(replay_items),
                        world_size,
                        replay_uniform_count,
                        full_pass_interval,
                    )
                refresh_interval = full_pass_interval
            controller = AdaptiveReplayController(
                replay_items=replay_items,
                priority_size=int(adaptive_replay_cfg.get("priority_size", max(1, len(replay_items) // 4))),
                seed=int(cfg.seed),
                refresh_interval=refresh_interval,
                rank=rank,
                world_size=world_size,
                initial_priority_tokens=initial_priority_tokens,
                require_full_pass=require_full_pass,
                max_priority_per_log=_optional_int(max_priority_per_log),
            )
            batch_sampler = AdaptiveRoleBatchSampler(
                role_to_indices=train_data.role_to_indices,
                role_counts=role_counts,
                batch_size=batch_size,
                seed=int(cfg.seed),
                controller=controller,
                rank=rank,
                world_size=world_size,
                batches_per_epoch=_optional_int(role_sampling_cfg.get("batches_per_epoch")),
            )
            lightning_module.configure_adaptive_replay(
                controller=controller,
                loss_cfg=adaptive_replay_cfg.get("loss", {}),
            )
        else:
            role_counts = _build_role_counts(role_sampling_cfg.counts)
            batch_sampler = RoleBalancedBatchSampler(
                role_to_indices=train_data.role_to_indices,
                role_counts=role_counts,
                batch_size=batch_size,
                seed=int(cfg.seed),
                rank=rank,
                world_size=world_size,
                batches_per_epoch=_optional_int(role_sampling_cfg.get("batches_per_epoch")),
            )
        dataloader_params = _normalize_dataloader_params(cfg.dataloader.params)
        dataloader_params.pop("batch_size", None)
        if adaptive_enabled:
            dataloader_params = _disable_dataloader_prefetch(
                dataloader_params,
                "Adaptive replay uses a step-synchronized dynamic sampler",
            )
        train_dataloader = DataLoader(
            train_data,
            collate_fn=custom_collate_fn,
            batch_sampler=batch_sampler,
            **dataloader_params,
        )
    elif cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=None if train_tokens is not None else cfg.train_logs,
            tokens=train_tokens,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=None if val_tokens is not None else cfg.val_logs,
            tokens=val_tokens,
        )
        train_dataloader = DataLoader(train_data, collate_fn=custom_collate_fn,  **_normalize_dataloader_params(cfg.dataloader.params), shuffle=True)
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)
        train_dataloader = DataLoader(train_data, collate_fn=custom_collate_fn,  **_normalize_dataloader_params(cfg.dataloader.params), shuffle=True)

    logger.info("Building Datasets")
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, collate_fn=custom_collate_fn, **_normalize_dataloader_params(cfg.dataloader.params), shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    checkpoint_save_top_k = cfg.get("checkpoint_save_top_k", 5)
    ema_cfg = cfg.get("ema")
    ema_enabled = bool(ema_cfg and ema_cfg.get("enabled", False))
    callbacks = []
    if ema_enabled:
        from navsim.planning.script.callback import EMA, EMAModelCheckpoint
        callbacks.append(EMAModelCheckpoint(
            monitor="val/loss_epoch",
            mode='min',
            save_top_k=checkpoint_save_top_k,
            every_n_epochs=int(ema_cfg.get("every_n_epochs", 1)),
            save_last=True,
        ))
        callbacks.append(EMA(
            decay=float(ema_cfg.get("decay", 0.999)),
            apply_ema_every_n_steps=int(ema_cfg.get("apply_every_n_steps", 1)),
            start_step=int(ema_cfg.get("start_step", 0)),
            save_ema_weights_in_callback_state=bool(ema_cfg.get("save_in_callback_state", False)),
            evaluate_ema_weights_instead=bool(ema_cfg.get("evaluate_ema_weights_instead", True)),
        ))
    else:
        callbacks.append(pl.callbacks.ModelCheckpoint(
            monitor="val/loss_epoch",
            mode='min',
            save_top_k=checkpoint_save_top_k,
            every_n_epochs=1,
            save_last=True,
        ))
    trainer = pl.Trainer(
        **cfg.trainer.params,
        callbacks=callbacks,
    )

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    main()
