from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import torch

from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)
from navsim.planning.training.dataset import CacheOnlyDataset

logger = logging.getLogger(__name__)

IndexRequest = Union[int, Tuple[int, Optional[str]]]


@dataclass(frozen=True)
class RoleDatasetSpec:
    """Cache dataset inputs for one training role."""

    name: str
    cache_path: str
    tokens_path: Optional[str] = None
    log_names: Optional[List[str]] = None
    teacher_cache_path: Optional[str] = None
    target_override_path: Optional[str] = None
    manifest_path: Optional[str] = None


def _uid_token(uid: str) -> str:
    return uid.split(":", 1)[1] if ":" in uid else uid


def _role_matches_manifest_group(role: str, group: str) -> bool:
    if role == group:
        return True
    return role == "failure" and group == "real_correction"


def load_manifest_metadata(path: Optional[str], role: str) -> Dict[str, Dict[str, Any]]:
    cleaned = str(path or "").strip().strip("\'\"")
    if cleaned.lower() in {"", "none", "null"}:
        return {}
    manifest_path = Path(cleaned)
    rows = json.loads(manifest_path.read_text())
    if isinstance(rows, dict):
        rows = rows.get("records") or rows.get("items") or rows.get("data") or []
    if not isinstance(rows, list):
        raise ValueError(f"Manifest metadata must be a list of rows: {manifest_path}")

    metadata: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        group = str(row.get("continual_group") or "")
        if group and not _role_matches_manifest_group(role, group):
            continue
        uid = str(row.get("uid") or "")
        token = str(row.get("token") or _uid_token(uid))
        if not token:
            continue
        entry = {
            "uid": uid or token,
            "token": token,
            "scene_id": str(row.get("scene_id") or token),
            "split": str(row.get("split") or ""),
            "log_name": str(row.get("log_name") or row.get("source_log") or ""),
            "source_log": str(row.get("source_log") or row.get("log_name") or ""),
            "start_idx": int(row.get("start_idx") or 0),
            "teacher_policy": str(row.get("teacher_policy") or ""),
            "teacher_source": str(row.get("teacher_source") or ""),
        }
        metadata[token] = entry
        metadata[entry["uid"]] = entry
        if entry["uid"]:
            metadata[_uid_token(entry["uid"])] = entry
    logger.info("Loaded %d manifest metadata entries for role=%s from %s", len(metadata), role, manifest_path)
    return metadata


def load_tokens(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    cleaned = str(path).strip().strip("\'\"")
    if cleaned.lower() in {"", "none", "null"}:
        return None
    token_path = Path(cleaned)
    tokens = [line.strip() for line in token_path.read_text().splitlines()]
    return [token for token in tokens if token and not token.startswith("#")]


class RoleMixedCacheOnlyDataset(torch.utils.data.Dataset):
    """Concatenates cache-only NAVSIM datasets while retaining metadata per sample."""

    def __init__(
        self,
        specs: Sequence[RoleDatasetSpec],
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
    ) -> None:
        super().__init__()
        self.datasets: List[CacheOnlyDataset] = []
        self.roles: List[str] = []
        self.index: List[tuple[int, int]] = []
        self.metadata: List[Dict[str, Any]] = []
        self.role_to_indices: Dict[str, List[int]] = {}

        for spec in specs:
            tokens = load_tokens(spec.tokens_path)
            manifest_metadata = load_manifest_metadata(spec.manifest_path, spec.name)
            dataset = CacheOnlyDataset(
                cache_path=spec.cache_path,
                feature_builders=feature_builders,
                target_builders=target_builders,
                log_names=spec.log_names,
                tokens=tokens,
            )
            if len(dataset) == 0:
                logger.warning("Role %s has no valid samples and will be skipped.", spec.name)
                continue

            dataset_id = len(self.datasets)
            self.datasets.append(dataset)
            self.roles.append(spec.name)
            self.role_to_indices.setdefault(spec.name, [])
            for local_idx, token in enumerate(dataset.tokens):
                global_idx = len(self.index)
                token_path = dataset._valid_cache_paths[token]
                self.index.append((dataset_id, local_idx))
                row_metadata = manifest_metadata.get(str(token), {})
                log_name = str(row_metadata.get("log_name") or token_path.parent.name)
                source_log = str(row_metadata.get("source_log") or log_name)
                self.metadata.append({
                    "uid": str(row_metadata.get("uid") or token),
                    "token": str(token),
                    "scene_id": str(row_metadata.get("scene_id") or token),
                    "split": str(row_metadata.get("split") or ""),
                    "role": spec.name,
                    "cache_path": str(token_path),
                    "log_name": log_name,
                    "source_log": source_log,
                    "start_idx": int(row_metadata.get("start_idx", local_idx)),
                    "teacher_policy": str(row_metadata.get("teacher_policy") or ""),
                    "teacher_source": str(row_metadata.get("teacher_source") or ""),
                    "teacher_cache_path": spec.teacher_cache_path,
                    "target_override_path": spec.target_override_path,
                    "replay_stream": None,
                })
                self.role_to_indices[spec.name].append(global_idx)

            logger.info(
                "Loaded role=%s samples=%d cache=%s tokens=%s",
                spec.name,
                len(dataset),
                spec.cache_path,
                spec.tokens_path,
            )

        if not self.index:
            raise ValueError("No valid samples were found in role-mixed training inputs.")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: IndexRequest):
        replay_stream: Optional[str] = None
        if isinstance(idx, tuple):
            idx, replay_stream = idx
        dataset_id, local_idx = self.index[int(idx)]
        features, targets, token = self.datasets[dataset_id][local_idx]
        meta = dict(self.metadata[int(idx)])
        meta["token"] = str(token)
        meta["replay_stream"] = replay_stream or meta.get("replay_stream")
        return features, targets, meta

    def replay_items(self) -> List[Dict[str, object]]:
        return [
            {
                "index": idx,
                "token": str(self.metadata[idx]["token"]),
                "log_name": str(self.metadata[idx].get("log_name") or ""),
                "source_log": str(self.metadata[idx].get("source_log") or self.metadata[idx].get("log_name") or ""),
                "start_idx": int(self.metadata[idx].get("start_idx") or 0),
            }
            for idx in self.role_to_indices.get("replay", [])
        ]


class RoleBalancedBatchSampler(torch.utils.data.Sampler[List[int]]):
    """Yields batches with configurable per-role counts."""

    def __init__(
        self,
        role_to_indices: Dict[str, Sequence[int]],
        role_counts: Dict[str, int],
        batch_size: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        batches_per_epoch: Optional[int] = None,
    ) -> None:
        if sum(role_counts.values()) != batch_size:
            raise ValueError(
                f"role_counts must sum to batch_size ({batch_size}), got {role_counts}"
            )

        missing = [role for role in role_counts if role not in role_to_indices]
        if missing:
            raise ValueError(f"Configured roles are missing from dataset: {missing}")

        empty = [role for role, indices in role_to_indices.items() if role in role_counts and not indices]
        if empty:
            raise ValueError(f"Configured roles have no samples: {empty}")

        self.role_to_indices = {role: list(indices) for role, indices in role_to_indices.items()}
        self.role_counts = dict(role_counts)
        self.batch_size = batch_size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.batches_per_epoch = batches_per_epoch or self._default_batches_per_epoch()
        self.epoch = 0

    def _default_batches_per_epoch(self) -> int:
        largest_role = max(
            len(self.role_to_indices[role]) / max(count, 1)
            for role, count in self.role_counts.items()
            if count > 0
        )
        return max(1, int(largest_role))

    def __len__(self) -> int:
        return self.batches_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _role_stream(self, role: str) -> Iterator[int]:
        base_indices = self.role_to_indices[role]
        cycle = 0
        while True:
            indices = list(base_indices)
            role_hash = int(hashlib.sha256(role.encode("utf-8")).hexdigest()[:8], 16)
            rng = random.Random(self.seed + self.epoch * 1000003 + cycle * 9176 + role_hash % 7919)
            rng.shuffle(indices)
            for idx in indices:
                yield idx
            cycle += 1

    def __iter__(self) -> Iterable[List[int]]:
        streams = {role: self._role_stream(role) for role in self.role_counts}
        rng = random.Random(self.seed + self.epoch * 104729 + self.rank)

        for global_batch_idx in range(self.batches_per_epoch * self.world_size):
            batch: List[int] = []
            for role, count in self.role_counts.items():
                batch.extend(next(streams[role]) for _ in range(count))
            rng.shuffle(batch)
            if global_batch_idx % self.world_size == self.rank:
                yield batch
