from __future__ import annotations

import hashlib
import random
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import torch

from navsim.planning.training.adaptive_replay.controller import AdaptiveReplayController

IndexRequest = Union[int, Tuple[int, Optional[str]]]


def _stable_role_hash(role: str) -> int:
    return int(hashlib.sha256(role.encode("utf-8")).hexdigest()[:8], 16)


class AdaptiveRoleBatchSampler(torch.utils.data.Sampler[List[IndexRequest]]):
    """Role sampler with replay split into priority and uniform streams."""

    def __init__(
        self,
        role_to_indices: Dict[str, Sequence[int]],
        role_counts: Dict[str, int],
        batch_size: int,
        seed: int,
        controller: AdaptiveReplayController,
        rank: int = 0,
        world_size: int = 1,
        batches_per_epoch: Optional[int] = None,
    ) -> None:
        expected = set(role_counts) - {"replay_priority", "replay_uniform", "failure", "synthetic"}
        if expected:
            raise ValueError(f"Unknown adaptive replay count roles: {sorted(expected)}")
        if sum(role_counts.values()) != batch_size:
            raise ValueError(f"role_counts must sum to batch_size ({batch_size}), got {role_counts}")
        if (role_counts.get("replay_priority", 0) or role_counts.get("replay_uniform", 0)) and "replay" not in role_to_indices:
            raise ValueError("Adaptive replay counts require a replay dataset role.")
        for role in ("failure", "synthetic"):
            if role_counts.get(role, 0) and role not in role_to_indices:
                raise ValueError(f"Configured role {role} is missing from the dataset.")

        self.role_to_indices = {role: list(indices) for role, indices in role_to_indices.items()}
        self.role_counts = {role: int(count) for role, count in role_counts.items() if int(count) > 0}
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.controller = controller
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.batches_per_epoch = batches_per_epoch or self._default_batches_per_epoch()
        if self.controller.require_full_pass and self.role_counts.get("replay_uniform", 0):
            if self.batches_per_epoch < self.controller.refresh_interval:
                raise ValueError(
                    "Adaptive replay batches_per_epoch must cover at least one full uniform pass "
                    f"({self.controller.refresh_interval} steps)."
                )
            if self.batches_per_epoch % self.controller.refresh_interval != 0:
                raise ValueError(
                    "Adaptive replay batches_per_epoch must be a multiple of refresh_interval "
                    "when require_full_pass=True."
                )
        self.epoch = 0

    def _default_batches_per_epoch(self) -> int:
        candidates: List[float] = []
        replay_count = self.role_counts.get("replay_uniform", 0) + self.role_counts.get("replay_priority", 0)
        if replay_count:
            candidates.append(len(self.role_to_indices["replay"]) / replay_count)
        for role in ("failure", "synthetic"):
            if self.role_counts.get(role, 0):
                candidates.append(len(self.role_to_indices[role]) / self.role_counts[role])
        if self.controller.require_full_pass and self.role_counts.get("replay_uniform", 0):
            candidates.append(self.controller.refresh_interval)
        return max(1, int(max(candidates) if candidates else 1))

    def __len__(self) -> int:
        return self.batches_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _static_role_stream(self, role: str) -> Iterator[IndexRequest]:
        base_indices = self.role_to_indices[role]
        cycle = 0
        while True:
            indices = list(base_indices)
            rng = random.Random(self.seed + self.epoch * 1000003 + cycle * 9176 + _stable_role_hash(role))
            rng.shuffle(indices)
            for idx in indices:
                yield idx
            cycle += 1

    def _adaptive_replay_stream(self, stream_name: str) -> Iterator[IndexRequest]:
        cycle = 0
        position = 0
        items_since_reset = 0
        generation = -1
        has_streamed = False
        ordered: List[int] = []
        while True:
            current_generation = self.controller.generation
            if current_generation != generation or not ordered or items_since_reset >= len(ordered):
                generation = current_generation
                if stream_name == "priority":
                    ordered = list(self.controller.priority_indices)
                else:
                    ordered = list(self.controller.replay_indices)
                position = 0
                items_since_reset = 0
                if has_streamed and self.world_size > 1 and ordered:
                    # The outer iterator consumes non-local global batches before
                    # yielding the next local batch.  If a replay generation
                    # changes mid-epoch, each rank resets from just after its own
                    # previous yield; offset the new stream so yielded batches
                    # remain globally sharded instead of all ranks replaying the
                    # same first window.  ``position`` is allowed to start
                    # negative and is applied modulo the order below so this
                    # pre-yield offset can cross the end of the stream without
                    # triggering an early reshuffle.
                    count = self.role_counts.get(f"replay_{stream_name}", 0)
                    position = (self.rank - (self.world_size - 1)) * count
                rng = random.Random(
                    self.seed
                    + self.epoch * 1000003
                    + generation * 65537
                    + cycle * 9176
                    + _stable_role_hash(f"replay_{stream_name}")
                )
                rng.shuffle(ordered)
                cycle += 1
            idx = ordered[position % len(ordered)]
            position += 1
            items_since_reset += 1
            has_streamed = True
            yield (idx, stream_name)

    def __iter__(self) -> Iterable[List[IndexRequest]]:
        streams = {
            "replay_priority": self._adaptive_replay_stream("priority"),
            "replay_uniform": self._adaptive_replay_stream("uniform"),
            "failure": self._static_role_stream("failure") if "failure" in self.role_to_indices else None,
            "synthetic": self._static_role_stream("synthetic") if "synthetic" in self.role_to_indices else None,
        }
        rng = random.Random(self.seed + self.epoch * 104729 + self.rank)
        for global_batch_idx in range(self.batches_per_epoch * self.world_size):
            batch: List[IndexRequest] = []
            for role, count in self.role_counts.items():
                stream = streams[role]
                if stream is None:
                    raise ValueError(f"No stream available for role {role}")
                batch.extend(next(stream) for _ in range(count))
            rng.shuffle(batch)
            if global_batch_idx % self.world_size == self.rank:
                yield batch
