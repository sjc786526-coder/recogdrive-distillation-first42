from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from navsim.planning.training.adaptive_replay.reference_bridge import load_diverse_priority

logger = logging.getLogger(__name__)


def stable_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReplayItem:
    index: int
    token: str
    log_name: str = ""
    start_idx: int = 0


class AdaptiveReplayController:
    """Tracks adaptive replay priority state and uniform replay observations."""

    def __init__(
        self,
        replay_items: Sequence[Dict[str, Any]],
        priority_size: int,
        seed: int,
        refresh_interval: int,
        rank: int = 0,
        world_size: int = 1,
        initial_priority_tokens: Optional[Sequence[str]] = None,
        require_full_pass: bool = False,
        max_priority_per_log: Optional[int] = None,
    ) -> None:
        if not replay_items:
            raise ValueError("Adaptive replay requires at least one replay sample.")
        if refresh_interval <= 0:
            raise ValueError("refresh_interval must be positive.")

        self.replay_items = [
            ReplayItem(
                index=int(item["index"]),
                token=str(item["token"]),
                log_name=str(item.get("source_log") or item.get("log_name", "")),
                start_idx=int(item.get("start_idx", position)),
            )
            for position, item in enumerate(replay_items)
        ]
        self.reference_diverse_priority = load_diverse_priority()
        self.reference_rows = [
            {
                "uid": item.token,
                "source_log": item.log_name or "unknown",
                "start_idx": item.start_idx,
            }
            for item in self.replay_items
        ]
        tokens = [item.token for item in self.replay_items]
        if len(tokens) != len(set(tokens)):
            raise ValueError("Adaptive replay tokens must be unique within the replay pool.")

        self.token_to_item = {item.token: item for item in self.replay_items}
        self.replay_indices = [item.index for item in self.replay_items]
        self.priority_size = min(int(priority_size), len(self.replay_items))
        self.seed = int(seed)
        self.refresh_interval = int(refresh_interval)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.require_full_pass = bool(require_full_pass)
        self.max_priority_per_log = max_priority_per_log

        if initial_priority_tokens:
            priority_tokens = [str(token) for token in initial_priority_tokens if str(token) in self.token_to_item]
            if len(priority_tokens) < self.priority_size:
                priority_tokens.extend(self._seed_priority_tokens(exclude=set(priority_tokens)))
            self.priority_tokens = priority_tokens[: self.priority_size]
        else:
            self.priority_tokens = self._seed_priority_tokens()

        self.generation = 0
        self.observations: Dict[str, Dict[str, Any]] = {}
        self.duplicate_observation_count = 0
        self.last_refresh: Optional[Dict[str, Any]] = None
        self.last_refresh_step = 0
        self.signature = stable_digest({
            "tokens": tokens,
            "logs": [item.log_name for item in self.replay_items],
            "start_idx": [item.start_idx for item in self.replay_items],
            "priority_size": self.priority_size,
            "seed": self.seed,
            "version": 1,
        })

    def _seed_priority_tokens(self, exclude: Optional[set[str]] = None) -> List[str]:
        exclude = exclude or set()
        tokens = [item.token for item in self.replay_items if item.token not in exclude]
        rng = random.Random(self.seed + 1709)
        rng.shuffle(tokens)
        return tokens[: self.priority_size]

    @property
    def priority_indices(self) -> List[int]:
        return [self.token_to_item[token].index for token in self.priority_tokens]

    def observe(self, token: str, drift: float, step: int, log_name: str = "") -> None:
        if token not in self.token_to_item:
            return
        drift = float(drift)
        if not math.isfinite(drift) or drift < 0:
            return
        previous = self.observations.get(token)
        if previous is not None:
            if self.require_full_pass:
                self.duplicate_observation_count += 1
                if self.duplicate_observation_count == 1:
                    logger.warning(
                        "Duplicate adaptive replay uniform observation before refresh; "
                        "keeping the largest drift per token until full unique coverage is available."
                    )
            if drift < float(previous["drift"]):
                return
        self.observations[token] = {"drift": drift, "step": int(step), "log_name": log_name}

    def should_sync(self, step: int) -> bool:
        return step > 0 and step % self.refresh_interval == 0 and step != self.last_refresh_step

    def _has_enough_observations(self, combined: Dict[str, Dict[str, Any]]) -> bool:
        if self.require_full_pass:
            return len(combined) >= len(self.replay_items)
        return len(combined) > 0

    def state_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "signature": self.signature,
            "generation": self.generation,
            "priority_tokens": list(self.priority_tokens),
            "observations": dict(self.observations),
            "duplicate_observation_count": self.duplicate_observation_count,
            "last_refresh": self.last_refresh,
            "last_refresh_step": self.last_refresh_step,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        if state.get("version") != 1 or state.get("signature") != self.signature:
            raise ValueError("Incompatible adaptive replay checkpoint state.")
        priority_tokens = [str(token) for token in state.get("priority_tokens", [])]
        if len(priority_tokens) != self.priority_size or not set(priority_tokens) <= set(self.token_to_item):
            raise ValueError("Invalid adaptive replay priority tokens in checkpoint.")
        self.generation = int(state.get("generation", 0))
        self.priority_tokens = priority_tokens
        self.observations = dict(state.get("observations", {}))
        self.duplicate_observation_count = int(state.get("duplicate_observation_count", 0))
        self.last_refresh = state.get("last_refresh")
        self.last_refresh_step = int(state.get("last_refresh_step", 0))

    def sync_state(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "generation": self.generation,
            "priority_sha256": stable_digest(self.priority_tokens),
            "observations": dict(self.observations),
            "observation_count": len(self.observations),
            "duplicate_observation_count": self.duplicate_observation_count,
        }

    def refresh_from_rank_states(self, rank_states: Sequence[Dict[str, Any]], step: int) -> Optional[Dict[str, Any]]:
        if not rank_states:
            return None
        ranks = {int(state.get("rank", -1)) for state in rank_states}
        if len(rank_states) != self.world_size or ranks != set(range(self.world_size)):
            raise ValueError("Adaptive replay refresh requires exactly one state from every rank.")
        generations = {int(state.get("generation", -1)) for state in rank_states}
        priority_hashes = {state.get("priority_sha256") for state in rank_states}
        if generations != {self.generation} or priority_hashes != {stable_digest(self.priority_tokens)}:
            raise ValueError("Adaptive replay ranks disagree on priority generation.")

        total_observations = sum(int(state.get("observation_count", len(state.get("observations", {})))) for state in rank_states)
        local_duplicate_observations = sum(int(state.get("duplicate_observation_count", 0)) for state in rank_states)
        combined: Dict[str, Dict[str, Any]] = {}
        for state in rank_states:
            for token, row in state.get("observations", {}).items():
                if token not in self.token_to_item:
                    continue
                drift = float(row.get("drift", -1.0))
                if not math.isfinite(drift) or drift < 0:
                    continue
                current = combined.get(token)
                if current is None or drift >= float(current["drift"]):
                    combined[token] = {
                        "drift": drift,
                        "step": int(row.get("step", step)),
                        "log_name": str(row.get("log_name", "")),
                    }

        cross_rank_duplicate_observations = max(0, total_observations - len(combined))
        duplicate_observations = local_duplicate_observations + cross_rank_duplicate_observations

        self.last_refresh_step = int(step)
        if self.require_full_pass:
            if len(combined) < len(self.replay_items):
                logger.info(
                    "Adaptive replay refresh skipped at step %s: partial unique uniform pass (%s/%s, rank_unique=%s, duplicates=%s).",
                    step,
                    len(combined),
                    len(self.replay_items),
                    total_observations,
                    duplicate_observations,
                )
                return None
        elif not self._has_enough_observations(combined):
            logger.info("Adaptive replay refresh skipped at step %s: no observations yet.", step)
            return None

        selected = self._select_priority(combined)
        previous = set(self.priority_tokens)
        self.priority_tokens = selected
        self.generation += 1
        self.observations = {}
        self.duplicate_observation_count = 0
        self.last_refresh = {
            "generation": self.generation,
            "step": int(step),
            "observations": len(combined),
            "duplicate_observations": duplicate_observations,
            "priority_changed": len(set(selected) - previous),
            "priority_sha256": stable_digest(selected),
            "observation_sha256": stable_digest(combined),
        }
        if hasattr(self, "_last_reference_diversity_passes"):
            self.last_refresh["diversity_pass_counts"] = self._last_reference_diversity_passes
        logger.info("Adaptive replay refreshed: %s", self.last_refresh)
        return dict(self.last_refresh)

    def _select_priority(self, observations: Dict[str, Dict[str, Any]]) -> List[str]:
        if self.reference_diverse_priority is not None and len(observations) == len(self.replay_items):
            values = {token: float(row["drift"]) for token, row in observations.items()}
            try:
                selected, passes = self.reference_diverse_priority(self.reference_rows, values, self.priority_size)
            except Exception as exc:
                logger.warning("Reference diverse_priority failed; falling back to local selector: %s", exc)
            else:
                self._last_reference_diversity_passes = dict(passes)
                return list(selected)

        if hasattr(self, "_last_reference_diversity_passes"):
            delattr(self, "_last_reference_diversity_passes")

        observed_tokens = list(observations)
        ranked = sorted(
            observed_tokens,
            key=lambda token: (-float(observations[token]["drift"]), token_hash(token)),
        )

        selected: List[str] = []
        per_log: Dict[str, int] = {}

        def try_add(token: str, enforce_log_cap: bool) -> bool:
            if token in selected:
                return False
            item = self.token_to_item[token]
            log_name = item.log_name or observations.get(token, {}).get("log_name", "")
            if enforce_log_cap and self.max_priority_per_log is not None and log_name:
                if per_log.get(log_name, 0) >= self.max_priority_per_log:
                    return False
            selected.append(token)
            if log_name:
                per_log[log_name] = per_log.get(log_name, 0) + 1
            return True

        for token in ranked:
            if len(selected) >= self.priority_size:
                break
            try_add(token, enforce_log_cap=True)

        for token in ranked + self.priority_tokens + self._seed_priority_tokens(exclude=set(selected)):
            if len(selected) >= self.priority_size:
                break
            try_add(token, enforce_log_cap=False)

        return selected[: self.priority_size]
