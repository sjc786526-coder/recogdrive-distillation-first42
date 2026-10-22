from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from navsim.planning.training.adaptive_replay.distillation import safe_token
from navsim.planning.training.adaptive_replay.reference_bridge import ensure_reference_src_on_path

logger = logging.getLogger(__name__)


def _clean_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    cleaned = str(value).strip().strip("'\"")
    if cleaned.lower() in {"", "none", "null"}:
        return None
    return Path(cleaned)


class ReferenceObjectiveAdapter:
    """Calls the adaptive SFT handoff objective code from the current Lightning loop."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = dict(cfg or {})
        self.replay_trace_path = _clean_path(
            self.cfg.get("replay_trace_path")
            or self.cfg.get("replay_paths")
            or self.cfg.get("replay_paths_path")
        )
        self.preference_targets_path = _clean_path(
            self.cfg.get("preference_targets_path")
            or self.cfg.get("preference_targets")
        )
        self.rollout_batch_size = max(1, int(self.cfg.get("rollout_batch_size", 8)))
        self.preference_beta = float(self.cfg.get("preference_beta", self.cfg.get("beta", 10.0)))
        self.preference_weight = float(self.cfg.get("preference_weight", 0.0))
        self.direct_beta = float(self.cfg.get("smooth_l1_beta", self.cfg.get("direct_beta", 0.05)))
        self.native_direct = bool(self.cfg.get("native_direct", False))
        self._targets_bank: Optional[Dict[str, Any]] = None
        self._loaded = False

    @property
    def replay_path_weight(self) -> float:
        return float(self.cfg.get("replay_path_weight", self.cfg.get("path_weight", 0.0)))

    @property
    def failure_sft_weight(self) -> float:
        return float(self.cfg.get("failure_sft_weight", self.cfg.get("real_sft_weight", 0.0)))

    @property
    def synthetic_sft_weight(self) -> float:
        return float(self.cfg.get("synthetic_sft_weight", self.cfg.get("synthetic_weight", 0.0)))

    def enabled(self) -> bool:
        return any(
            weight != 0.0
            for weight in (self.replay_path_weight, self.failure_sft_weight, self.synthetic_sft_weight)
        ) or self.native_direct

    def _load_modules(self) -> None:
        if self._loaded:
            return
        ensure_reference_src_on_path()
        self.replay_paths = importlib.import_module("tools.training.simscale_continual.replay_paths")
        self.masked_batching = importlib.import_module("tools.training.simscale_continual.masked_batching")
        self.preference_batching = importlib.import_module("tools.training.simscale_continual.preference_batching")
        self.rollout_batching = importlib.import_module("tools.training.simscale_continual.rollout_batching")
        self.common = importlib.import_module("tools.training.simscale_sft.common")
        self._loaded = True

    def payloads_from_batch(self, head: torch.nn.Module, features: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        device = next(head.parameters()).device
        context = features["last_hidden_state"].to(device=device, dtype=torch.float32)
        history = features["history_trajectory"].detach().float().cpu()
        command = features["high_command_one_hot"].detach().float().cpu()
        status = features["status_feature"].detach().float().cpu()
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context.shape[-1] != head.config.input_embedding_dim:
            context = head.feature_encoder(context).float()
        if history.ndim == 2:
            history = history.unsqueeze(0)
        if command.ndim == 1:
            command = command.unsqueeze(0)
        if status.ndim == 1:
            status = status.unsqueeze(0)
        return [
            {
                "context": context[idx],
                "history": history[idx],
                "command": command[idx],
                "status": status[idx],
            }
            for idx in range(context.shape[0])
        ]

    def _load_targets_bank(self) -> Dict[str, Any]:
        if self._targets_bank is not None:
            return self._targets_bank
        if self.preference_targets_path is None:
            self._targets_bank = {}
            return self._targets_bank
        data = torch.load(self.preference_targets_path, map_location="cpu", weights_only=False)
        self._targets_bank = data.get("targets", data) if isinstance(data, dict) else {}
        return self._targets_bank

    def _target_from_bank(
        self,
        token: str,
        fallback: torch.Tensor,
        override_path: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        bank = self._load_targets_bank()
        target = bank.get(token) or bank.get(safe_token(token))
        if target is None and override_path:
            path = _clean_path(override_path)
            if path is not None and path.is_file():
                data = torch.load(path, map_location="cpu", weights_only=False)
                target = data.get("targets", data).get(token) if isinstance(data, dict) and "targets" in data else data
        preferred = fallback.detach().cpu().float()
        rejected = preferred
        if isinstance(target, dict):
            preferred = target.get("preferred", target.get("trajectory", preferred))
            rejected = target.get("rejected", preferred)
        elif isinstance(target, torch.Tensor):
            preferred = target
            rejected = target
        return {
            "preferred": preferred.detach().cpu().float().reshape(8, 3),
            "rejected": rejected.detach().cpu().float().reshape(8, 3),
        }


    def preferred_target(
        self,
        token: str,
        fallback: torch.Tensor,
        override_path: Optional[str] = None,
    ) -> torch.Tensor:
        return self._target_from_bank(token, fallback, override_path)["preferred"]

    def _trace_root(self) -> Optional[Path]:
        if self.replay_trace_path is None:
            return None
        traces = self.replay_trace_path / "traces"
        return traces if traces.is_dir() else self.replay_trace_path

    def _find_trace_path(self, token: str) -> Optional[Path]:
        root = self._trace_root()
        if root is None:
            return None
        for stem in (token, safe_token(token)):
            candidate = root / f"{stem}.pt"
            if candidate.is_file():
                return candidate
        return None

    def load_trace(self, token: str) -> Optional[Dict[str, Any]]:
        path = self._find_trace_path(token)
        if path is None:
            return None
        return torch.load(path, map_location="cpu", weights_only=False)

    def replay_path_losses(
        self,
        head: torch.nn.Module,
        payloads: Sequence[Dict[str, torch.Tensor]],
        tokens: Sequence[str],
        indices: Sequence[int],
    ) -> Tuple[Optional[torch.Tensor], Dict[int, torch.Tensor]]:
        if self.replay_path_weight == 0.0:
            return None, {}
        self._load_modules()
        examples = []
        final_predictions: Dict[int, torch.Tensor] = {}
        for batch_idx in indices:
            trace = self.load_trace(tokens[batch_idx])
            if trace is None:
                continue
            steps = int(len(trace["timesteps"]))
            step_idx = int(torch.randint(steps, (1,)).item())
            examples.append(self.replay_paths.example(head, payloads[batch_idx], trace, step_idx))
            if isinstance(trace.get("final_prediction"), torch.Tensor):
                final_predictions[batch_idx] = trace["final_prediction"].detach().cpu().float().reshape(8, 3)
        if not examples:
            return None, final_predictions
        return self.masked_batching.loss_vector(head, None, examples, replay=False), final_predictions

    def correction_sft_losses(
        self,
        head: torch.nn.Module,
        payloads: Sequence[Dict[str, torch.Tensor]],
        tokens: Sequence[str],
        targets: torch.Tensor,
        indices: Sequence[int],
        override_paths: Sequence[Optional[str]],
    ) -> Tuple[Optional[torch.Tensor], Dict[int, torch.Tensor]]:
        if not indices:
            return None, {}
        self._load_modules()
        selected_payloads = [payloads[idx] for idx in indices]
        selected_targets = [
            self._target_from_bank(tokens[idx], targets[idx], override_paths[idx] if idx < len(override_paths) else None)
            for idx in indices
        ]
        examples = self.preference_batching.prepare_pairs(head, selected_payloads, selected_targets)
        losses, _diagnostics = self.preference_batching.combined_loss(
            head,
            None,
            examples,
            beta=self.preference_beta,
            preference_weight=self.preference_weight,
        )
        direct_targets = {idx: target["preferred"] for idx, target in zip(indices, selected_targets)}
        return losses, direct_targets

    def native_direct_outputs(
        self,
        head: torch.nn.Module,
        payloads: Sequence[Dict[str, torch.Tensor]],
        targets: Sequence[torch.Tensor],
        tokens: Sequence[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._load_modules()
        for attr in ("use_expert_features", "use_bit_drive", "use_last_rd"):
            if not hasattr(head.config, attr):
                setattr(head.config, attr, False)
        losses = []
        predictions = []
        for begin in range(0, len(payloads), self.rollout_batch_size):
            group_payloads = list(payloads[begin: begin + self.rollout_batch_size])
            group_targets = list(targets[begin: begin + self.rollout_batch_size])
            group_tokens = list(tokens[begin: begin + self.rollout_batch_size])
            seeds = [self.common.seed_for(token) for token in group_tokens]
            prediction = self.rollout_batching.native_rollout_batch(head, group_payloads, seeds)
            predictions.append(prediction)
            losses.append(self.rollout_batching.trajectory_loss_vector(head, prediction, group_targets, beta=self.direct_beta))
        if not losses:
            empty = next(head.parameters()).new_zeros(0)
            return empty, empty.reshape(0, 8, 3)
        return torch.cat(losses), torch.cat(predictions)

    def native_direct_losses(
        self,
        head: torch.nn.Module,
        payloads: Sequence[Dict[str, torch.Tensor]],
        targets: Sequence[torch.Tensor],
        tokens: Sequence[str],
    ) -> torch.Tensor:
        losses, _predictions = self.native_direct_outputs(head, payloads, targets, tokens)
        return losses
