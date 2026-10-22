from __future__ import annotations

import copy
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers.feature_extraction_utils import BatchFeature

try:
    from navsim.agents.recogdrive.blocks.attention import Attention
except Exception:  # pragma: no cover - keeps the objective importable for non-ReCogDrive agents.
    Attention = None

logger = logging.getLogger(__name__)

_CONTEXT_MASK = ContextVar("gd_ema_context_mask", default=None)


def _clean_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    cleaned = str(value).strip().strip("'\"")
    if cleaned.lower() in {"", "none", "null"}:
        return None
    return Path(cleaned)


def _safe_key(value: str) -> str:
    return value.replace("/", "__").replace(":", "__")


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_list(value: Any, default: Sequence[str]) -> List[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(part) for part in value]


def _extract_tensor(value: Any) -> Optional[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for key in ("target", "trajectory", "final_trajectory", "pred_traj", "preferred", "action"):
            item = value.get(key)
            if isinstance(item, torch.Tensor):
                return item
    return None


def _enable_context_mask(module: torch.nn.Module) -> bool:
    if Attention is None:
        return False
    if hasattr(module, "_gd_ema_mask_handles"):
        return True

    handles = []

    def inject_mask(_module: torch.nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
        mask = _CONTEXT_MASK.get()
        if mask is not None and kwargs.get("encoder_hidden_states") is not None:
            if kwargs.get("attention_mask") is None:
                kwargs = dict(kwargs, attention_mask=mask)
                return args, kwargs
        return None

    for child in module.modules():
        if isinstance(child, Attention):
            handles.append(child.register_forward_pre_hook(inject_mask, with_kwargs=True))
    if handles:
        module._gd_ema_mask_handles = handles
        return True
    return False


@contextmanager
def _context_mask(mask: Optional[torch.Tensor]):
    token = _CONTEXT_MASK.set(mask)
    try:
        yield
    finally:
        _CONTEXT_MASK.reset(token)


class GDEMAObjective:
    """Original GD EMA continual-learning objective adapted to the Lightning loop."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = dict(cfg or {})
        self.enabled = _as_bool(self.cfg.get("enabled"), False)
        self.replace_agent_loss = _as_bool(self.cfg.get("replace_agent_loss"), True)
        self.gd_weight = float(self.cfg.get("gd_weight", 10.0))
        self.real_weight = float(self.cfg.get("real_correction_weight", self.cfg.get("failure_weight", 1.0)))
        self.synthetic_weight = float(self.cfg.get("synthetic_weight", 1.0))
        self.use_masked_context = _as_bool(self.cfg.get("use_masked_context"), True)
        self.require_replay_targets = _as_bool(self.cfg.get("require_replay_targets"), True)
        self.require_non_supervision_targets = _as_bool(self.cfg.get("require_non_supervision_targets"), True)
        self.freeze_prefixes = tuple(_as_list(self.cfg.get("freeze_prefixes"), ("feature_encoder.", "action_aware_aux_head.", "eta.")))
        self.target_bank_path = _clean_path(self.cfg.get("target_bank_path") or self.cfg.get("targets_path") or self.cfg.get("targets"))
        self.teacher: Optional[torch.nn.Module] = None
        self._target_bank: Optional[Dict[str, Any]] = None
        self._missing_targets_seen: set[str] = set()

    def setup(self, agent: Any) -> None:
        if not self.enabled:
            return
        head = getattr(agent, "action_head", None)
        if head is None:
            raise ValueError("GD EMA objective requires agent.action_head")
        self._freeze_student_prefixes(head)
        self.teacher = copy.deepcopy(head).eval()
        self.teacher.requires_grad_(False)
        logger.info("Configured GD EMA objective with gd_weight=%s target_bank=%s", self.gd_weight, self.target_bank_path)

    def _freeze_student_prefixes(self, head: torch.nn.Module) -> None:
        frozen = 0
        for name, parameter in head.named_parameters():
            if any(name.startswith(prefix) for prefix in self.freeze_prefixes):
                if parameter.requires_grad:
                    frozen += parameter.numel()
                parameter.requires_grad_(False)
        if frozen:
            logger.info("GD EMA froze %d action-head parameters for prefixes %s", frozen, self.freeze_prefixes)

    def _load_target_bank(self) -> Dict[str, Any]:
        if self._target_bank is not None:
            return self._target_bank
        if self.target_bank_path is None:
            self._target_bank = {}
            return self._target_bank
        data = torch.load(self.target_bank_path, map_location="cpu", weights_only=False)
        if isinstance(data, dict) and "targets" in data and isinstance(data["targets"], dict):
            data = data["targets"]
        if not isinstance(data, dict):
            raise ValueError(f"GD EMA target bank must be a dict or {{'targets': dict}}: {self.target_bank_path}")
        self._target_bank = data
        logger.info("Loaded GD EMA target bank with %d entries from %s", len(data), self.target_bank_path)
        return self._target_bank

    def _lookup_target(self, uid: str, token: str) -> Optional[torch.Tensor]:
        bank = self._load_target_bank()
        for key in (uid, _safe_key(uid), token, _safe_key(token)):
            if key and key in bank:
                return _extract_tensor(bank[key])
        return None

    def _role_weight(self, role: str) -> float:
        if role == "replay":
            return self.gd_weight
        if role in {"real_correction", "failure"}:
            return self.real_weight
        if role == "synthetic":
            return self.synthetic_weight
        return 0.0

    def _targets_for_batch(
        self,
        targets: torch.Tensor,
        metadata: Dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        roles = list(metadata.get("roles", []))
        uids = list(metadata.get("uids", metadata.get("tokens", [])))
        tokens = list(metadata.get("tokens", uids))
        teacher_policies = list(metadata.get("teacher_policies", [""] * len(roles)))

        selected: List[torch.Tensor] = []
        for idx, role in enumerate(roles):
            uid = str(uids[idx] if idx < len(uids) else tokens[idx])
            token = str(tokens[idx] if idx < len(tokens) else uid)
            policy = str(teacher_policies[idx] if idx < len(teacher_policies) else "")
            bank_target = self._lookup_target(uid, token)
            requires_bank = (role == "replay" and self.require_replay_targets) or (
                policy not in {"", "supervision"} and self.require_non_supervision_targets
            )
            if bank_target is None and requires_bank:
                key = uid or token
                if key not in self._missing_targets_seen:
                    self._missing_targets_seen.add(key)
                raise KeyError(
                    f"Missing GD EMA teacher target for uid={uid!r} token={token!r} role={role!r} "
                    f"policy={policy!r}. Set gd_ema.target_bank_path to teacher_targets_curated.pt."
                )
            tensor = bank_target if bank_target is not None else targets[idx].detach().cpu()
            selected.append(tensor.detach().float().reshape(8, 3))
        return torch.stack(selected, dim=0).to(device=device, dtype=dtype)

    def _context_inputs(
        self,
        head: torch.nn.Module,
        features: Dict[str, torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        context = features["last_hidden_state"].to(device=device, dtype=dtype)
        if context.ndim == 2:
            context = context.unsqueeze(0)
        lengths = features.get("last_hidden_state_lengths")
        mask = None
        context_mean = None
        if lengths is not None:
            lengths = lengths.to(device=device).long().clamp_min(1)
            valid = torch.arange(context.shape[1], device=device)[None, :] < lengths[:, None]
            context_mean = (context * valid.unsqueeze(-1).to(dtype)).sum(dim=1) / lengths.to(dtype).unsqueeze(-1)
            if self.use_masked_context:
                mask = torch.zeros_like(valid, dtype=dtype).masked_fill(~valid, float("-inf"))[:, None, None, :]
        dit_context = {"context_tokens": context, "context_mean": context_mean, "expert_step_condition": None}
        return dit_context, mask

    def _action_inputs(
        self,
        features: Dict[str, torch.Tensor],
        selected_targets: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> BatchFeature:
        history = features["history_trajectory"].to(device=device, dtype=dtype)
        status = features["status_feature"].to(device=device, dtype=dtype)
        if history.ndim == 2:
            history = history.unsqueeze(0)
        if status.ndim == 1:
            status = status.unsqueeze(0)
        history_flat = history.reshape(history.shape[0], -1)
        state = torch.cat([status, history_flat], dim=1)
        return BatchFeature(data={
            "state": state,
            "his_traj": history_flat,
            "status_feature": status,
            "action": selected_targets,
        })

    def loss(
        self,
        agent: Any,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        metadata: Dict[str, Any],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        if not self.enabled:
            return None, {}
        if not isinstance(metadata, dict):
            raise ValueError("GD EMA objective requires role metadata from RoleMixedCacheOnlyDataset")
        roles = list(metadata.get("roles", []))
        if not roles:
            return None, {}
        head = getattr(agent, "action_head", None)
        if head is None or self.teacher is None:
            raise RuntimeError("GD EMA objective has not been set up")
        if not hasattr(head, "ddpm_sqrt_alphas_cumprod"):
            raise ValueError("GD EMA objective currently requires ddpm/ddim diffusion buffers")

        device = next(head.parameters()).device
        dtype = next(head.parameters()).dtype
        teacher = self.teacher.to(device=device, dtype=dtype).eval()
        if self.use_masked_context:
            _enable_context_mask(head)
            _enable_context_mask(teacher)

        fallback_targets = targets["trajectory"].detach().cpu().float()
        selected_targets = self._targets_for_batch(fallback_targets, metadata, device=device, dtype=dtype)
        action_inputs = self._action_inputs(features, selected_targets, device=device, dtype=dtype)
        dit_context, mask = self._context_inputs(head, features, device=device, dtype=dtype)

        normalized = head.norm_odo(action_inputs.action)
        noise = torch.randn_like(normalized)
        timesteps = head.sample_time(normalized.shape[0], device=device, dtype=normalized.dtype)
        noisy = (
            head.extract(head.ddpm_sqrt_alphas_cumprod, timesteps, normalized.shape) * normalized
            + head.extract(head.ddpm_sqrt_one_minus_alphas_cumprod, timesteps, normalized.shape) * noise
        )

        with _context_mask(mask):
            prediction = head._denoise_model_output(noisy, timesteps, dit_context, action_inputs)
            desired = noise.clone()
            replay_indices = [idx for idx, role in enumerate(roles) if role == "replay"]
            if replay_indices:
                with torch.no_grad():
                    teacher_prediction = teacher._denoise_model_output(noisy, timesteps, dit_context, action_inputs)
                replay_index_tensor = torch.tensor(replay_indices, device=device, dtype=torch.long)
                desired[replay_index_tensor] = teacher_prediction[replay_index_tensor].to(dtype=desired.dtype)

        per_sample = (prediction - desired).square().flatten(1).mean(1)
        weights = torch.tensor([self._role_weight(str(role)) for role in roles], device=device, dtype=per_sample.dtype)
        active = weights != 0
        if not bool(active.any()):
            return None, {}
        loss = (per_sample * weights).sum() / max(1, len(roles))
        metrics: Dict[str, torch.Tensor] = {"gd_ema_loss": loss.detach()}
        for name, accepted in {
            "gd_ema_replay_loss": {"replay"},
            "gd_ema_real_correction_loss": {"real_correction", "failure"},
            "gd_ema_synthetic_loss": {"synthetic"},
        }.items():
            mask_tensor = torch.tensor([role in accepted for role in roles], device=device, dtype=torch.bool)
            if bool(mask_tensor.any()):
                metrics[name] = per_sample[mask_tensor].mean().detach()
        return loss, metrics
