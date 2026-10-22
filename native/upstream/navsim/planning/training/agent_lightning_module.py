import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn.functional as F

from torch import Tensor
from typing import Any, Dict, List, Optional, Tuple

from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.training.adaptive_replay import AdaptiveReplayController, TrajectoryReferenceProvider
from navsim.planning.training.adaptive_replay.reference_objectives import ReferenceObjectiveAdapter
from navsim.planning.training.gd_ema_objective import GDEMAObjective


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent
        self.adaptive_replay_controller: Optional[AdaptiveReplayController] = None
        self.adaptive_replay_loss_cfg: Dict[str, Any] = {}
        self.reference_provider = TrajectoryReferenceProvider()
        self.reference_objective_adapter: Optional[ReferenceObjectiveAdapter] = None
        self.gd_ema_objective: Optional[GDEMAObjective] = None

    def configure_adaptive_replay(
        self,
        controller: AdaptiveReplayController,
        loss_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.adaptive_replay_controller = controller
        self.adaptive_replay_loss_cfg = dict(loss_cfg or {})
        self.reference_objective_adapter = ReferenceObjectiveAdapter(self.adaptive_replay_loss_cfg)

    def configure_gd_ema(self, loss_cfg: Optional[Dict[str, Any]] = None) -> None:
        self.gd_ema_objective = GDEMAObjective(loss_cfg or {})
        self.gd_ema_objective.setup(self.agent)

    def _metadata_tokens(self, metadata: Any) -> Any:
        if isinstance(metadata, dict):
            return metadata.get("tokens")
        return metadata

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets, metadata = batch
        tokens_list = self._metadata_tokens(metadata)
        gd_ema_active = (
            logging_prefix == "train"
            and self.gd_ema_objective is not None
            and self.gd_ema_objective.enabled
        )
        replace_agent_loss = (
            logging_prefix == "train"
            and self.adaptive_replay_controller is not None
            and self._should_use_reference_objectives()
            and bool(self.adaptive_replay_loss_cfg.get("replace_agent_loss", False))
        ) or (gd_ema_active and bool(self.gd_ema_objective.replace_agent_loss))

        loss: Optional[Tensor]
        if replace_agent_loss:
            loss = None
        else:
            prediction = self.agent.forward(features, targets, tokens_list)
            loss = self.agent.compute_loss(features, targets, prediction)

        if logging_prefix == "train" and self.adaptive_replay_controller is not None:
            adaptive_loss = self._compute_adaptive_replay_loss(features, targets, metadata)
            if adaptive_loss is not None:
                loss = adaptive_loss if loss is None else loss + adaptive_loss

        if gd_ema_active:
            gd_loss, gd_metrics = self.gd_ema_objective.loss(self.agent, features, targets, metadata)
            if gd_loss is not None:
                loss = gd_loss if loss is None else loss + gd_loss
                for metric_name, metric_value in gd_metrics.items():
                    self.log(f"train/{metric_name}", metric_value, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        if loss is None:
            raise RuntimeError("Training loss was configured to be replaced, but no replacement objective produced a loss.")

        self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def _should_use_reference_objectives(self) -> bool:
        adapter = self.reference_objective_adapter
        if adapter is None:
            return False
        if bool(self.adaptive_replay_loss_cfg.get("use_reference_objectives", False)):
            return True
        return (
            adapter.replay_path_weight != 0.0
            or adapter.failure_sft_weight != 0.0
            or adapter.synthetic_sft_weight != 0.0
            or bool(self.adaptive_replay_loss_cfg.get("native_direct", False))
        )


    def _compute_adaptive_replay_loss(
        self,
        features: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        metadata: Any,
    ) -> Optional[Tensor]:
        if self._should_use_reference_objectives():
            return self._compute_reference_adaptive_replay_loss(features, targets, metadata)

        if not isinstance(metadata, dict):
            return None

        roles: List[str] = list(metadata.get("roles", []))
        tokens: List[str] = list(metadata.get("tokens", []))
        uids: List[str] = list(metadata.get("uids", tokens))
        scene_ids: List[str] = list(metadata.get("scene_ids", tokens))
        replay_streams: List[Optional[str]] = list(metadata.get("replay_streams", []))
        teacher_roots: List[Optional[str]] = list(metadata.get("teacher_cache_paths", []))
        log_names: List[str] = list(metadata.get("log_names", []))
        if not roles:
            return None

        role_weights = {
            "replay": float(self.adaptive_replay_loss_cfg.get("replay_direct_weight", 0.0)),
            "failure": float(self.adaptive_replay_loss_cfg.get("failure_direct_weight", 0.0)),
            "synthetic": float(self.adaptive_replay_loss_cfg.get("synthetic_direct_weight", 0.0)),
        }
        if all(weight == 0.0 for weight in role_weights.values()):
            return None

        deterministic = bool(self.adaptive_replay_loss_cfg.get("deterministic", True))
        beta = float(self.adaptive_replay_loss_cfg.get("smooth_l1_beta", 0.05))
        student = self.agent.predict_final_trajectory_from_features(features, deterministic=deterministic)
        target_device = student.device
        batch_targets = targets["trajectory"].to(target_device, dtype=student.dtype)

        sample_losses: List[Tensor] = []
        sample_weights: List[float] = []
        uniform_observations: List[Tuple[str, float, str]] = []

        for idx, role in enumerate(roles):
            weight = role_weights.get(role, 0.0)
            if weight == 0.0:
                continue

            reference = None
            if role == "replay":
                reference = self.reference_provider.load(
                    tokens[idx],
                    teacher_roots[idx] if idx < len(teacher_roots) else None,
                    log_names[idx] if idx < len(log_names) else "",
                )
            if reference is None and role in {"failure", "synthetic"}:
                reference = batch_targets[idx].detach().cpu()
            if reference is None:
                continue

            reference = reference.to(target_device, dtype=student.dtype)
            if reference.ndim == 3 and reference.shape[0] == 1:
                reference = reference.squeeze(0)
            per_sample = F.smooth_l1_loss(student[idx], reference, beta=beta, reduction="none").mean()
            sample_losses.append(per_sample)
            sample_weights.append(weight)

            if role == "replay" and idx < len(replay_streams) and replay_streams[idx] == "uniform":
                drift = torch.linalg.norm((student[idx, :, :2].detach() - reference[:, :2].detach()).float(), dim=-1).mean()
                uniform_observations.append((tokens[idx], float(drift.cpu()), log_names[idx] if idx < len(log_names) else ""))

        controller = self.adaptive_replay_controller
        if controller is not None and uniform_observations:
            step = int(self.trainer.global_step) + 1 if self.trainer is not None else 0
            for token, drift, log_name in uniform_observations:
                controller.observe(token, drift, step=step, log_name=log_name)
            self._maybe_refresh_adaptive_replay(step)

        if not sample_losses:
            return None
        weights = torch.tensor(sample_weights, device=target_device, dtype=student.dtype)
        losses = torch.stack(sample_losses)
        direct_loss = (losses * weights).sum() / max(1, len(roles))
        self.log("train/adaptive_direct_loss", direct_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        return direct_loss


    def _compute_reference_adaptive_replay_loss(
        self,
        features: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        metadata: Any,
    ) -> Optional[Tensor]:
        if not isinstance(metadata, dict) or self.reference_objective_adapter is None:
            return None
        head = getattr(self.agent, "action_head", None)
        if head is None:
            return None

        roles: List[str] = list(metadata.get("roles", []))
        tokens: List[str] = list(metadata.get("tokens", []))
        uids: List[str] = list(metadata.get("uids", tokens))
        scene_ids: List[str] = list(metadata.get("scene_ids", tokens))
        replay_streams: List[Optional[str]] = list(metadata.get("replay_streams", []))
        teacher_roots: List[Optional[str]] = list(metadata.get("teacher_cache_paths", []))
        log_names: List[str] = list(metadata.get("log_names", []))
        override_paths: List[Optional[str]] = list(metadata.get("target_override_paths", []))
        if not roles:
            return None

        adapter = self.reference_objective_adapter
        payloads = adapter.payloads_from_batch(head, features)
        batch_targets_cpu = targets["trajectory"].detach().cpu().float()
        device = next(head.parameters()).device
        total = next(head.parameters()).new_tensor(0.0)
        active_terms = 0
        preferred_targets: Dict[int, torch.Tensor] = {}

        replay_indices = [idx for idx, role in enumerate(roles) if role == "replay"]
        failure_indices = [idx for idx, role in enumerate(roles) if role in {"failure", "real_correction"}]
        synthetic_indices = [idx for idx, role in enumerate(roles) if role == "synthetic"]

        path_losses, replay_final_targets = adapter.replay_path_losses(head, payloads, tokens, replay_indices)
        preferred_targets.update(replay_final_targets)
        if path_losses is not None and adapter.replay_path_weight != 0.0:
            total = total + path_losses.sum() * adapter.replay_path_weight
            active_terms += int(path_losses.numel())
            self.log("train/reference_replay_path_loss", path_losses.mean(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        if adapter.failure_sft_weight != 0.0:
            failure_losses, failure_targets = adapter.correction_sft_losses(
                head, payloads, uids, batch_targets_cpu, failure_indices, override_paths
            )
            preferred_targets.update(failure_targets)
            if failure_losses is not None:
                total = total + failure_losses.sum() * adapter.failure_sft_weight
                active_terms += int(failure_losses.numel())
                self.log("train/reference_failure_sft_loss", failure_losses.mean(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        if adapter.synthetic_sft_weight != 0.0:
            synthetic_losses, synthetic_targets = adapter.correction_sft_losses(
                head, payloads, uids, batch_targets_cpu, synthetic_indices, override_paths
            )
            preferred_targets.update(synthetic_targets)
            if synthetic_losses is not None:
                total = total + synthetic_losses.sum() * adapter.synthetic_sft_weight
                active_terms += int(synthetic_losses.numel())
                self.log("train/reference_synthetic_sft_loss", synthetic_losses.mean(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        role_direct_weights = {
            "replay": float(self.adaptive_replay_loss_cfg.get("replay_direct_weight", 0.0)),
            "failure": float(self.adaptive_replay_loss_cfg.get("failure_direct_weight", 0.0)),
            "real_correction": float(self.adaptive_replay_loss_cfg.get("failure_direct_weight", 0.0)),
            "synthetic": float(self.adaptive_replay_loss_cfg.get("synthetic_direct_weight", 0.0)),
        }
        direct_indices: List[int] = []
        direct_targets: List[torch.Tensor] = []
        direct_weights: List[float] = []
        for idx, role in enumerate(roles):
            weight = role_direct_weights.get(role, 0.0)
            if weight == 0.0:
                continue
            reference = preferred_targets.get(idx)
            if reference is None and role == "replay":
                reference = self.reference_provider.load(
                    tokens[idx],
                    teacher_roots[idx] if idx < len(teacher_roots) else None,
                    log_names[idx] if idx < len(log_names) else "",
                )
            if reference is None and role in {"failure", "real_correction", "synthetic"}:
                reference = adapter.preferred_target(
                    uids[idx] if idx < len(uids) else tokens[idx],
                    batch_targets_cpu[idx],
                    override_paths[idx] if idx < len(override_paths) else None,
                )
            if reference is None:
                reference = batch_targets_cpu[idx]
            if reference.ndim == 3 and reference.shape[0] == 1:
                reference = reference.squeeze(0)
            direct_indices.append(idx)
            direct_targets.append(reference.detach().cpu().float().reshape(8, 3))
            direct_weights.append(weight)

        uniform_observations: List[Tuple[str, float, str]] = []
        if direct_indices:
            if adapter.native_direct:
                direct_payloads = [payloads[idx] for idx in direct_indices]
                direct_seed_keys = [scene_ids[idx] if idx < len(scene_ids) else tokens[idx] for idx in direct_indices]
                direct_losses, direct_predictions = adapter.native_direct_outputs(head, direct_payloads, direct_targets, direct_seed_keys)
                weights = torch.tensor(direct_weights, device=device, dtype=direct_losses.dtype)
                total = total + (direct_losses * weights).sum()
                active_terms += int(direct_losses.numel())
                self.log("train/reference_native_direct_loss", direct_losses.mean(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
                for local_idx, batch_idx in enumerate(direct_indices):
                    if roles[batch_idx] == "replay" and batch_idx < len(replay_streams) and replay_streams[batch_idx] == "uniform":
                        target_xy = direct_targets[local_idx].to(device=direct_predictions.device, dtype=direct_predictions.dtype)[:, :2]
                        drift = torch.linalg.norm((direct_predictions[local_idx, :, :2].detach() - target_xy).float(), dim=-1).mean()
                        uniform_observations.append((tokens[batch_idx], float(drift.cpu()), log_names[batch_idx] if batch_idx < len(log_names) else ""))
            else:
                deterministic = bool(self.adaptive_replay_loss_cfg.get("deterministic", True))
                beta = float(self.adaptive_replay_loss_cfg.get("smooth_l1_beta", 0.05))
                student = self.agent.predict_final_trajectory_from_features(features, deterministic=deterministic)
                direct_losses = []
                for batch_idx, reference in zip(direct_indices, direct_targets):
                    reference_device = reference.to(device=student.device, dtype=student.dtype)
                    direct_losses.append(F.smooth_l1_loss(student[batch_idx], reference_device, beta=beta, reduction="none").mean())
                    if roles[batch_idx] == "replay" and batch_idx < len(replay_streams) and replay_streams[batch_idx] == "uniform":
                        drift = torch.linalg.norm((student[batch_idx, :, :2].detach() - reference_device[:, :2].detach()).float(), dim=-1).mean()
                        uniform_observations.append((tokens[batch_idx], float(drift.cpu()), log_names[batch_idx] if batch_idx < len(log_names) else ""))
                direct_losses_tensor = torch.stack(direct_losses)
                weights = torch.tensor(direct_weights, device=student.device, dtype=student.dtype)
                total = total + (direct_losses_tensor * weights).sum()
                active_terms += int(direct_losses_tensor.numel())
                self.log("train/reference_direct_loss", direct_losses_tensor.mean(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        controller = self.adaptive_replay_controller
        if controller is not None and uniform_observations:
            step = int(self.trainer.global_step) + 1 if self.trainer is not None else 0
            for token, drift, log_name in uniform_observations:
                controller.observe(token, drift, step=step, log_name=log_name)
            self._maybe_refresh_adaptive_replay(step)

        if active_terms == 0:
            return None
        reference_loss = total / max(1, len(roles))
        self.log("train/reference_objective_loss", reference_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        return reference_loss

    def _maybe_refresh_adaptive_replay(self, step: int) -> None:
        controller = self.adaptive_replay_controller
        if controller is None or not controller.should_sync(step):
            return

        local_state = controller.sync_state()
        if dist.is_available() and dist.is_initialized():
            states: List[Optional[Dict[str, Any]]] = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(states, local_state)
            rank_states = [state for state in states if state is not None]
        else:
            rank_states = [local_state]

        refresh = controller.refresh_from_rank_states(rank_states, step=step)
        if refresh is not None:
            self.log("train/adaptive_replay_generation", float(refresh["generation"]), on_step=True, prog_bar=False, sync_dist=True)
            self.log("train/adaptive_replay_observations", float(refresh["observations"]), on_step=True, prog_bar=False, sync_dist=True)
            self.log("train/adaptive_replay_changed", float(refresh["priority_changed"]), on_step=True, prog_bar=False, sync_dist=True)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        每次保存 checkpoint 时，只保留 state_dict 中不以 'agent.model' 开头的条目。
        """
        if self.adaptive_replay_controller is not None:
            checkpoint["adaptive_replay"] = self.adaptive_replay_controller.state_dict()
        save_backbone_base = bool(getattr(self.agent, "save_backbone_base", True))
        filtered_sd = {}
        for k, v in checkpoint['state_dict'].items():
            if k.startswith('agent.model'):
                continue
            if not save_backbone_base and k.startswith('agent.backbone.'):
                if '.lora_' in k:
                    filtered_sd[k] = v
                continue
            filtered_sd[k] = v
        checkpoint['state_dict'] = filtered_sd

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.adaptive_replay_controller is not None and "adaptive_replay" in checkpoint:
            self.adaptive_replay_controller.load_state_dict(checkpoint["adaptive_replay"])

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()


class AgentLightningDiT(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets, tokens_list = batch
        prediction = self.agent.forward(features,targets,tokens_list)
        if logging_prefix == 'train':
            predictions = self.agent.compute_loss(features, targets, prediction)

            loss = predictions.loss
            reward = predictions.reward
            policy_loss = predictions.policy_loss
            bc_loss = predictions.bc_loss
            self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/reward", reward, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/policy_loss", policy_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/bc_loss", bc_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            prediction = self.agent.forward(features,targets)
            loss = self.agent.compute_loss(features, targets, prediction)
            self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        每次保存 checkpoint 时，只保留 state_dict 中不以 'agent.model' 开头的条目。
        """
        filtered_sd = {
            k: v
            for k, v in checkpoint['state_dict'].items()
            if not k.startswith('agent.model')
        }
        checkpoint['state_dict'] = filtered_sd

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        #print(batch_idx)
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()