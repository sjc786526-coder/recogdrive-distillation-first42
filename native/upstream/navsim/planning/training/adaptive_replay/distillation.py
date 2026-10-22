from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def safe_token(token: str) -> str:
    return token.replace("/", "__").replace(":", "__")


class TrajectoryReferenceProvider:
    """Loads cached teacher or preferred final trajectories for direct distillation."""

    def __init__(self) -> None:
        self._memory: Dict[str, torch.Tensor] = {}

    def load(self, token: str, root: Optional[str], log_name: str = "") -> Optional[torch.Tensor]:
        if not root:
            return None
        key = f"{root}|{log_name}|{token}"
        if key in self._memory:
            return self._memory[key]
        path = self._find_path(Path(root), token, log_name)
        if path is None:
            return None
        tensor = self._load_tensor(path)
        if tensor is not None:
            self._memory[key] = tensor.float()
        return tensor

    def _find_path(self, root: Path, token: str, log_name: str) -> Optional[Path]:
        stem_options = [token, safe_token(token)]
        suffixes = [".pt", ".pth", ".pkl", ".pickle", ".gz"]
        parents = [root]
        if log_name:
            parents.insert(0, root / log_name)
        for parent in parents:
            for stem in stem_options:
                direct = parent / stem
                if direct.is_file():
                    return direct
                for suffix in suffixes:
                    candidate = parent / f"{stem}{suffix}"
                    if candidate.is_file():
                        return candidate
        return None

    def _load_tensor(self, path: Path) -> Optional[torch.Tensor]:
        if path.suffix == ".gz":
            with gzip.open(path, "rb") as handle:
                data = pickle.load(handle)
        elif path.suffix in {".pkl", ".pickle"}:
            with path.open("rb") as handle:
                data = pickle.load(handle)
        else:
            data = torch.load(path, map_location="cpu")
        return self._extract_tensor(data)

    def _extract_tensor(self, data: Any) -> Optional[torch.Tensor]:
        if isinstance(data, torch.Tensor):
            return data
        if isinstance(data, dict):
            for key in ("final_trajectory", "pred_traj", "trajectory", "target", "action"):
                value = data.get(key)
                if isinstance(value, torch.Tensor):
                    return value
            nested = data.get("targets")
            if isinstance(nested, dict):
                for key in ("preferred", "final_trajectory", "trajectory"):
                    value = nested.get(key)
                    if isinstance(value, torch.Tensor):
                        return value
        return None
