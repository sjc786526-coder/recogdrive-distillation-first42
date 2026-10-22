from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_REFERENCE_ENV = "RECOGDRIVE_ADAPTIVE_REPLAY_REFERENCE_SRC"
_REFERENCE_MODULE = "tools.training.simscale_continual.adaptive_replay_sampling"


def _default_reference_src() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "scripts" / "recogdrive_adaptive_replay_sft_handoff_20260923" / "src"


def _reference_src() -> Path:
    return Path(os.environ.get(_REFERENCE_ENV, _default_reference_src())).resolve()


def ensure_reference_src_on_path() -> Optional[Path]:
    src = _reference_src()
    if not src.is_dir():
        logger.warning("Adaptive replay reference source is unavailable: %s", src)
        return None
    src_string = str(src)
    if src_string not in sys.path:
        sys.path.append(src_string)
    return src


def load_diverse_priority() -> Optional[Callable[[Sequence[Dict[str, Any]], Dict[str, float], int], Tuple[List[str], Dict[str, int]]]]:
    src = ensure_reference_src_on_path()
    if src is None:
        return None

    try:
        module = importlib.import_module(_REFERENCE_MODULE)
    except Exception as exc:
        logger.warning("Failed to import adaptive replay reference source from %s: %s", src, exc)
        return None

    diverse_priority = getattr(module, "diverse_priority", None)
    if not callable(diverse_priority):
        logger.warning("Reference module %s does not expose diverse_priority.", _REFERENCE_MODULE)
        return None
    return diverse_priority
