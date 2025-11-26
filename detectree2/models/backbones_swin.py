"""Utilities to enable the Swin Transformer backbone from SwinT_detectron2."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLED_SWINT = REPO_ROOT / "third_party" / "SwinT_detectron2"
SWINT_PACKAGE = BUNDLED_SWINT / "swint"


def _get_swint_root() -> Path:
    """Load the vendored SwinT_detectron2 and return its root directory."""
    if not SWINT_PACKAGE.exists():
        raise FileNotFoundError(
            "Vendored SwinT_detectron2 not found. Clone https://github.com/xiaohu2015/SwinT_detectron2.git "
            f"into {BUNDLED_SWINT} so that {SWINT_PACKAGE} exists."
        )

    if str(BUNDLED_SWINT) not in sys.path:
        sys.path.insert(0, str(BUNDLED_SWINT))

    try:
        swint_module = importlib.import_module("swint")
    except ImportError as exc:
        raise FileNotFoundError(
            "Failed to import vendored SwinT_detectron2. Ensure third_party/SwinT_detectron2 is present "
            "and contains the swint package."
        ) from exc

    swint_root = Path(swint_module.__file__).resolve().parent.parent
    if not (swint_root / "configs").exists() or not (swint_root / "models").exists():
        raise FileNotFoundError(
            f"Vendored SwinT_detectron2 at {BUNDLED_SWINT} is missing configs/ or models/. "
            "Ensure you cloned the full repository."
        )

    return swint_root


SWINT_ROOT = _get_swint_root()
DEFAULT_SWINT_CONFIG = SWINT_ROOT / "configs" / "SwinT" / "mask_rcnn_swint_T_FPN_3x.yaml"
DEFAULT_SWINT_WEIGHTS = SWINT_ROOT / "models" / "swin_tiny_patch4_window7_224_d2.pth"


def prepare_swint_config(
    cfg,
    config_path: Optional[str] = None,
) -> str:
    """Register Swin config nodes/backbone builders and return the config path to merge."""
    cfg_path = Path(config_path) if config_path else DEFAULT_SWINT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Swin config not found at {cfg_path}. Verify your vendored SwinT_detectron2 or override config_path."
        )

    # Imports must happen after ensuring swint is importable
    from swint.config import add_swint_config  # type: ignore
    from swint import swin_transformer  # noqa: F401

    add_swint_config(cfg)
    return str(cfg_path)
