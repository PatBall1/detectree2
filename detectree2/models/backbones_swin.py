"""Utilities to enable the Swin Transformer backbone from third_party/SwinT_detectron2."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

SWINT_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "SwinT_detectron2"
DEFAULT_SWINT_CONFIG = (
    SWINT_ROOT / "configs" / "SwinT" / "mask_rcnn_swint_T_FPN_3x.yaml"
)
DEFAULT_SWINT_WEIGHTS = SWINT_ROOT / "models" / "swin_tiny_patch4_window7_224_d2.pth"


def _ensure_swint_on_path() -> None:
    """Make sure the vendored Swin repo is importable before touching detectron config."""
    swint_path = str(SWINT_ROOT)
    if swint_path not in sys.path:
        sys.path.insert(0, swint_path)


def prepare_swint_config(
    cfg,
    config_path: Optional[str] = None,
) -> str:
    """Register Swin config nodes/backbone builders and return the config path to merge."""
    if not SWINT_ROOT.exists():
        raise FileNotFoundError(
            f"SwinT_detectron2 repo is missing at {SWINT_ROOT}. "
            "Clone https://github.com/xiaohu2015/SwinT_detectron2 into third_party first."
        )

    _ensure_swint_on_path()
    # Imports must happen after sys.path mutation
    from swint.config import add_swint_config  # type: ignore
    from swint import swin_transformer  # noqa: F401

    add_swint_config(cfg)
    cfg_path = Path(config_path) if config_path else DEFAULT_SWINT_CONFIG
    return str(cfg_path)

