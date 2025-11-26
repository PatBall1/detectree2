"""Utilities to enable the Swin Transformer backbone from SwinT_detectron2."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Optional

def _get_swint_root() -> Path:
    """Locate the SwinT_detectron2 install (pip-installed)."""
    try:
        swint_module = importlib.import_module("swint")
        swint_root = Path(swint_module.__file__).resolve().parent
    except ImportError as exc:
        raise FileNotFoundError(
            "SwinT_detectron2 is not available. Install it with "
            "`pip install git+https://github.com/xiaohu2015/SwinT_detectron2.git#egg=swint`."
        ) from exc

    if not swint_root.exists():
        raise FileNotFoundError(
            "SwinT_detectron2 is not available. Install it with "
            "`pip install git+https://github.com/xiaohu2015/SwinT_detectron2.git#egg=swint`."
        )
    return swint_root


SWINT_ROOT = _get_swint_root()
DEFAULT_SWINT_CONFIG = (
    SWINT_ROOT / "configs" / "SwinT" / "mask_rcnn_swint_T_FPN_3x.yaml"
)
DEFAULT_SWINT_WEIGHTS = SWINT_ROOT / "models" / "swin_tiny_patch4_window7_224_d2.pth"


def prepare_swint_config(
    cfg,
    config_path: Optional[str] = None,
) -> str:
    """Register Swin config nodes/backbone builders and return the config path to merge."""
    swint_root = _get_swint_root()

    cfg_path = Path(config_path) if config_path else swint_root / "configs" / "SwinT" / "mask_rcnn_swint_T_FPN_3x.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Swin config not found at {cfg_path}. Verify your swint installation or override config_path."
        )

    # Imports must happen after ensuring swint is importable
    from swint.config import add_swint_config  # type: ignore
    from swint import swin_transformer  # noqa: F401

    # Register custom config nodes/backbone builders
    add_swint_config(cfg)
    return str(cfg_path)
