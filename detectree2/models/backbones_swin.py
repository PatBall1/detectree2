"""
Utilities to enable the Swin Transformer backbone from the vendored SwinT_detectron2 code.

Goal: after `pip install detectree2`, users can set `use_swint_backbone=True` and train/infer
without manually cloning or pip-installing extra repositories.

Design:
- SwinT_detectron2 is vendored under: detectree2/third_party/SwinT_detectron2
- Config YAML is read from the vendored package data (read-only is fine).
- Weights are downloaded to a user-writable cache directory by default.
"""

from __future__ import annotations

import importlib
import os
import sys
import urllib.request
from pathlib import Path
from typing import Optional, Tuple


# detectree2/models/backbones_swin.py -> detectree2/
PKG_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_SWINT_ROOT = PKG_ROOT / "third_party" / "SwinT_detectron2"

# Default Swin Mask R-CNN weights (Detectron2-format) published by SwinT_detectron2
SWINT_WEIGHTS_URL = (
    "https://github.com/xiaohu2015/SwinT_detectron2/releases/download/v1.0/mask_rcnn_swint_T_coco17.pth"
)

# Default config within the vendored repo
DEFAULT_SWINT_CONFIG_REL = Path("configs") / "SwinT" / "mask_rcnn_swint_T_FPN_3x.yaml"

# Default weights filename (downloaded to cache)
DEFAULT_SWINT_WEIGHTS_NAME = "mask_rcnn_swint_T_coco17.pth"


def _cache_dir() -> Path:
    """
    Return a user-writable cache directory for detectree2 assets.
    Uses ~/.cache on Unix-y systems. On Windows, fall back to %LOCALAPPDATA%.
    """
    # Prefer XDG_CACHE_HOME if set
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "detectree2"

    # Windows fallback
    localapp = os.environ.get("LOCALAPPDATA")
    if localapp:
        return Path(localapp) / "detectree2" / "Cache"

    # Default Unix/macOS
    return Path.home() / ".cache" / "detectree2"


def _weights_cache_path(filename: str = DEFAULT_SWINT_WEIGHTS_NAME) -> Path:
    return _cache_dir() / "weights" / filename


def _validate_bundled_swint() -> None:
    """
    Validate that the vendored SwinT_detectron2 exists in the installed package.
    """
    if not BUNDLED_SWINT_ROOT.exists():
        raise FileNotFoundError(
            "Vendored SwinT_detectron2 not found inside detectree2. "
            f"Expected directory: {BUNDLED_SWINT_ROOT}\n"
            "This usually means packaging did not include detectree2/third_party/SwinT_detectron2."
        )

    swint_pkg_dir = BUNDLED_SWINT_ROOT / "swint"
    cfg_path = BUNDLED_SWINT_ROOT / DEFAULT_SWINT_CONFIG_REL
    if not swint_pkg_dir.exists():
        raise FileNotFoundError(
            f"Vendored SwinT_detectron2 is missing 'swint/' at: {swint_pkg_dir}\n"
            "Ensure your package_data includes third_party/SwinT_detectron2/swint/**/*.py"
        )
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Vendored SwinT_detectron2 is missing default config at: {cfg_path}\n"
            "Ensure your package_data includes third_party/SwinT_detectron2/**/*.yaml"
        )


def _ensure_swint_importable() -> None:
    """
    Ensure the vendored `swint` package is importable by adding its repo root to sys.path.
    """
    _validate_bundled_swint()
    root_str = str(BUNDLED_SWINT_ROOT)
    if root_str not in sys.path:
        # Put first so we prefer the vendored swint over any globally-installed package named "swint"
        sys.path.insert(0, root_str)

    # Sanity check import
    try:
        importlib.import_module("swint")
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "Failed to import vendored 'swint' package from SwinT_detectron2. "
            "This indicates the vendored code is missing or incompatible."
        ) from exc


def _default_swint_paths() -> Tuple[Path, Path]:
    """
    Return (default_config_path, default_weights_path).
    Config is read from the vendored package.
    Weights are stored in a writable cache directory.
    """
    _validate_bundled_swint()
    default_cfg = BUNDLED_SWINT_ROOT / DEFAULT_SWINT_CONFIG_REL
    default_weights = _weights_cache_path(DEFAULT_SWINT_WEIGHTS_NAME)
    return default_cfg, default_weights


def prepare_swint_config(cfg, config_path: Optional[str] = None) -> str:
    """
    Register Swin config nodes/backbone builders and return the config path to merge.

    Args:
        cfg: detectron2 config (CfgNode)
        config_path: optional override YAML path. If provided, it must exist.

    Returns:
        A string path to a Swin config YAML suitable for cfg.merge_from_file(...).
    """
    _ensure_swint_importable()

    if config_path:
        cfg_path = Path(config_path).expanduser().resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"Swin config not found at {cfg_path}")
    else:
        cfg_path, _ = _default_swint_paths()

    # Imports must happen after ensuring swint is importable
    from swint.config import add_swint_config  # type: ignore
    from swint import swin_transformer  # noqa: F401

    add_swint_config(cfg)
    return str(cfg_path)


def ensure_swint_weights(weights_path: Optional[str] = None) -> str:
    """
    Ensure SwinT weights are present locally.

    Args:
        weights_path: optional explicit path to weights. If provided, must exist.

    Returns:
        A string path to a local .pth weights file.
    """
    if weights_path:
        target = Path(weights_path).expanduser().resolve()
        if target.exists():
            return str(target)
        raise FileNotFoundError(
            f"Swin weights not found at {target}. Provide a valid path or omit "
            "swint_weights_path to auto-download default weights."
        )

    _, default_weights = _default_swint_paths()
    target = default_weights
    if target.exists():
        return str(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")

    try:
        with urllib.request.urlopen(SWINT_WEIGHTS_URL) as resp, open(tmp, "wb") as out:
            out.write(resp.read())
        tmp.replace(target)
    except Exception as exc:  # noqa: BLE001
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"Failed to download Swin weights from {SWINT_WEIGHTS_URL}. "
            f"Try downloading manually and placing them at {target}."
        ) from exc

    return str(target)
