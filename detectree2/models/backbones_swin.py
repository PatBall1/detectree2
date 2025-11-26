"""Utilities to enable the Swin Transformer backbone from SwinT_detectron2."""

from __future__ import annotations

import importlib
import shutil
import tempfile
import zipfile
import urllib.request
import sys
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLED_SWINT = REPO_ROOT / "third_party" / "SwinT_detectron2"
SWINT_PACKAGE = BUNDLED_SWINT / "swint"
SWINT_WEIGHTS_URL = "https://github.com/xiaohu2015/SwinT_detectron2/releases/download/v1.0/mask_rcnn_swint_T_coco17.pth"
SWINT_REPO_ZIP = "https://github.com/xiaohu2015/SwinT_detectron2/archive/refs/heads/master.zip"


def _download_swint_repo() -> None:
    """Download and unpack SwinT_detectron2 into the vendored third_party location."""
    BUNDLED_SWINT.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix="swint_repo_"))
    archive_path = tmp_dir / "swint_repo.zip"
    try:
        with urllib.request.urlopen(SWINT_REPO_ZIP) as resp, open(archive_path, "wb") as out:
            shutil.copyfileobj(resp, out)

        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(tmp_dir)

        extracted_dirs = list(tmp_dir.glob("SwinT_detectron2-*"))
        if not extracted_dirs:
            raise RuntimeError("Downloaded SwinT_detectron2 archive did not contain expected folder.")

        extracted = extracted_dirs[0]
        if BUNDLED_SWINT.exists():
            shutil.rmtree(BUNDLED_SWINT)
        shutil.move(str(extracted), str(BUNDLED_SWINT))
    except Exception as exc:  # noqa: BLE001
        raise FileNotFoundError(
            f"Vendored SwinT_detectron2 not found and automatic download from {SWINT_REPO_ZIP} failed. "
            "Download or clone the repository manually into third_party/SwinT_detectron2."
        ) from exc
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _get_swint_root() -> Path:
    """Load the vendored SwinT_detectron2 and return its root directory."""
    if not SWINT_PACKAGE.exists():
        _download_swint_repo()

    if str(BUNDLED_SWINT) not in sys.path:
        sys.path.insert(0, str(BUNDLED_SWINT))

    try:
        swint_module = importlib.import_module("swint")
    except ImportError:
        # Retry once by re-downloading if import fails
        _download_swint_repo()
        swint_module = importlib.import_module("swint")

    swint_root = Path(swint_module.__file__).resolve().parent.parent
    if not (swint_root / "configs").exists() or not (swint_root / "models").exists():
        # Attempt to re-download once if files are missing
        _download_swint_repo()
        swint_root = Path(importlib.import_module("swint").__file__).resolve().parent.parent
        if not (swint_root / "configs").exists() or not (swint_root / "models").exists():
            raise FileNotFoundError(
                f"Vendored SwinT_detectron2 at {BUNDLED_SWINT} is missing configs/ or models/. "
                "Automatic download was attempted and failed; clone the repository manually."
            )

    return swint_root

def _default_swint_paths() -> tuple[Path, Path]:
    """Return default config and weights paths, resolving the vendored root lazily."""
    swint_root = _get_swint_root()
    default_cfg = swint_root / "configs" / "SwinT" / "mask_rcnn_swint_T_FPN_3x.yaml"
    default_weights = swint_root / "models" / "mask_rcnn_swint_T_coco17.pth"
    return default_cfg, default_weights


def prepare_swint_config(
    cfg,
    config_path: Optional[str] = None,
) -> str:
    """Register Swin config nodes/backbone builders and return the config path to merge."""
    if config_path:
        cfg_path = Path(config_path)
        swint_root = cfg_path.resolve().parents[2]
        if str(swint_root) not in sys.path:
            sys.path.insert(0, str(swint_root))
    else:
        default_cfg, _ = _default_swint_paths()
        cfg_path = default_cfg
        swint_root = cfg_path.parent.parent
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Swin config not found at {cfg_path}. Verify your vendored SwinT_detectron2 or override config_path."
        )

    # Imports must happen after ensuring swint is importable
    from swint.config import add_swint_config  # type: ignore
    from swint import swin_transformer  # noqa: F401

    add_swint_config(cfg)
    return str(cfg_path)


def ensure_swint_weights(weights_path: Optional[str] = None) -> str:
    """
    Ensure SwinT weights are present locally. If no path is provided, download the
    default Swin-T Mask R-CNN weights into the vendored models directory.
    """
    if weights_path:
        target = Path(weights_path).expanduser()
        if target.exists():
            return str(target)
        raise FileNotFoundError(
            f"Swin weights not found at {target}. Provide a valid path or remove "
            "swint_weights_path to download the default weights automatically."
        )

    _, default_weights = _default_swint_paths()
    target = default_weights
    if target.exists():
        return str(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        with urllib.request.urlopen(SWINT_WEIGHTS_URL) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
        tmp.rename(target)
    except Exception as exc:  # noqa: BLE001
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"Failed to download Swin weights from {SWINT_WEIGHTS_URL}. "
            "Download manually and place in third_party/SwinT_detectron2/models/."
        ) from exc

    return str(target)
