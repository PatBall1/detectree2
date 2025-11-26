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


def _get_swint_root() -> Path:
    """Load the vendored SwinT_detectron2 and return its root directory."""
    if not SWINT_PACKAGE.exists():
        _download_swint_repo()

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
DEFAULT_SWINT_WEIGHTS = SWINT_ROOT / "models" / "mask_rcnn_swint_T_coco17.pth"


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


def ensure_swint_weights(weights_path: Optional[str] = None) -> str:
    """
    Ensure SwinT weights are present locally. If no path is provided, download the
    default Swin-T Mask R-CNN weights into the vendored models directory.
    """
    target = Path(weights_path).expanduser() if weights_path else DEFAULT_SWINT_WEIGHTS
    if target.exists():
        return str(target)

    if weights_path:
        raise FileNotFoundError(
            f"Swin weights not found at {target}. Provide a valid path or remove "
            "swint_weights_path to download the default weights automatically."
        )

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
