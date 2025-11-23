"""End-to-end CLI: tile, predict, project, stitch, clean, and write a GPKG."""

import argparse
from pathlib import Path
from typing import Tuple

import geopandas as gpd
from detectron2.engine import DefaultPredictor

from detectree2.models.outputs import clean_crowns, project_to_geojson, stitch_crowns
from detectree2.models.predict import predict_on_data
from detectree2.models.train import setup_cfg
from detectree2.preprocessing.tiling import tile_data

REPO_ROOT = Path(__file__).resolve().parent

# Hardcoded paths and parameters for the full workflow.
ORTHOMOSAIC = REPO_ROOT / "tmp_infer" / "PLOT20-150M.tif"
OUTPUT_ROOT = REPO_ROOT / "tmp_infer" / "inference_output"
TILE_WIDTH = 40
TILE_HEIGHT = 40
BUFFER = 30
SCORE_THRESH = 0.3
MAX_PREDS = 0  # 0 = all tiles
CLEAN_IOU = 0.6
CLEAN_CONFIDENCE = 0.5
EDGE_SHIFT = 1
SIMPLIFY_TOLERANCE = 0.3
DEVICE = "cpu"

DEFAULT_SWINT_CONFIG = (
    REPO_ROOT
    / "third_party"
    / "SwinT_detectron2"
    / "configs"
    / "SwinT"
    / "mask_rcnn_swint_T_FPN_3x.yaml"
)
DEFAULT_SWINT_WEIGHTS = (
    REPO_ROOT
    / "third_party"
    / "SwinT_detectron2"
    / "models"
    / "mask_rcnn_swint_T_coco17.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detectree2 landscape prediction pipeline (tiles -> crowns.gpkg)."
    )
    parser.add_argument(
        "--no-swint-backbone",
        dest="use_swint_backbone",
        action="store_false",
        help="Disable Swin Transformer backbone config (defaults to enabled).",
    )
    parser.set_defaults(use_swint_backbone=True)
    return parser.parse_args()


def build_cfg(use_swint_backbone: bool):
    cfg = setup_cfg(
        base_model=str(DEFAULT_SWINT_CONFIG),
        trains=(),
        tests=(),
        update_model=str(DEFAULT_SWINT_WEIGHTS),
        use_swint_backbone=use_swint_backbone,
        swint_config_path=str(DEFAULT_SWINT_CONFIG),
        swint_weights_path=str(DEFAULT_SWINT_WEIGHTS),
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH
    cfg.MODEL.DEVICE = DEVICE
    return cfg


def tile_if_needed() -> Path:
    tiles_dir = OUTPUT_ROOT / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    has_tiles = any(tiles_dir.glob("*.tif")) or any(tiles_dir.glob("*.png"))
    if has_tiles:
        print(f"Found existing tiles in {tiles_dir}, skipping tiling.")
        return tiles_dir

    print(
        f"Tiling {ORTHOMOSAIC} -> {tiles_dir} "
        f"(buffer {BUFFER}, size {TILE_WIDTH}x{TILE_HEIGHT})."
    )
    tile_data(
        img_path=str(ORTHOMOSAIC),
        out_dir=str(tiles_dir),
        buffer=BUFFER,
        tile_width=TILE_WIDTH,
        tile_height=TILE_HEIGHT,
        mode="rgb",
    )
    return tiles_dir


def predict_tiles(predictor: DefaultPredictor, tiles_dir: Path) -> Tuple[Path, Path]:
    pred_folder = "predictions"
    predict_on_data(
        directory=str(tiles_dir),
        out_folder=pred_folder,
        predictor=predictor,
        save=True,
        num_predictions=MAX_PREDS,
    )
    preds_dir = tiles_dir / pred_folder
    geo_dir = OUTPUT_ROOT / "predictions_geo"
    project_to_geojson(str(tiles_dir), str(preds_dir), str(geo_dir))
    return preds_dir, geo_dir


def stitch_and_clean(geo_dir: Path) -> gpd.GeoDataFrame:
    crowns = stitch_crowns(str(geo_dir), shift=EDGE_SHIFT)
    cleaned = clean_crowns(
        crowns,
        iou_threshold=CLEAN_IOU,
        confidence=CLEAN_CONFIDENCE,
    )
    if SIMPLIFY_TOLERANCE > 0:
        cleaned = cleaned.set_geometry(cleaned.geometry.simplify(SIMPLIFY_TOLERANCE))
    return cleaned


def main() -> None:
    args = parse_args()
    if not ORTHOMOSAIC.is_file():
        raise FileNotFoundError(f"Missing orthomosaic at {ORTHOMOSAIC}")
    if not DEFAULT_SWINT_WEIGHTS.is_file():
        raise FileNotFoundError(f"Missing weights at {DEFAULT_SWINT_WEIGHTS}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    tiles_dir = tile_if_needed()

    cfg = build_cfg(args.use_swint_backbone)
    predictor = DefaultPredictor(cfg)
    _, geo_dir = predict_tiles(predictor, tiles_dir)

    cleaned = stitch_and_clean(geo_dir)

    out_gpkg = OUTPUT_ROOT / "crowns_out.gpkg"
    cleaned.to_file(out_gpkg, driver="GPKG")
    print(f"Wrote {len(cleaned)} crowns to {out_gpkg}")


if __name__ == "__main__":
    main()
