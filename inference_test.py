"""Hardcoded inference pipeline using the Swin backbone."""

from pathlib import Path

from detectron2.engine import DefaultPredictor
from detectree2.models.outputs import clean_crowns, project_to_geojson, stitch_crowns
from detectree2.models.predict import predict_on_data
from detectree2.models.train import setup_cfg
from detectree2.preprocessing.tiling import tile_data

REPO_ROOT = Path(__file__).resolve().parent
ORTHOMOSAIC = REPO_ROOT / "tmp_infer" / "PLOT20-150M.tif"
OUT_ROOT = REPO_ROOT / "tmp_infer" / "inference_output"
TILE_WIDTH = 40
TILE_HEIGHT = 40
BUFFER = 30
SCORE_THRESH = 0.3
CLEAN_IOU = 0.6
CLEAN_CONFIDENCE = 0.5
EDGE_SHIFT = 1.0
SIMPLIFY_TOLERANCE = 0.3
DEVICE = "cpu"  # set to "cuda" if available
# Point to custom Swin weights (set to None to auto-download default)
CUSTOM_SWINT_WEIGHTS = str(Path.home() / "Downloads" / "model_19.pth")


def main() -> None:
    if not ORTHOMOSAIC.is_file():
        raise FileNotFoundError(f"Missing orthomosaic: {ORTHOMOSAIC}")

    tiles_dir = OUT_ROOT / "tiles"
    preds_dir = tiles_dir / "predictions"
    geo_dir = OUT_ROOT / "predictions_geo"
    out_gpkg = OUT_ROOT / "crowns_out.gpkg"

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tiles_dir.mkdir(parents=True, exist_ok=True)

    tile_data(
        img_path=str(ORTHOMOSAIC),
        out_dir=str(tiles_dir),
        buffer=BUFFER,
        tile_width=TILE_WIDTH,
        tile_height=TILE_HEIGHT,
        mode="rgb",
    )

    cfg = setup_cfg(
        trains=(),
        tests=(),
        use_swint_backbone=True,
        swint_config_path=None,
        swint_weights_path=CUSTOM_SWINT_WEIGHTS,
    )
    cfg.MODEL.DEVICE = DEVICE
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH

    predictor = DefaultPredictor(cfg)
    predict_on_data(
        directory=str(tiles_dir),
        out_folder="predictions",
        predictor=predictor,
        save=True,
        num_predictions=0,
    )

    project_to_geojson(str(tiles_dir), str(preds_dir), str(geo_dir))
    crowns = stitch_crowns(str(geo_dir), shift=EDGE_SHIFT)
    cleaned = clean_crowns(
        crowns,
        iou_threshold=CLEAN_IOU,
        confidence=CLEAN_CONFIDENCE,
    )
    if SIMPLIFY_TOLERANCE > 0:
        cleaned = cleaned.set_geometry(cleaned.geometry.simplify(SIMPLIFY_TOLERANCE))

    cleaned.to_file(out_gpkg, driver="GPKG")
    print(f"Wrote {len(cleaned)} crowns to {out_gpkg}")


if __name__ == "__main__":
    main()
