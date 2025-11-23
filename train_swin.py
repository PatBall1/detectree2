"""Minimal Swin Transformer training entrypoint for Detectree2.

Hard-coded configuration: edit the constants below to point to your tiled data,
class mapping, and output directory. Data root should contain per-fold
subdirectories (e.g., ``fold1``, ``fold2``) with GeoJSON labels produced by the
tiling pipeline.
"""

from pathlib import Path

from detectree2.models.backbones_swin import DEFAULT_SWINT_CONFIG, DEFAULT_SWINT_WEIGHTS
from detectree2.models.train import MyTrainer, register_train_data, setup_cfg

REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path("/path/to/your/tiles")  # <-- update to your tiled dataset root
DATASET_NAME = "Trees"
VAL_FOLD = 1  # 1-based index of the fold to hold out for validation
OUT_DIR = REPO_ROOT / "train_outputs" / "swin_run"
MAX_ITER = 3000
BATCH_SIZE = 2
NUM_WORKERS = 2
LR = 3e-4
EVAL_PERIOD = 200
PATIENCE = 5
IMGMODE = "rgb"  # "rgb" or "ms"
NUM_BANDS = 3
CLASS_MAPPING = None  # Optional path to class-to-index mapping (json or pickle)
SWINT_CONFIG = str(DEFAULT_SWINT_CONFIG)
SWINT_WEIGHTS = str(DEFAULT_SWINT_WEIGHTS)
DEVICE = "cuda"
RESUME = False


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Register train/val folds.
    register_train_data(DATA_ROOT,
                        DATASET_NAME,
                        val_fold=VAL_FOLD,
                        class_mapping_file=CLASS_MAPPING)
    train_name = f"{DATASET_NAME}_train"
    val_name = f"{DATASET_NAME}_val"

    cfg = setup_cfg(
        base_model=SWINT_CONFIG,
        trains=(train_name, ),
        tests=(val_name, ),
        workers=NUM_WORKERS,
        ims_per_batch=BATCH_SIZE,
        base_lr=LR,
        max_iter=MAX_ITER,
        eval_period=EVAL_PERIOD,
        out_dir=str(OUT_DIR),
        resize="fixed",
        imgmode=IMGMODE,
        num_bands=NUM_BANDS,
        class_mapping_file=CLASS_MAPPING,
        use_swint_backbone=True,
        swint_config_path=SWINT_CONFIG,
        swint_weights_path=SWINT_WEIGHTS,
    )
    cfg.MODEL.DEVICE = DEVICE

    trainer = MyTrainer(cfg, patience=PATIENCE)
    trainer.resume_or_load(resume=RESUME)
    trainer.train()


if __name__ == "__main__":
    main()
