"""Generate predictions.

This module contains the code to generate predictions on tiled data.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio
from detectron2.engine import DefaultPredictor
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from tqdm import tqdm

from detectree2.models.train import get_filenames, get_tree_dicts

# Code to convert RLE data from the output instances into Polygons,
# a small amout of info is lost but is fine.
# https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py <-- found here


def predict_on_data(
    directory: str | Path = "./",
    out_folder: str | Path = "predictions",
    predictor=DefaultPredictor,
    eval: bool=False,
    save: bool = True,
    num_predictions=0,
) -> None:
    """Make predictions on tiled data.

    Predicts crowns for all images (.png or .tif) present in a directory and outputs masks as JSON files.

    Args:
        directory (str): Directory containing the images.
        out_folder (str): Output folder for predictions.
        predictor (DefaultPredictor): The predictor object.
        eval (bool): Whether to use evaluation mode.
        save (bool): Whether to save the predictions.
        num_predictions (int): Number of predictions to make.

    Returns:
        None
    """
    directory = Path(directory)
    pred_dir = Path(out_folder)
    pred_dir.mkdir(parents=True, exist_ok=True)

    if eval:
        dataset_dicts = get_tree_dicts(directory)
        if dataset_dicts:
            sample_file = Path(dataset_dicts[0]["file_name"])
            _, mode = get_filenames(sample_file.parent)
        else:
            mode = None
    else:
        dataset_dicts, mode = get_filenames(directory)

    num_to_pred = len(dataset_dicts) if num_predictions == 0 else num_predictions

    for d in tqdm(
        dataset_dicts[:num_to_pred],
        desc=f"Predicting files in mode {mode}",
    ):
        file_name = Path(d["file_name"])
        file_ext = file_name.suffix.lower()

        if file_ext == ".png":
            # RGB image, read with cv2
            img = cv2.imread(str(file_name))
            if img is None:
                print(f"Failed to read image {file_name} with cv2.")
                continue
            img = np.array(img)  # Explicitly convert to numpy array
        elif file_ext == ".tif":
            # Multispectral image, read with rasterio
            with rasterio.open(file_name) as src:
                img = src.read()
                # Transpose to match expected format (H, W, C)
                img = img.transpose(1, 2, 0)
        else:
            print(f"Unsupported file extension {file_ext} for file {file_name}")
            continue

        outputs = predictor(img)

        if save:
            # Save predictions to JSON file
            output_file = pred_dir / f"Prediction_{file_name.stem}.json"
            evaluations = instances_to_coco_json(outputs["instances"].to("cpu"),
                                                 str(file_name))
            with output_file.open("w") as f:
                json.dump(evaluations, f)


if __name__ == "__main__":
    predict_on_data()
