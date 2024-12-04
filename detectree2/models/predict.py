"""Generate predictions.

This module contains the code to generate predictions on tiled data.
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
import rasterio
from detectron2.engine import DefaultPredictor
from detectron2.evaluation.coco_evaluation import instances_to_coco_json

from detectree2.models.train import get_filenames, get_tree_dicts

# Code to convert RLE data from the output instances into Polygons,
# a small amout of info is lost but is fine.
# https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py <-- found here


def predict_on_data(
    directory: str = "./",
    out_folder: str = "predictions",
    predictor=DefaultPredictor,
    eval=False,
    save: bool = True,
    num_predictions=0,
):
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
    pred_dir = os.path.join(directory, out_folder)
    Path(pred_dir).mkdir(parents=True, exist_ok=True)

    if eval:
        dataset_dicts = get_tree_dicts(directory)
        if len(dataset_dicts) > 0:
            sample_file = dataset_dicts[0]["file_name"]
            _, mode = get_filenames(os.path.dirname(sample_file))
        else:
            mode = None
    else:
        dataset_dicts, mode = get_filenames(directory)

    total_files = len(dataset_dicts)
    num_to_pred = len(
        dataset_dicts) if num_predictions == 0 else num_predictions

    print(f"Predicting {num_to_pred} files in mode {mode}")

    for i, d in enumerate(dataset_dicts[:num_to_pred], start=1):
        file_name = d["file_name"]
        file_ext = os.path.splitext(file_name)[1].lower()
        if file_ext == ".png":
            # RGB image, read with cv2
            img = cv2.imread(file_name)
            if img is None:
                print(f"Failed to read image {file_name} with cv2.")
                continue
        elif file_ext == ".tif":
            # Multispectral image, read with rasterio
            with rasterio.open(file_name) as src:
                img = src.read()
                # Transpose to match expected format (H, W, C)
                img = np.transpose(img, (1, 2, 0))
        else:
            print(f"Unsupported file extension {file_ext} for file {file_name}")
            continue

        outputs = predictor(img)

        # Create the output file name
        file_name_only = os.path.basename(file_name)
        file_name_json = os.path.splitext(file_name_only)[0] + ".json"
        output_file = os.path.join(pred_dir, f"Prediction_{file_name_json}")

        if save:
            # Save predictions to JSON file
            evaluations = instances_to_coco_json(outputs["instances"].to("cpu"),
                                                 file_name)
            with open(output_file, "w") as dest:
                json.dump(evaluations, dest)

        if i % 50 == 0:
            print(f"Predicted {i} files of {total_files}")


if __name__ == "__main__":
    predict_on_data()
