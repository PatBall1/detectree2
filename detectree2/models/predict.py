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
    mode="rgb",
):
    """Make predictions on tiled data.

    Predicts crowns for all images present in a directory and outputs masks as JSON files.

    Args:
        directory (str): Directory containing images to predict on.
        out_folder (str): Folder to save predictions.
        predictor: The predictor object (e.g., DefaultPredictor).
        eval (bool): Whether to use evaluation mode.
        save (bool): Whether to save the predictions.
        num_predictions (int): Number of predictions to make (0 for all).
        mode (str): Image mode, 'rgb' or 'ms' (multispectral).

    """
    pred_dir = os.path.join(directory, out_folder)
    Path(pred_dir).mkdir(parents=True, exist_ok=True)

    if eval:
        dataset_dicts = get_tree_dicts(directory)
    else:
        dataset_dicts = get_filenames(directory, mode=mode)

    total_files = len(dataset_dicts)

    # Decide the number of items to predict on
    if num_predictions == 0:
        num_to_pred = len(dataset_dicts)
    else:
        num_to_pred = num_predictions

    print(f"Predicting {num_to_pred} files")

    for i, d in enumerate(dataset_dicts[:num_to_pred], start=1):
        if mode == "rgb":
            img = cv2.imread(d["file_name"])
            if img is None:
                print(f"Failed to read image {d['file_name']} with cv2.")
                continue
        elif mode == "ms":
            try:
                with rasterio.open(d["file_name"]) as src:
                    img = src.read()    # shape is (bands, H, W)
                    img = np.transpose(img, (1, 2, 0))    # shape (H, W, bands)
            except Exception as e:
                print(
                    f"Failed to read image {d['file_name']} with rasterio: {e}")
                continue
        else:
            print(f"Unknown mode '{mode}'.")
            continue

        outputs = predictor(img)

        # Create the output file name
        file_name_path = d["file_name"]
        file_name = os.path.basename(os.path.normpath(file_name_path))
        file_root, file_ext = os.path.splitext(file_name)
        file_name = file_root + ".json"
        output_file = os.path.join(pred_dir, f"Prediction_{file_name}")

        if save:
            # Convert predictions to JSON and save them
            evaluations = instances_to_coco_json(outputs["instances"].to("cpu"),
                                                 d["file_name"])
            with open(output_file, "w") as dest:
                json.dump(evaluations, dest)

        if i % 50 == 0:
            print(f"Predicted {i} files of {total_files}")


if __name__ == "__main__":
    predict_on_data()
