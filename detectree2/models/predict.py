import json
import os
import random
from pathlib import Path

import cv2
import geopandas as gpd
from detectron2.engine import DefaultPredictor
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from fiona.crs import from_epsg

from detectree2.models.train import get_filenames


def predict_on_data(
    directory: str,
    predictor=DefaultPredictor,
    save: bool = True,
):
    """Make predictions on tiled data.

    Predicts crowns for all png images present in a directory and outputs masks
    as jsons
    """

    pred_dir = os.path.join(directory, "predictions")

    Path(pred_dir).mkdir(parents=True, exist_ok=True)

    dataset_dicts = get_filenames(directory)

    # Works out if all items in folder should be predicted on

    num_to_pred = len(dataset_dicts)

    for d in random.sample(dataset_dicts, num_to_pred):
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)

        # Creating the file name of the output file
        file_name_path = d["file_name"]
        # Strips off all slashes so just final file name left
        file_name = os.path.basename(os.path.normpath(file_name_path))
        file_name = file_name.replace("png", "json")
        output_file = os.path.join(pred_dir, f"Prediction_{file_name}")
        print(output_file)

        if save:
            # Converting the predictions to json files and saving them in the specfied output file.
            evaluations = instances_to_coco_json(outputs["instances"].to("cpu"), d["file_name"])
            with open(output_file, "w") as dest:
                json.dump(evaluations, dest)


def stitch_crowns(folder: str, shift: int = 1):
    """Stitch together predicted crowns."""
    folder = Path(folder)
    files = folder.glob("*geojson")
    crowns = gpd.GeoDataFrame(columns=["Confidence score", "geometry"], geometry="geometry", crs=from_epsg(32622))
    for file in files:
        crowns_tile = gpd.read_file(file)
        crowns_tile.crs = "epsg:32622"
        # crowns_tile = crowns_tile.set_crs(from_epsg(32622))
        # print(crowns_tile)

        geo = box_make(file, shift)
        # geo.plot()
        crowns_tile = gpd.sjoin(crowns_tile, geo, "inner", "within")
        # print(crowns_tile)
        crowns = crowns.append(crowns_tile)
        # print(crowns)
    return crowns


if __name__ == "__main__":
    print("something")
