import cv2
import random
import os
import json
import geopandas as gpd
import pycocotools.mask as mask_util
from pathlib import Path
from shapely.geometry import box
from fiona.crs import from_epsg
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.engine import DefaultPredictor
from detectree2.models.train import get_filenames

# Code to convert RLE data from the output instances into Polygons, a small about of info is lost but is fine.
# https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py <-- found here


def polygonFromMask(maskedArr):
    """
    Turn mask into polygons
    """
    contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    segmentation = []
    for contour in contours:
        # Valid polygons have >= 6 coordinates (3 points)
        if contour.size >= 6:
            segmentation.append(contour.flatten().tolist())
    RLEs = mask_util.frPyObjects(segmentation, maskedArr.shape[0],
                                 maskedArr.shape[1])
    RLE = mask_util.merge(RLEs)
    # RLE = mask.encode(np.asfortranarray(maskedArr))
    # area = mask_util.area(RLE)
    # [x, y, w, h] = cv2.boundingRect(maskedArr)

    return segmentation[0]    #, [x, y, w, h], area


def predict_on_data(
    directory: str = None,
    predictor=DefaultPredictor,
    save: bool = True,
):
    """Make predictions on tiled data

    Predicts crowns for all png images present in a directory and outputs masks 
    as jsons
    """

    pred_dir = directory + "predictions"

    Path(pred_dir).mkdir(parents=True, exist_ok=True)

    dataset_dicts = get_filenames(directory)

    # Works out if all items in folder should be predicted on

    num_to_pred = len(dataset_dicts)

    for d in random.sample(dataset_dicts, num_to_pred):
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)

        ### Creating the file name of the output file
        file_name_path = d["file_name"]
        file_name = os.path.basename(
            os.path.normpath(file_name_path)
        )    #Strips off all slashes so just final file name left
        file_name = file_name.replace("png", "json")

        output_file = pred_dir + "/Prediction_" + file_name
        print(output_file)

        if save:
            # Converting the predictions to json files and saving them in the 
            # specfied output file.
            evaluations = instances_to_coco_json(outputs["instances"].to("cpu"),
                                                 d["file_name"])
            with open(output_file, "w") as dest:
                json.dump(evaluations, dest)


def filename_geoinfo(filename: str):
    """Return geographic info of a tile from its filename
    """
    parts = os.path.basename(filename).split("_")

    parts = [int(part) for part in parts[-6:-1]]
    minx = parts[0]
    miny = parts[1]
    width = parts[2]
    buffer = parts[3]
    crs = parts[4]
    return (minx, miny, width, buffer, crs)


def box_filter(filename: str, shift: int = 0):
    """Create a bounding box from a file name to filter edge crowns
    """
    minx, miny, width, buffer, crs = filename_geoinfo(filename)
    bounding_box = box_make(minx, miny, width, buffer, crs, shift)
    return bounding_box


def box_make(minx: int,
             miny: int,
             width: int,
             buffer: int,
             crs,
             shift: int = 0):
    """Generate bounding box from geographic specifications
    """
    bbox = box(
        minx - buffer + shift,
        miny - buffer + shift,
        minx + width + buffer - shift,
        miny + width + buffer - shift,
    )
    geo = gpd.GeoDataFrame({"geometry": bbox}, index=[0], crs=from_epsg(crs))
    return geo


def stitch_crowns(folder: str, shift: int = 1):
    """Stitch together predicted crowns
  """
    folder = Path(folder)
    files = folder.glob("*geojson")
    _, _, _, _, crs = filename_geoinfo(list(files)[0])
    files = folder.glob("*geojson")
    crowns = gpd.GeoDataFrame(columns=["Confidence score", "geometry"],
                              geometry="geometry",
                              crs=from_epsg(crs))    # initiate an empty gpd.GDF
    for file in files:
        crowns_tile = gpd.read_file(file)
        #crowns_tile.crs = "epsg:32622"
        #crowns_tile = crowns_tile.set_crs(from_epsg(32622))
        #print(crowns_tile)

        geo = box_filter(file, shift)
        #geo.plot()
        crowns_tile = gpd.sjoin(crowns_tile, geo, "inner", "within")
        #print(crowns_tile)
        crowns = crowns.append(crowns_tile)
        #print(crowns)
    return crowns


if __name__ == "__main__":
    print("something")
