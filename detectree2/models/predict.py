from http.client import REQUEST_URI_TOO_LONG
import json
import os
import random
from pathlib import Path

import cv2
import geopandas as gpd
from detectron2.engine import DefaultPredictor
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from fiona.crs import from_epsg
from shapely.geometry import box, shape

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
    # RLEs = mask_util.frPyObjects(segmentation, maskedArr.shape[0],
    #                             maskedArr.shape[1])
    #RLE = mask_util.merge(RLEs)
    # RLE = mask.encode(np.asfortranarray(maskedArr))
    # area = mask_util.area(RLE)
    # [x, y, w, h] = cv2.boundingRect(maskedArr)

    return segmentation[0]  # , [x, y, w, h], area


def predict_on_data(
    directory: str = "./",
    predictor=DefaultPredictor,
    save: bool = True,
):
    """Make predictions on tiled data

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
            # Converting the predictions to json files and saving them in the
            # specfied output file.
            evaluations = instances_to_coco_json(outputs["instances"].to("cpu"),
                                                 d["file_name"])
            with open(output_file, "w") as dest:
                json.dump(evaluations, dest)


def filename_geoinfo(filename):
    """Return geographic info of a tile from its filename
    """
    parts = os.path.basename(filename).split("_")

    parts = [int(part) for part in parts[-6:-1]]  # type: ignore
    minx = parts[0]
    miny = parts[1]
    width = parts[2]
    buffer = parts[3]
    crs = parts[4]
    return (minx, miny, width, buffer, crs)


def box_filter(filename, shift: int = 0):
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
    crowns_path = Path(folder)
    files = crowns_path.glob("*geojson")
    _, _, _, _, crs = filename_geoinfo(list(files)[0])
    files = crowns_path.glob("*geojson")
    crowns = gpd.GeoDataFrame(columns=["Confidence score", "geometry"],
                              geometry="geometry",
                              crs=from_epsg(crs))    # initiate an empty gpd.GDF
    for file in files:
        crowns_tile = gpd.read_file(file)
        #crowns_tile.crs = "epsg:32622"
        #crowns_tile = crowns_tile.set_crs(from_epsg(32622))
        # print(crowns_tile)

        geo = box_filter(file, shift)
        # geo.plot()
        crowns_tile = gpd.sjoin(crowns_tile, geo, "inner", "within")
        # print(crowns_tile)
        crowns = crowns.append(crowns_tile)
        # print(crowns)
    return crowns


def calc_iou(shape1, shape2):
    """Calculate the IoU of two shapes
    """
    iou = shape1.intersection(shape2).area / shape1.union(shape2).area
    return iou

def clean_crowns(crowns: gpd.GeoDataFrame):
    """Clean overlapping crowns
  
    Outputs can contain highly overlapping crowns including in the buffer region.
    This function removes crowns with a high degree of overlap with others but a 
    lower Confidence Score.
    """
    crowns_out = gpd.GeoDataFrame()
    for index, row in crowns.iterrows():  #iterate over each crown
        if index % 1000 == 0:
            print(str(index) + " / " + str(len(crowns)) + " cleaned") 
        if crowns.intersects(shape(row.geometry)).sum() == 1: # if there is not a crown interesects with the row (other than itself)
            crowns_out = crowns_out.append(row) # retain it
        else:
            intersecting = crowns.loc[crowns.intersects(shape(row.geometry))] # Find those crowns that intersect with it
            intersecting = intersecting.reset_index().drop("index", axis=1)
            iou = []
            for index1, row1 in intersecting.iterrows(): # iterate over those intersecting crowns
                #print(row1.geometry)
                iou.append(calc_iou(row.geometry, row1.geometry)) # Calculate the IoU with each of those crowns
            #print(iou)
            intersecting['iou'] = iou
            matches = intersecting[intersecting['iou'] > 0.75]  # Remove those crowns with a poor match
            matches = matches.sort_values('Confidence score', ascending=False).reset_index().drop('index', axis=1)
            match = matches.loc[[0]]  # Of the remaining crowns select the crown with the highest confidence
            if match['iou'][0] < 1:   # If the most confident is not the initial crown
                continue
            else:
                match = match.drop('iou', axis=1)
                #print(index)
                crowns_out = crowns_out.append(match)
    return crowns_out.reset_index()


if __name__ == "__main__":
    print("something")
