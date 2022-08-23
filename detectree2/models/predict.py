import json
import os
import random
from http.client import REQUEST_URI_TOO_LONG
from pathlib import Path

import cv2
import geopandas as gpd
from detectron2.engine import DefaultPredictor
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from fiona.crs import from_epsg
from shapely.geometry import box, shape

from detectree2.models.train import get_filenames
from detectron2.data.build import DatasetMapper
import time
from PIL import Image
import pycocotools.mask as mask_util

# Code to convert RLE data from the output instances into Polygons, a small about of info is lost but is fine.
# https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py <-- found here

def reproject_to_geojson_spatially(data,
                                   output_fold=None,
                                   pred_fold=None,
                                   EPSG="32650"):    # noqa:N803
    """Reprojects the coordinates back so the crowns can be overlaid with the original tif file of the entire region.
    Takes a json and changes it to a geojson so it can overlay with crowns.
    Another copy is produced to overlay with PNGs.
    """

    Path(output_fold).mkdir(parents=True, exist_ok=True)
    entries = os.listdir(pred_fold)
    #print(entries)
    # scale to deal with the resolution
    scalingx = data.transform[0]
    scalingy = -data.transform[4]

    for file in entries:
        if ".json" in file:
            # create a geofile for each tile --> the EPSG value might need to be changed.
            geofile = {
                "type": "FeatureCollection",
                "crs": {
                    "type": "name",
                    "properties": {
                        "name": "urn:ogc:def:crs:EPSG::" + EPSG
                    }
                },
                "features": []
            }

            # create a dictionary for each file to store data used multiple times
            img_dict = {}
            img_dict["filename"] = file
            #print(img_dict["filename"])

            file_mins = file.replace(".json", "")
            file_mins_split = file_mins.split("_")
            minx = int(file_mins_split[-4])
            miny = int(file_mins_split[-3])
            tile_height = int(file_mins_split[-2])
            buffer = int(file_mins_split[-1])
            height = (tile_height + 2 * buffer) / scalingx

            # update the image dictionary to store all information cleanly
            img_dict.update({
                "minx": minx,
                "miny": miny,
                "height": height,
                "buffer": buffer
            })
            # print("Img dict:", img_dict)

            # load the json file we need to convert into a geojson
            with open(pred_fold + img_dict["filename"]) as prediction_file:
                datajson = json.load(prediction_file)
            # print("data_json:",datajson)

            # json file is formated as a list of segmentation polygons so cycle through each one
            for crown_data in datajson:
                # just a check that the crown image is correct
                if str(minx) + '_' + str(miny) in crown_data["image_id"]:
                    crown = crown_data["segmentation"]
                    confidence_score = crown_data['score']

                    # changing the coords from RLE format so can be read as numbers, here the numbers are
                    # integers so a bit of info on position is lost
                    mask_of_coords = mask_util.decode(crown)
                    crown_coords = polygon_from_mask(mask_of_coords)
                    moved_coords = []

                    # coords from json are in a list of [x1, y1, x2, y2,... ] so convert them to [[x1, y1], ...]
                    # format and at the same time rescale them so they are in the correct position for QGIS
                    for c in range(0, len(crown_coords), 2):
                        x_coord = crown_coords[c]
                        y_coord = crown_coords[c + 1]

                        # print("ycoord:", y_coord)
                        # print("height:", height)

                        # rescaling the coords depending on where the tile is in the original image, note the
                        # correction factors have been manually added as outputs did not line up with predictions
                        # from training script
                        if minx == int(data.bounds[0]) and miny == int(data.bounds[1]):
                            #print("Bottom Corner")
                            x_coord = (x_coord) * scalingx + minx
                            y_coord = (height - y_coord) * scalingy + miny
                        elif minx == int(data.bounds[0]):
                            #print("Left Edge")
                            x_coord = (x_coord) * scalingx + minx
                            y_coord = (height
                                       - y_coord) * scalingy - buffer + miny
                        elif miny == int(data.bounds[1]):
                            #print("Bottom Edge")
                            x_coord = (x_coord) * scalingx - buffer + minx
                            y_coord = (height
                                       - y_coord) * scalingy - buffer + miny
                        else:
                            # print("Anywhere else")
                            x_coord = (x_coord) * scalingx - buffer + minx
                            y_coord = (height
                                       - y_coord) * scalingy - buffer + miny

                        moved_coords.append([x_coord, y_coord])

                    geofile["features"].append({
                        "type": "Feature",
                        "properties": {
                            "Confidence score": confidence_score
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [moved_coords]
                        }
                    })

            # Check final form is correct - compare to a known geojson file if error appears.
            # print("geofile",geofile)

            output_geo_file = output_fold + img_dict["filename"].replace(
                '.json', "_" + EPSG + '_lidar.geojson')
            # print("output location:", output_geo_file)
            with open(output_geo_file, "w") as dest:
                json.dump(geofile, dest)

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

class MyPredictor(DefaultPredictor):
  def __init__(self, cfg, mode):
    self.cfg = cfg.clone()  # cfg can be modified by model
    self.model = build_model(self.cfg)
    self.model.eval()
    self.mode = mode
    cfg.DATASETS.TEST = ('pigs',)
    if len(cfg.DATASETS.TEST):
      self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

    checkpointer = DetectionCheckpointer(self.model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    self.aug = self.augmentation()

    self.input_format = cfg.INPUT.FORMAT
    assert self.input_format in ["RGB", "BGR"], self.input_format

  def __call__(self, original_image):
    """
    Args:
      original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

    Returns:
      predictions (dict):
        the output of the model for one image only.
    See :doc:`/tutorials/models` for details about the format.
        """
    with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        # Apply pre-processing to image.
        if self.input_format == "RGB":
            # whether the model expects BGR inputs or RGB
            original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        if self.aug != None:
          image = self.aug.get_transform(original_image).apply_image(original_image)
        else:
          image = original_image
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}
        predictions = self.model([inputs])[0]
    

  def _predict(self, IN_DIR, save = True):
    dataset_dicts = []
    files = glob.glob(IN_DIR + "*.png")
    for filename in [file for file in files]:
      file = {}
      filename = os.path.join(IN_DIR, filename)
      file["file_name"] = filename
      dataset_dicts.append(file)
  
    # Works out if all items in folder should be predicted on

    num_to_pred = len(dataset_dicts)

    pred_dir = IN_DIR + "predictions"

    Path(pred_dir).mkdir(parents=True, exist_ok=True)

    for data in random.sample(dataset_dicts,num_to_pred):
      with torch.no_grad():
        print(data["file_name"])
        img = cv2.imread(data["file_name"])
        if self.input_format == "RGB":
          # whether the model expects BGR inputs or RGB
          img = img[:, :, ::-1]
        height, width = img.shape[:2]
        if self.aug != None:
          image = self.aug.get_transform(img).apply_image(img)
        else:
          image = img
        image = img
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}
        predictions = self.model([inputs])[0]
      file_name_path = data["file_name"]
      file_name = os.path.basename(os.path.normpath(file_name_path))  #Strips off all slashes so just final file name left
      file_name = file_name.replace("png","json")
    
      output_file = pred_dir + "/predictions_" + file_name

      if save: 
        ## Converting the predictions to json files and saving them in the specfied output file.
        evaluations= instances_to_coco_json(predictions["instances"].to("cpu"),data["file_name"])
        with open(output_file, "w") as dest:
          json.dump(evaluations,dest)


  def predict(self, save=True):

    for i in range(len(self.cfg.IN_DIR)):

      self._predict(self.cfg.IN_DIR[i])
      files = glob.glob(self.cfg.IN_DIR[i] + "*.tif")
      data = rasterio.open(files[0])
      reproject_to_geojson_spatially(data, self.cfg.OUT_DIR[-1], self.cfg.IN_DIR[i] + "predictions/", EPSG = "32650")

    folder = self.cfg.OUT_DIR[-1]

    crowns = stitch_crowns(folder, 1)

    crowns = clean_crowns(crowns)

    x = crowns.buffer(0.0001)
    tolerance = 0.03
    simplified = x.simplify(tolerance, preserve_topology=True)

    crowns.to_file(folder + "crowns_out.gpkg")


  def augmentation(self):
    if self.mode == 'resize_fixed':
      return T.ResizeShortestEdge(
        [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST)
    if self.mode == 'No_resize':
      return None
    else:
      print('No such a mode')
      return T.ResizeShortestEdge(
        [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST)


if __name__ == "__main__":
    print("something")
