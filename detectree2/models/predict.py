import cv2
import random
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import json
import png
import glob
import rasterio
import geopandas
from geopandas.tools import sjoin
from pathlib import Path
import fiona
from rasterio.plot import show
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd
from fiona.crs import from_epsg
import pycrs
import descartes
import rasterio
from rasterio.transform import from_origin
import rasterio.features
import pycocotools.mask as mask_util
import fiona
from shapely.geometry import shape, mapping, box
from shapely.geometry.multipolygon import MultiPolygon
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectree2.models.train import get_tree_dicts, get_filenames


# Code to convert RLE data from the output instances into Polygons, a small about of info is lost but is fine.
# https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py <-- found here

def polygonFromMask(maskedArr):
    """
    Turn mask into polygons
    """
    contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    segmentation = []
    for contour in contours:
        # Valid polygons have >= 6 coordinates (3 points)
        if contour.size >= 6:
            segmentation.append(contour.flatten().tolist())
    RLEs = mask_util.frPyObjects(segmentation, maskedArr.shape[0], maskedArr.shape[1])
    RLE = mask_util.merge(RLEs)
    # RLE = mask.encode(np.asfortranarray(maskedArr))
    area = mask_util.area(RLE)
    [x, y, w, h] = cv2.boundingRect(maskedArr)

    return segmentation[0] #, [x, y, w, h], area


def predict_on_data(
  directory: str = None,
  predictor = DefaultPredictor,
  trees_metadata = None,
  save: bool = True,
  scale = 1,
  num_predictions = 0
  ):
  """Prediction produced from a folder of tiled data
  """

  pred_dir = directory + "predictions"

  Path(pred_dir).mkdir(parents=True, exist_ok=True)


  dataset_dicts = get_filenames(directory)
  
  # Works out if all items in folder should be predicted on

  num_to_pred = len(dataset_dicts)

  for d in random.sample(dataset_dicts,num_to_pred):
    img = cv2.imread(d["file_name"])
    # cv2_imshow(img)
    outputs = predictor(img)
    #v = Visualizer(img[:, :, ::-1], metadata=trees_metadata, scale=scale, instance_mode=ColorMode.SEGMENTATION)   # remove the colors of unsegmented pixels
    #v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #image = cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
    #display(Image.fromarray(image))

    ### Creating the file name of the output file
    file_name_path = d["file_name"]
    file_name = os.path.basename(os.path.normpath(file_name_path))  #Strips off all slashes so just final file name left
    file_name = file_name.replace("png","json")
    
    output_file = pred_dir + "/Prediction_" + file_name
    print(output_file)

    if save: 
      ## Converting the predictions to json files and saving them in the specfied output file.
      evaluations= instances_to_coco_json(outputs["instances"].to("cpu"),d["file_name"])
      with open(output_file, "w") as dest:
        json.dump(evaluations,dest)



if __name__ == "__main__":
  print("something")