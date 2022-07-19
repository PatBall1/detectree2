# necessary basic libraries
import pandas as pd
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import json
import png
import glob

# geospatial libraries
import rasterio
import geopandas
from geopandas.tools import sjoin
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


if __name__ == "__main__":
  print("something")