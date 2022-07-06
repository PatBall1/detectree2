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

# import more geospatial libraries
import rasterio
from rasterio.transform import from_origin
import rasterio.features
import pycocotools.mask as mask_util

import fiona

from shapely.geometry import shape, mapping, box
from shapely.geometry.multipolygon import MultiPolygon




def polygonFromMask(maskedArr):
    """
    Code to convert RLE data from the output instances into Polygons, a small about of info is lost but is fine.
    https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py <-- found here
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



def reproject_to_geojson(directory = None):
    """
    Takes a json and changes it to a geojson so it can overlay tiffs in GIS
    """

    entries = os.listdir(directory)

    for file in entries:
        if ".json" in file: 
            #create a geofile for each tile --> the EPSG value might need to be changed.
            geofile = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::26917"}}, "features":[]}

            # create a dictionary for each file to store data used multiple times
            img_dict = {}
            img_dict["filename"] = file

            file_mins = file.replace(".json", "")
            file_mins_split = file_mins.split("_")
            img_dict["minx"]= file_mins_split[-4]
            img_dict["miny"]= file_mins_split[-3]

            # load the json file we need to convert into a geojson
            with open(directory+img_dict["filename"]) as prediction_file:
                datajson = json.load(prediction_file)
            #print(datajson)
        
            img_dict["width"] = datajson[0]["segmentation"]["size"][0]
            img_dict["height"] = datajson[0]["segmentation"]["size"][1]
            print(img_dict)

            # json file is formated as a list of segmentation polygons so cycle through each one
            for crown_data in datajson:
            #just a check that the crown image is correct
                if img_dict["minx"]+'_'+img_dict["miny"] in crown_data["image_id"]:
                    crown = crown_data["segmentation"]

                    # changing the coords from RLE format so can be read as numbers, here the numbers are
                    # integers so a bit of info on position is lost
                    mask_of_coords = mask_util.decode(crown)
                    crown_coords = polygonFromMask(mask_of_coords)
                    rescaled_coords = []

                    # coords from json are in a list of [x1, y1, x2, y2,... ] so convert them to [[x1, y1], ...]
                    # format and at the same time rescale them so they are in the correct position for QGIS
                    for c in range(0, len(crown_coords), 2): 
                        x_coord=crown_coords[c]
                        y_coord=crown_coords[c+1]
                        rescaled_coords.append([x_coord,-y_coord])

                    geofile["features"].append({"type": "Feature", "properties": {}, "geometry" :{"type": "Polygon", "coordinates": [rescaled_coords]}})

            # Check final form is correct - compare to a known geojson file if error appears.
            print(geofile)

            output_geo_file = directory + img_dict["filename"].replace('.json','.geojson')
            print(output_geo_file)
            with open(output_geo_file, "w") as dest:
                json.dump(geofile,dest)