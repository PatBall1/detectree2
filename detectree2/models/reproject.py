# necessary basic libraries
import cv2
import os
import json
from pathlib import Path

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


# Reprojecting the crowns to overlay with the cropped crowns and the cropped png
def reproject_to_geojson(directory = None, EPSG = "26917"):
    """
    Takes a json and changes it to a geojson so it can overlay with crowns
    Another copy is produced to overlay with PNGs
    """

    entries = os.listdir(directory)

    for file in entries:
        if ".json" in file: 
            #create a geofile for each tile --> the EPSG value might need to be changed.
            geofile = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::" + EPSG }}, "features":[]}

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
            # print(img_dict)

            # json file is formated as a list of segmentation polygons so cycle through each one
            for crown_data in datajson:
            #just a check that the crown image is correct
                if img_dict["minx"]+'_'+img_dict["miny"] in crown_data["image_id"]:
                    crown = crown_data["segmentation"]
                    confidence_score = crown_data['score']

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

                        if EPSG == "26917":
                            rescaled_coords.append([x_coord,-y_coord])
                        else:
                            rescaled_coords.append([x_coord,-y_coord+int(img_dict["height"])])

                    geofile["features"].append({"type": "Feature", "properties": {"Confidence score": confidence_score}, "geometry" :{"type": "Polygon", "coordinates": [rescaled_coords]}})

            # Check final form is correct - compare to a known geojson file if error appears.
            print(geofile)

            output_geo_file = directory + img_dict["filename"].replace('.json',"_"+EPSG+'.geojson')
            print(output_geo_file)
            with open(output_geo_file, "w") as dest:
                json.dump(geofile,dest)


# Reprojects the coordinates back so the crowns can be overlaid with the original tif file of the entire region
def reproject_to_geojson_spatially(data, output_fold = None, pred_fold = None, EPSG = "26917"):
    """
    Takes a json and changes it to a geojson so it can overlay with crowns of the original tif
    """

    Path(output_fold).mkdir(parents=True, exist_ok=True)
    entries = os.listdir(pred_fold)

    # scale to deal with the resolution
    scalingx = data.transform[0]
    scalingy = -data.transform[4]

    for file in entries:
        if ".json" in file: 
            #create a geofile for each tile --> the EPSG value might need to be changed.
            geofile = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::" + EPSG }}, "features":[]}

            # create a dictionary for each file to store data used multiple times
            img_dict = {}
            img_dict["filename"] = file
            print(img_dict["filename"])

            file_mins = file.replace(".json", "")
            file_mins_split = file_mins.split("_")
            minx = int(file_mins_split[-4])
            miny = int(file_mins_split[-3])
            tile_height = int(file_mins_split[-2])
            buffer = int(file_mins_split[-1])
            height = (tile_height + 2*buffer)/scalingx

            # update the image dictionary to store all information cleanly
            img_dict.update({"minx":minx, "miny":miny, "height":height, "buffer": buffer})
            # print("Img dict:", img_dict)

            # load the json file we need to convert into a geojson
            with open(pred_fold+img_dict["filename"]) as prediction_file:
                datajson = json.load(prediction_file)
            # print("data_json:",datajson)


            # json file is formated as a list of segmentation polygons so cycle through each one
            for crown_data in datajson:
            #just a check that the crown image is correct
                if str(minx)+'_'+str(miny) in crown_data["image_id"]:
                    crown = crown_data["segmentation"]
                    confidence_score = crown_data['score']

                    # changing the coords from RLE format so can be read as numbers, here the numbers are
                    # integers so a bit of info on position is lost
                    mask_of_coords = mask_util.decode(crown)
                    crown_coords = polygonFromMask(mask_of_coords)
                    moved_coords = []
                
                    # coords from json are in a list of [x1, y1, x2, y2,... ] so convert them to [[x1, y1], ...]
                    # format and at the same time rescale them so they are in the correct position for QGIS
                    for c in range(0, len(crown_coords), 2): 
                        x_coord=crown_coords[c]
                        y_coord=crown_coords[c+1]

                        # print("ycoord:", y_coord)
                        # print("height:", height)

                        # rescaling the coords depending on where the tile is in the original image, note the correction
                        # factors have been manually added as outputs did not line up with predictions from training script
                        if minx == data.bounds[0] and miny == data.bounds[1]:
                          # print("Bottom Corner")
                          x_coord = (x_coord)*scalingx + minx
                          y_coord = (height-y_coord)*scalingy + miny
                        elif minx == data.bounds[0]: 
                          # print("Left Edge")
                          x_coord = (x_coord)*scalingx + minx
                          y_coord = (height-y_coord)*scalingy - buffer + miny
                        elif miny == data.bounds[1]:
                          # print("Bottom Edge")
                          x_coord = (x_coord)*scalingx - buffer + minx
                          y_coord = (height-y_coord)*scalingy - buffer + miny
                        else:
                          # print("Anywhere else") 
                          x_coord = (x_coord)*scalingx - buffer + minx 
                          y_coord = (height-y_coord)*scalingy - buffer + miny

                        moved_coords.append([x_coord,y_coord])

                    geofile["features"].append({"type": "Feature", "properties": {"Confidence score": confidence_score}, "geometry" :{"type": "Polygon", "coordinates": [moved_coords]}})

            # Check final form is correct - compare to a known geojson file if error appears.
            # print("geofile",geofile)

            output_geo_file = output_fold + img_dict["filename"].replace('.json',"_"+EPSG+'_lidar.geojson')
            # print("output location:", output_geo_file)
            with open(output_geo_file, "w") as dest:
              json.dump(geofile,dest)
