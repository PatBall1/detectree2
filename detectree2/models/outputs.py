"""Process and clean predictions.

Funtions to process model predictions into outputs for model evaluation and
mapping crowns in geographic space.
"""
import json
import os
from http.client import REQUEST_URI_TOO_LONG  # noqa: F401
from pathlib import Path

import cv2
import geopandas as gpd
import pycocotools.mask as mask_util
from fiona.crs import from_epsg
from shapely.geometry import box, shape


def polygon_from_mask(masked_arr):
    """Convert RLE data from the output instances into Polygons.

    Leads to a small about of data loss but does not affect performance?
    https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py <-- found here
    """

    contours, _ = cv2.findContours(masked_arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    segmentation = []
    for contour in contours:
        # Valid polygons have >= 6 coordinates (3 points)
        if contour.size >= 6:
            segmentation.append(contour.flatten().tolist())
    # rles = mask_util.frPyObjects(segmentation, masked_arr.shape[0], masked_arr.shape[1])
    # RLE = mask_util.merge(RLEs) # not used
    # RLE = mask.encode(np.asfortranarray(masked_arr))
    # area = mask_util.area(RLE) # not used
    [x, y, w, h] = cv2.boundingRect(masked_arr)

    if len(segmentation) > 0:
        return segmentation[0]  # , [x, y, w, h], area
    else:
        return 0


def to_eval_geojson(directory=None):  # noqa:N803
    """Converts predicted jsons to a geojson for evaluation (not mapping!).

    Reproject the crowns to overlay with the cropped crowns and cropped pngs.
    Another copy is produced to overlay with pngs.
    """

    entries = os.listdir(directory)

    for file in entries:
        if ".json" in file:

            # create a dictionary for each file to store data used multiple times
            img_dict = {}
            img_dict["filename"] = file

            file_mins = file.replace(".json", "")
            file_mins_split = file_mins.split("_")
            img_dict["minx"] = file_mins_split[-5]
            img_dict["miny"] = file_mins_split[-4]
            epsg = file_mins_split[-1]
            # create a geofile for each tile --> the EPSG value should be done
            # automatically
            geofile = {
                "type": "FeatureCollection",
                "crs": {
                    "type": "name",
                    "properties": {
                        "name": "urn:ogc:def:crs:EPSG::" + epsg
                    },
                },
                "features": [],
            }

            # load the json file we need to convert into a geojson
            with open(directory + "/" + img_dict["filename"]) as prediction_file:
                datajson = json.load(prediction_file)

            img_dict["width"] = datajson[0]["segmentation"]["size"][0]
            img_dict["height"] = datajson[0]["segmentation"]["size"][1]
            # print(img_dict)

            # json file is formated as a list of segmentation polygons so cycle through each one
            for crown_data in datajson:
                # just a check that the crown image is correct
                if img_dict["minx"] + "_" + img_dict["miny"] in crown_data["image_id"]:
                    crown = crown_data["segmentation"]
                    confidence_score = crown_data["score"]

                    # changing the coords from RLE format so can be read as numbers, here the numbers are
                    # integers so a bit of info on position is lost
                    mask_of_coords = mask_util.decode(crown)
                    crown_coords = polygon_from_mask(mask_of_coords)
                    if crown_coords == 0:
                        continue
                    rescaled_coords = []

                    # coords from json are in a list of [x1, y1, x2, y2,... ] so convert them to [[x1, y1], ...]
                    # format and at the same time rescale them so they are in the correct position for QGIS
                    for c in range(0, len(crown_coords), 2):
                        x_coord = crown_coords[c]
                        y_coord = crown_coords[c + 1]

                        if epsg == "26917":
                            rescaled_coords.append([x_coord, -y_coord])
                        else:
                            rescaled_coords.append([x_coord, -y_coord + int(img_dict["height"])])

                    geofile["features"].append({
                        "type": "Feature",
                        "properties": {
                            "Confidence score": confidence_score
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [rescaled_coords],
                        },
                    })

            # Check final form is correct - compare to a known geojson file if
            # error appears.
            print(geofile)

            output_geo_file = os.path.join(directory, img_dict["filename"].replace(".json", "_eval.geojson"))
            print(output_geo_file)
            with open(output_geo_file, "w") as dest:
                json.dump(geofile, dest)


def project_to_geojson(data, output_fold=None, pred_fold=None):  # noqa:N803
    """Projects json predictions back in geographic space.

    Takes a json and changes it to a geojson so it can overlay with orthomosaic. Another copy is produced to overlay
    with PNGs.
    """

    Path(output_fold).mkdir(parents=True, exist_ok=True)
    entries = os.listdir(pred_fold)

    # scale to deal with the resolution
    scalingx = data.transform[0]
    scalingy = -data.transform[4]

    for file in entries:
        if ".json" in file:
            # create a dictionary for each file to store data used multiple times
            img_dict = {}
            img_dict["filename"] = file
            print(img_dict["filename"])

            file_mins = file.replace(".json", "")
            file_mins_split = file_mins.split("_")
            minx = int(file_mins_split[-5])
            miny = int(file_mins_split[-4])
            tile_height = int(file_mins_split[-3])
            buffer = int(file_mins_split[-2])
            height = (tile_height + 2 * buffer) / scalingx
            epsg = file_mins_split[-1]
            # create a geofile for each tile --> the EPSG value should be done
            # automatically
            geofile = {
                "type": "FeatureCollection",
                "crs": {
                    "type": "name",
                    "properties": {
                        "name": "urn:ogc:def:crs:EPSG::" + epsg
                    },
                },
                "features": [],
            }
            # update the image dictionary to store all information cleanly
            img_dict.update({"minx": minx, "miny": miny, "height": height, "buffer": buffer})
            # print("Img dict:", img_dict)

            # load the json file we need to convert into a geojson
            with open(pred_fold + "/" + img_dict["filename"]) as prediction_file:
                datajson = json.load(prediction_file)
            # print("data_json:",datajson)

            # json file is formated as a list of segmentation polygons so cycle through each one
            for crown_data in datajson:
                # just a check that the crown image is correct
                if str(minx) + "_" + str(miny) in crown_data["image_id"]:
                    crown = crown_data["segmentation"]
                    confidence_score = crown_data["score"]

                    # changing the coords from RLE format so can be read as numbers, here the numbers are
                    # integers so a bit of info on position is lost
                    mask_of_coords = mask_util.decode(crown)
                    crown_coords = polygon_from_mask(mask_of_coords)
                    if crown_coords == 0:
                        continue
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
                        if minx == data.bounds[0] and miny == data.bounds[1]:
                            # print("Bottom Corner")
                            x_coord = (x_coord) * scalingx + minx
                            y_coord = (height - y_coord) * scalingy + miny
                        elif minx == data.bounds[0]:
                            # print("Left Edge")
                            x_coord = (x_coord) * scalingx + minx
                            y_coord = (height - y_coord) * scalingy - buffer + miny
                        elif miny == data.bounds[1]:
                            # print("Bottom Edge")
                            x_coord = (x_coord) * scalingx - buffer + minx
                            y_coord = (height - y_coord) * scalingy - buffer + miny
                        else:
                            # print("Anywhere else")
                            x_coord = (x_coord) * scalingx - buffer + minx
                            y_coord = (height - y_coord) * scalingy - buffer + miny

                        moved_coords.append([x_coord, y_coord])

                    geofile["features"].append({
                        "type": "Feature",
                        "properties": {
                            "Confidence score": confidence_score
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [moved_coords],
                        },
                    })

            # Check final form is correct - compare to a known geojson file if error appears.
            # print("geofile",geofile)

            output_geo_file = os.path.join(output_fold, img_dict["filename"].replace(".json", ".geojson"))
            # print("output location:", output_geo_file)
            with open(output_geo_file, "w") as dest:
                json.dump(geofile, dest)


def filename_geoinfo(filename):
    """Return geographic info of a tile from its filename."""
    parts = os.path.basename(filename).split("_")

    parts = [int(part) for part in parts[-6:-1]]  # type: ignore
    minx = parts[0]
    miny = parts[1]
    width = parts[2]
    buffer = parts[3]
    crs = parts[4]
    return (minx, miny, width, buffer, crs)


def box_filter(filename, shift: int = 0):
    """Create a bounding box from a file name to filter edge crowns."""
    minx, miny, width, buffer, crs = filename_geoinfo(filename)
    bounding_box = box_make(minx, miny, width, buffer, crs, shift)
    return bounding_box


def box_make(minx: int, miny: int, width: int, buffer: int, crs, shift: int = 0):
    """Generate bounding box from geographic specifications."""
    bbox = box(
        minx - buffer + shift,
        miny - buffer + shift,
        minx + width + buffer - shift,
        miny + width + buffer - shift,
    )
    geo = gpd.GeoDataFrame({"geometry": bbox}, index=[0], crs=from_epsg(crs))
    return geo


def stitch_crowns(folder: str, shift: int = 1):
    """Stitch together predicted crowns."""
    crowns_path = Path(folder)
    files = crowns_path.glob("*geojson")
    _, _, _, _, crs = filename_geoinfo(list(files)[0])
    files = crowns_path.glob("*geojson")
    crowns = gpd.GeoDataFrame(
        columns=["Confidence score", "geometry"],
        geometry="geometry",
        crs=from_epsg(crs),
    )  # initiate an empty gpd.GDF
    for file in files:
        crowns_tile = gpd.read_file(file)
        # crowns_tile.crs = "epsg:32622"
        # crowns_tile = crowns_tile.set_crs(from_epsg(32622))
        # print(crowns_tile)

        geo = box_filter(file, shift)
        # geo.plot()
        crowns_tile = gpd.sjoin(crowns_tile, geo, "inner", "within")
        # print(crowns_tile)
        crowns = crowns.append(crowns_tile)
        # print(crowns)
    crowns = crowns.drop("index_right", axis=1).reset_index().drop("index", axis=1)
    # crowns = crowns.drop("index", axis=1)
    return crowns


def calc_iou(shape1, shape2):
    """Calculate the IoU of two shapes."""
    iou = shape1.intersection(shape2).area / shape1.union(shape2).area
    return iou


def clean_crowns(crowns: gpd.GeoDataFrame, iou_threshold=0.7):
    """Clean overlapping crowns.

    Outputs can contain highly overlapping crowns including in the buffer region.
    This function removes crowns with a high degree of overlap with others but a
    lower Confidence Score.
    """
    crowns_out = gpd.GeoDataFrame()
    for index, row in crowns.iterrows():  # iterate over each crown
        if index % 1000 == 0:
            print(str(index) + " / " + str(len(crowns)) + " cleaned")
        # if there is not a crown interesects with the row (other than itself)
        if crowns.intersects(shape(row.geometry)).sum() == 1:
            crowns_out = crowns_out.append(row)  # retain it
        else:
            # Find those crowns that intersect with it
            intersecting = crowns.loc[crowns.intersects(shape(row.geometry))]
            intersecting = intersecting.reset_index().drop("index", axis=1)
            iou = []
            for (
                    index1,
                    row1,
            ) in intersecting.iterrows():  # iterate over those intersecting crowns
                # print(row1.geometry)
                iou.append(calc_iou(row.geometry, row1.geometry))  # Calculate the IoU with each of those crowns
            # print(iou)
            intersecting["iou"] = iou
            matches = intersecting[intersecting["iou"] > iou_threshold]  # Remove those crowns with a poor match
            matches = matches.sort_values("Confidence score", ascending=False).reset_index().drop("index", axis=1)
            match = matches.loc[[0]]  # Of the remaining crowns select the crown with the highest confidence
            if match["iou"][0] < 1:  # If the most confident is not the initial crown
                continue
            else:
                match = match.drop("iou", axis=1)
                # print(index)
                crowns_out = crowns_out.append(match)
    return crowns_out.reset_index()


if __name__ == "__main__":
    print("to do")
