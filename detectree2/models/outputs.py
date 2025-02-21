"""Process and clean predictions.

Funtions to process model predictions into outputs for model evaluation and
mapping crowns in geographic space.
"""
import glob
import json
import os
import re
from http.client import REQUEST_URI_TOO_LONG  # noqa: F401
from pathlib import Path
from typing import Any, Dict, List, TypedDict

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
import rasterio
from rasterio.crs import CRS
from shapely.affinity import scale
from shapely.geometry import Polygon, box, shape
from shapely.ops import orient

# Type aliases definitions
Feature = Dict[str, Any]
CRSType = TypedDict("CRSType", {"type": str, "properties": Dict[str, str]})
GeoFile = TypedDict("GeoFile", {"type": str, "crs": CRSType, "features": List[Feature]})


def polygon_from_mask(masked_arr):
    """Convert RLE data from the output instances into Polygons.

    Leads to a small about of data loss but does not affect performance?
    https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py <-- adapted from here
    """

    contours, _ = cv2.findContours(
        masked_arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    segmentation = []
    for contour in contours:
        # Valid polygons have >= 6 coordinates (3 points) -  for security we use 10
        if contour.size >= 10:
            contour = contour.flatten().tolist()
            # Ensure the polygon is closed (get rid of fiona warning?)
            if contour[:2] != contour[-2:]:  # if not closed
                # continue # better to skip?
                contour.extend(contour[:2])  # small artifacts due to this?
            segmentation.append(contour)

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
                        # TODO: make flexible to deal with hemispheres
                        if epsg == "26917":
                            rescaled_coords.append([x_coord, -y_coord])
                        else:
                            rescaled_coords.append(
                                [x_coord, -y_coord + int(img_dict["height"])])

                    geofile["features"].append({
                        "type": "Feature",
                        "properties": {
                            "Confidence_score": confidence_score
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [rescaled_coords],
                        },
                    })

            # Check final form is correct - compare to a known geojson file if
            # error appears.
            # print(geofile)

            output_geo_file = os.path.join(
                directory, img_dict["filename"].replace(".json", "_eval.geojson"))
            # print(output_geo_file)
            with open(output_geo_file, "w") as dest:
                json.dump(geofile, dest)


def project_to_geojson(tiles_path, pred_fold=None, output_fold=None, multi_class: bool = False):  # noqa:N803
    """Projects json predictions back in geographic space.

    Takes a json and changes it to a geojson so it can overlay with orthomosaic. Another copy is produced to overlay
    with PNGs.

    Args:
        tiles_path (str): Path to the tiles folder.
        pred_fold (str): Path to the predictions folder.
        output_fold (str): Path to the output folder.

    Returns:
        None
    """
    Path(output_fold).mkdir(parents=True, exist_ok=True)
    entries = list(Path(pred_fold) / file for file in os.listdir(pred_fold) if Path(file).suffix == ".json")
    total_files = len(entries)
    print(f"Projecting {total_files} files")

    for idx, filename in enumerate(entries, start=1):
        if idx % 50 == 0:
            print(f"Projecting file {idx} of {total_files}: {filename}")

        tifpath = Path(tiles_path) / Path(filename.name.replace("Prediction_", "")).with_suffix(".tif")

        data = rasterio.open(tifpath)
        epsg = CRS.from_string(data.crs.wkt)
        epsg = epsg.to_epsg()
        raster_transform = data.transform

        geofile = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {
                    "name": "urn:ogc:def:crs:EPSG::" + str(epsg)
                },
            },
            "features": [],
        }  # type: GeoFile

        # load the json file we need to convert into a geojson
        with open(filename, "r") as prediction_file:
            datajson = json.load(prediction_file)

        # json file is formated as a list of segmentation polygons so cycle through each one
        for crown_data in datajson:
            if multi_class:
                category = crown_data["category_id"]
                # print(category)
            crown = crown_data["segmentation"]
            confidence_score = crown_data["score"]

            # changing the coords from RLE format so can be read as numbers, here the numbers are
            # integers so a bit of info on position is lost
            mask_of_coords = mask_util.decode(crown)
            crown_coords = polygon_from_mask(mask_of_coords)
            if crown_coords == 0:
                continue

            crown_coords_array = np.array(crown_coords).reshape(-1, 2)
            x_coords, y_coords = rasterio.transform.xy(transform=raster_transform,
                                                       rows=crown_coords_array[:, 1],
                                                       cols=crown_coords_array[:, 0])
            moved_coords = list(zip(x_coords, y_coords))
            if multi_class:
                geofile["features"].append({
                    "type": "Feature",
                    "properties": {
                        "Confidence_score": confidence_score,
                        "category": category,
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [moved_coords],
                    },
                })
            else:
                geofile["features"].append({
                    "type": "Feature",
                    "properties": {
                        "Confidence_score": confidence_score
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [moved_coords],
                    },
                })
                # print(geofile["features"])

        output_geo_file = os.path.join(output_fold, filename.with_suffix(".geojson").name)

        with open(output_geo_file, "w") as dest:
            json.dump(geofile, dest)


def filename_geoinfo(filename):
    """Return geographic info of a tile from its filename."""
    parts = os.path.basename(filename).replace(".geojson", "").split("_")

    parts = [int(part) for part in parts[-5:]]  # type: ignore
    minx = parts[0]
    miny = parts[1]
    width = parts[2]
    buffer = parts[3]
    crs = parts[4]
    return (minx, miny, width, buffer, crs)


def box_filter(filename, shift: int = 0):
    """Create a bounding box from a file name to filter edge crowns.

    Args:
        filename: Name of the file.
        shift: Number of meters to shift the size of the bounding box in by. This is to avoid edge crowns.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the bounding box.
    """
    minx, miny, width, buffer, crs = filename_geoinfo(filename)
    bounding_box = box_make(minx, miny, width, buffer, crs, shift)
    return bounding_box


def box_make(minx: int, miny: int, width: int, buffer: int, crs, shift: int = 0):
    """Generate bounding box from geographic specifications.

    Args:
        minx: Minimum x coordinate.
        miny: Minimum y coordinate.
        width: Width of the tile.
        buffer: Buffer around the tile.
        crs: Coordinate reference system.
        shift: Number of meters to shift the size of the bounding box in by. This is to avoid edge crowns.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the bounding box.
    """
    bbox = box(
        minx - buffer + shift,
        miny - buffer + shift,
        minx + width + buffer - shift,
        miny + width + buffer - shift,
    )
    geo = gpd.GeoDataFrame({"geometry": bbox}, index=[0], crs=CRS.from_epsg(crs))
    return geo


def stitch_crowns(folder: str, shift: int = 1):
    """Stitch together predicted crowns.

    Args:
        folder: Path to folder containing geojson files.
        shift: Number of meters to shift the size of the bounding box in by. This is to avoid edge crowns.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing all the crowns.
    """
    crowns_path = Path(folder)
    files = list(crowns_path.glob("*geojson"))
    if len(files) == 0:
        raise FileNotFoundError("No geojson files found in folder.")

    _, _, _, _, crs = filename_geoinfo(files[0])

    total_files = len(files)
    crowns_list = []

    for idx, file in enumerate(files, start=1):
        if idx % 50 == 0:
            print(f"Stitching file {idx} of {total_files}: {file}")

        crowns_tile = gpd.read_file(file)  # This throws a huge amount of warnings fiona closed ring detected

        geo = box_filter(file, shift)

        crowns_tile = gpd.sjoin(crowns_tile, geo, "inner", "within")

        crowns_list.append(crowns_tile)

    crowns = pd.concat(crowns_list, ignore_index=True)
    crowns = crowns.drop("index_right", axis=1)

    if not isinstance(crowns, gpd.GeoDataFrame):
        crowns = gpd.GeoDataFrame(crowns, crs=CRS.from_epsg(crs))

    return crowns


def calc_iou(shape1, shape2):
    """Calculate the IoU of two shapes."""
    iou = shape1.intersection(shape2).area / shape1.union(shape2).area
    return iou


def clean_crowns(crowns: gpd.GeoDataFrame,
                 iou_threshold: float = 0.7,
                 confidence: float = 0.2,
                 area_threshold: float = 2,
                 field: str = "Confidence_score") -> gpd.GeoDataFrame:
    """Clean overlapping crowns.

    Outputs can contain highly overlapping crowns including in the buffer region.
    This function removes crowns with a high degree of overlap with others but a
    lower Confidence Score.

    Args:
        crowns (gpd.GeoDataFrame): Crowns to be cleaned.
        iou_threshold (float, optional): IoU threshold that determines whether crowns are overlapping.
        confidence (float, optional): Minimum confidence score for crowns to be retained. Defaults to 0.2. Note that
            this should be adjusted to fit "field".
        area_threshold (float, optional): Minimum area of crowns to be retained. Defaults to 1m2 (assuming UTM).
        field (str): Field to used to prioritise selection of crowns. Defaults to "Confidence_score" but this should
            be changed to "Area" if using a model that outputs area.

    Returns:
        gpd.GeoDataFrame: Cleaned crowns.
    """
    # Filter any rows with empty or invalid geometry
    crowns = crowns[~crowns.is_empty & crowns.is_valid]

    # Filter any rows with polgon of less than 1m2 as these are likely to be artifacts
    crowns = crowns[crowns.area > area_threshold]

    crowns.reset_index(drop=True, inplace=True)

    cleaned_crowns = []
    print(f"Cleaning {len(crowns)} crowns")

    for index, row in crowns.iterrows():
        if index % 1000 == 0:
            print(f"{index} / {len(crowns)} crowns cleaned")

        intersecting_rows = crowns[crowns.intersects(shape(row.geometry))]

        if len(intersecting_rows) > 1:
            iou_values = intersecting_rows.geometry.map(lambda x: calc_iou(row.geometry, x))
            intersecting_rows = intersecting_rows.assign(iou=iou_values)

            # Filter rows with IoU over threshold and get the one with the highest confidence score
            match = intersecting_rows[intersecting_rows["iou"] > iou_threshold].nlargest(1, field)

            if match["iou"].iloc[0] < 1:
                continue

        else:
            match = row.to_frame().T

        cleaned_crowns.append(match)

    crowns_out = pd.concat(cleaned_crowns, ignore_index=True)

    # Drop 'iou' column if it exists
    if "iou" in crowns_out.columns:
        crowns_out = crowns_out.drop("iou", axis=1)

    # Ensuring crowns_out is a GeoDataFrame
    if not isinstance(crowns_out, gpd.GeoDataFrame):
        crowns_out = gpd.GeoDataFrame(crowns_out, crs=crowns.crs)
    else:
        crowns_out = crowns_out.set_crs(crowns.crs)

    # Filter remaining crowns based on confidence score
    if confidence != 0:
        crowns_out = crowns_out[crowns_out[field] > confidence]

    return crowns_out.reset_index(drop=True)


def post_clean(unclean_df: gpd.GeoDataFrame,
               clean_df: gpd.GeoDataFrame,
               iou_threshold: float = 0.3,
               field: str = "Confidence_score") -> gpd.GeoDataFrame:
    """Fill in the gaps left by clean_crowns.

    Args:
        unclean_df (gpd.GeoDataFrame): Unclean crowns.
        clean_df (gpd.GeoDataFrame): Clean crowns.
        iou_threshold (float, optional): IoU threshold that determines whether predictions are considered overlapping.
        crowns are overlapping. Defaults to 0.3.
    """
    # Spatial join between unclean and clean dataframes using the new syntax
    joined_df = gpd.sjoin(unclean_df, clean_df, how="inner", predicate="intersects")

    to_remove = []
    for idx, row in joined_df.iterrows():
        # Using the default suffix 'left' for columns from the unclean_df and 'right' for columns from the clean_df
        unclean_shape = unclean_df.loc[idx, "geometry"]
        clean_shape = clean_df.loc[row["index_right"], "geometry"]

        unclean_shape = unclean_shape.buffer(0)
        clean_shape = clean_shape.buffer(0)

        intersection_area = unclean_shape.intersection(clean_shape).area
        union_area = unclean_shape.union(clean_shape).area
        iou = intersection_area / union_area

        if iou > iou_threshold:
            to_remove.append(idx)

    reduced_unclean_df = unclean_df.drop(index=to_remove)

    # Concatenate the reduced unclean dataframe with the clean dataframe
    result_df = pd.concat([clean_df, reduced_unclean_df], ignore_index=True)

    result_df.reset_index(drop=True, inplace=True)

    reclean_df = clean_crowns(result_df, iou_threshold=iou_threshold, field=field)

    return reclean_df.reset_index(drop=True)


def load_geopandas_dataframes(folder):
    """Load all GeoPackage files in a folder into a list of GeoDataFrames."""
    all_files = glob.glob(f"{folder}/*.gpkg")
    filenames = [f for f in all_files if re.match(rf"{folder}/crowns_\d+\.gpkg", f)]

    # Load each file into a GeoDataFrame and add it to a list
    geopandas_dataframes = [gpd.read_file(filename) for filename in filenames]

    return geopandas_dataframes


def normalize_polygon(polygon, num_points):
    """Normalize a polygon to a set number of points."""
    # Orient polygon to ensure consistent vertex order (counterclockwise)
    polygon = orient(polygon, sign=1.0)

    # Get all points
    points = list(polygon.exterior.coords)

    # Get point with minimum average of x and y
    min_avg_point = min(points, key=lambda point: sum(point) / len(point))

    # Rotate points to start from min_avg_point
    min_avg_point_idx = points.index(min_avg_point)
    points = points[min_avg_point_idx:] + points[:min_avg_point_idx]

    # Create a new polygon with ordered points
    polygon = Polygon(points)

    total_perimeter = polygon.length
    distance_between_points = total_perimeter / num_points

    normalized_points = [polygon.boundary.interpolate(i * distance_between_points) for i in range(num_points)]
    return Polygon(normalized_points)


def average_polygons(polygons, weights=None, num_points=300):
    """Average a set of polygons."""
    normalized_polygons = [normalize_polygon(poly, num_points) for poly in polygons]

    avg_polygon_points = []
    for i in range(num_points):
        if weights:
            points_at_i = [
                np.array(poly.exterior.coords[i]) * weight
                for poly, weight in zip(normalized_polygons, weights)
            ]
            avg_point_at_i = sum(points_at_i) / sum(weights)
        else:
            points_at_i = [np.array(poly.exterior.coords[i]) for poly in normalized_polygons]
            avg_point_at_i = sum(points_at_i) / len(normalized_polygons)
        avg_polygon_points.append(tuple(avg_point_at_i))
    avg_polygon = Polygon(avg_polygon_points)

    # Compute the average centroid of the input polygons
    average_centroid = (
        np.mean([poly.centroid.x for poly in polygons]),
        np.mean([poly.centroid.y for poly in polygons])
    )

    # Compute the average area of the input polygons
    average_area = np.mean([poly.area for poly in polygons])

    # Calculate the scale factor
    scale_factor = np.sqrt(average_area / avg_polygon.area)

    # Scale the average polygon
    avg_polygon_scaled = scale(avg_polygon, xfact=scale_factor, yfact=scale_factor, origin=average_centroid)

    return avg_polygon_scaled


def combine_and_average_polygons(gdfs, iou=0.9):
    """Combine and average polygons."""
    # Combine all dataframes into one
    combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    combined_gdf["Confidence_score"] = pd.to_numeric(combined_gdf["Confidence_score"], errors="coerce")

    # Create spatial index
    sindex = combined_gdf.sindex

    # Empty lists to store results
    new_polygons = []
    combined_counts = []
    summed_confidences = []

    total_rows = combined_gdf.shape[0]
    print(f"Total rows: {total_rows}")

    # Iterate over each polygon
    for idx, row in combined_gdf.iterrows():
        if idx % 500 == 0:
            print(f"Processing row {idx} of {total_rows}")
        polygon = row.geometry
        confidence = row.Confidence_score if "Confidence_score" in combined_gdf.columns else None

        possible_matches_index = list(sindex.intersection(polygon.bounds))
        possible_matches = combined_gdf.iloc[possible_matches_index]
        exact_matches = possible_matches[possible_matches.intersects(polygon)]

        significant_matches = []
        significant_confidences = []

        for idx_match, row_match in exact_matches.iterrows():
            match = row_match.geometry
            match_confidence = row_match.Confidence_score if "Confidence_score" in combined_gdf.columns else None

            intersection = polygon.intersection(match)
            if intersection.area / (polygon.area + match.area - intersection.area) > iou:
                significant_matches.append(match)
                if match_confidence is not None:
                    significant_confidences.append(match_confidence)

        if len(significant_matches) > 1:
            averaged_polygon = average_polygons(
                significant_matches,
                significant_confidences if "Confidence_score" in combined_gdf.columns else None
            )
            new_polygons.append(averaged_polygon)
            combined_counts.append(len(significant_matches))
            if confidence is not None:
                summed_confidences.append(sum(significant_confidences))
        else:
            new_polygons.append(polygon)
            combined_counts.append(1)
            if confidence is not None:
                summed_confidences.append(confidence)

    # Create a new GeoPandas dataframe with the averaged polygons
    new_gdf = gpd.GeoDataFrame(geometry=new_polygons)
    new_gdf["combined_counts"] = combined_counts
    if "Confidence_score" in combined_gdf.columns:
        new_gdf["summed_confidences"] = summed_confidences

    # Set crs
    new_gdf.set_crs(gdfs[0].crs, inplace=True)

    return new_gdf


def clean_predictions(directory, iou_threshold=0.7):
    """Clean predictions prior to accuracy assessment."""
    pred_fold = directory
    entries = os.listdir(pred_fold)

    for file in entries:
        if ".json" in file:
            print(file)
            with open(pred_fold + "/" + file) as prediction_file:
                datajson = json.load(prediction_file)

            crowns = gpd.GeoDataFrame()

            for shp in datajson:
                crown_coords = polygon_from_mask(
                    mask_util.decode(shp["segmentation"]))
                if crown_coords == 0:
                    continue
                rescaled_coords = []
                # coords from json are in a list of [x1, y1, x2, y2,... ] so convert them to [[x1, y1], ...]
                # format and at the same time rescale them so they are in the correct position for QGIS
                for c in range(0, len(crown_coords), 2):
                    x_coord = crown_coords[c]
                    y_coord = crown_coords[c + 1]
                    rescaled_coords.append([x_coord, y_coord])
                crowns = pd.concat([crowns, gpd.GeoDataFrame({'Confidence_score': shp['score'],
                                                              'geometry': [Polygon(rescaled_coords)]},
                                                             geometry=[Polygon(rescaled_coords)])])

            crowns = crowns.reset_index().drop('index', axis=1)
            crowns, indices = clean_outputs(crowns, iou_threshold)
            datajson_reduced = [datajson[i] for i in indices]
            print("data_json:", len(datajson), " ", len(datajson_reduced))
            with open(pred_fold + "/" + file, "w") as dest:
                json.dump(datajson_reduced, dest)


def clean_outputs(crowns: gpd.GeoDataFrame, iou_threshold=0.7):
    """Clean predictions prior to accuracy assessment.

    Outputs can contain highly overlapping crowns including in the buffer region.
    This function removes crowns with a high degree of overlap with others but a
    lower Confidence Score.
    """
    crowns = crowns[crowns.is_valid]
    crowns_out = gpd.GeoDataFrame()
    indices = []
    for index, row in crowns.iterrows():  # iterate over each crown
        if index % 1000 == 0:
            print(str(index) + " / " + str(len(crowns)) + " cleaned")
        # if there is not a crown interesects with the row (other than itself)
        if crowns.intersects(row.geometry).sum() == 1:
            crowns_out = pd.concat(crowns_out, row)  # type: ignore
        else:
            # Find those crowns that intersect with it
            intersecting = crowns.loc[crowns.intersects(row.geometry)]
            intersecting = intersecting.reset_index().drop("index", axis=1)
            iou = []
            for index1, row1 in intersecting.iterrows():  # iterate over those intersecting crowns
                # print(row1.geometry)
                # area = row.geometry.intersection(row.geometry).area
                # area1 = row1.geometry.intersection(row1.geometry).area
                # intersection_1 = row.geometry.intersection(row1.geometry).area
                # if intersection_1 >= area*0.8 or intersection_1 >= area1*0.8:
                #    print("contained")
                #    iou.append(1)
                # else:
                # Calculate the IoU with each of those crowns
                iou.append(calc_iou(row.geometry, row1.geometry))
            # print(iou)
            intersecting['iou'] = iou
            # Remove those crowns with a poor match
            matches = intersecting[intersecting['iou'] > iou_threshold]
            matches = matches.sort_values(
                'Confidence_score', ascending=False).reset_index().drop('index', axis=1)
            # Of the remaining crowns select the crown with the highest confidence
            match = matches.loc[[0]]
            if match['iou'][0] < 1:   # If the most confident is not the initial crown
                continue
            else:
                match = match.drop('iou', axis=1)
                indices.append(index)
                crowns_out = pd.concat([crowns_out, match])
    return crowns_out, indices


if __name__ == "__main__":
    print("to do")
