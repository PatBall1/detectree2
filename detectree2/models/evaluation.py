"""Evaluate model performance.

Classes and functions to evaluate model performances.
"""
import json
import os
from pathlib import Path
from statistics import median

import numpy as np
import rasterio
import rasterio.drivers
from rasterio.mask import mask
from shapely.geometry import Polygon, shape

# Initialising the parent class so any attributes or functions that are common
# to both features should be placed in here


class Feature:
    """Feature class to store.

    Longer class information.
    """

    def __init__(self, filename, directory, number, feature, lidar_filename, lidar_img, EPSG):  # noqa:N803
        """Initialise a crown feature with all the required attributes.

        Args:
            filename: name of the file within the directory in questions
            directory: the path to the file folder
            number: a label added to each crown to allow for identification
            feature: dictionary containing all the information about a crown
            lidar_filename: the full path to the crown file that overlays with the lidar data
            lidar_img: path to the lidar image of an entire region
            EPSG: area code of tree location
        """
        self.filename = filename
        self.directory = directory
        self.number = number
        self.properties = feature["properties"]
        self.geometry = feature["geometry"]
        self.GIoU = 0
        self.EPSG = EPSG
        self.lidar_filename = lidar_filename
        self.lidar_img = lidar_img
        self.GIoU_other_feat_num = -1
        self.poly_area()
        self.tree_height()

    def get_tuple_coords(self, coords):
        """Converts coordinates' data structure from a list of lists to a list of tuples."""
        coord_tuples = []

        for entry in coords:
            coord_tuples.append((entry[0], entry[1]))

        return coord_tuples

    def poly_area(self):
        """Calculates the area of the feature from scaled geojson."""
        polygon = Polygon(self.get_tuple_coords(self.geometry["coordinates"][0]))

        self.crown_area = polygon.area

    def tree_height(self):
        """Crops the lidar tif to the features and calculates height.

        Calculates the median height to account for
        error at the boundaries. If no lidar file is inputted than the height is
        given as 0
        """
        if self.lidar_img is None:
            self.height = 0
        else:
            with open(self.lidar_filename) as lidar_file:
                lidar_json = json.load(lidar_file)

            # Want coord tuples for the unmoved crown coordinates so using the
            # lidar copied crown file
            lidar_coords = lidar_json["features"][self.number]["geometry"]["coordinates"][0]
            geo = [{
                "type": "Polygon",
                "coordinates": [self.get_tuple_coords(lidar_coords)],
            }]

            with rasterio.open(self.lidar_img) as src:
                out_image, out_transform = mask(src, geo, all_touched=True,
                                                crop=True)
            out_meta = src.meta.copy()    # noqa:F841

            # remove all the values that are nodata values and recorded as negatives
            fixed_array = (out_image[out_image > 0])

            # the lidar data can have missed out areas or have noise meaning
            # the array is empty hence we will give this feature height 0 so
            # it is still used in calculating F1 scores in general but ignored
            # if any height restriction is used
            if len(fixed_array) != 0:
                sorted_array = np.sort(fixed_array)
                self.height = sorted_array[int(len(sorted_array) * 0.5)]
            else:
                self.height = 0


class GeoFeature:
    """Feature class to store.

    Longer class information.
    """

    def __init__(self, filename, directory, number, feature,
                 lidar_img, EPSG):    # noqa:N803
        """Initialise a crown feature with all the required attributes.

        Args:
            filename: name of the file within the directory in questions
            directory: the path to the file folder
            number: a label added to each crown to allow for identification
            feature: dictionary containing all the information about a crown
            lidar_img: path to the lidar image of an entire region
            EPSG: area code of tree location
        """
        self.filename = filename
        self.directory = directory
        self.number = number
        self.properties = feature['properties']
        self.geometry = feature['geometry']
        self.GIoU = 0
        self.EPSG = EPSG
        self.lidar_img = lidar_img
        self.GIoU_other_feat_num = -1
        self.poly_area()
        self.tree_height()

    def get_tuple_coords(self, coords):
        """Converts coordinates' data structure from a list of lists to a list
        of tuples."""
        coord_tuples = []

        for entry in coords:
            coord_tuples.append((entry[0], entry[1]))

        return coord_tuples

    def poly_area(self):
        """Calculates the area of the feature from scaled geojson."""
        polygon = Polygon(self.get_tuple_coords(
            self.geometry['coordinates'][0]))

        self.crown_area = polygon.area

    def tree_height(self):
        """Crops the lidar tif to the features and calculates height

        Calculates the median height to account for
        error at the boundaries. If no lidar file is inputted than the height is
        given as 0
        """
        if self.lidar_img is None:
            self.height = 0
        else:

            coords = self.geometry['coordinates'][0]
            geo = [{
                'type': 'Polygon',
                'coordinates': [self.get_tuple_coords(coords)]
            }]

            with rasterio.open(self.lidar_img) as src:
                out_image, out_transform = mask(src, geo, all_touched=True,
                                                crop=True)
            out_meta = src.meta.copy()    # noqa:F841

            # remove all the values that are nodata values and recorded as negatives
            fixed_array = out_image[out_image > 0]

            # the lidar data can have missed out areas or have noise meaning
            # the array is empty hence we will give this feature height 0 so
            # it is still used in calculating F1 scores in general but ignored
            # if any height restriction is used
            if len(fixed_array) != 0:
                sorted_array = np.sort(fixed_array)
                self.height = sorted_array[int(len(sorted_array) * 0.5)]
            else:
                self.height = 0


# Regular functions now
def get_tile_width(file):
    """Split up the file name to get width and buffer then adding to get overall width."""
    filename = file.replace(".geojson", "")
    filename_split = filename.split("_")

    tile_width = (2 * int(filename_split[-2]) + int(filename_split[-3]))

    return tile_width


def get_epsg(file):
    """Splitting up the file name to get EPSG"""
    filename = file.replace(".geojson", "")
    filename_split = filename.split("_")

    epsg = filename_split[-1]
    return epsg


def get_tile_origin(file):
    """Splitting up the file name to get tile origin"""
    filename = file.replace(".geojson", "")
    filename_split = filename.split("_")

    buffer = int(filename_split[-2])
    # center = int(filename_split[-3])
    y0 = int(filename_split[-4])
    x0 = int(filename_split[-5])

    origin = [x0 - buffer, y0 - buffer]
    return origin


def feat_threshold_tests(
    feature_instance,
    conf_threshold,
    area_threshold,
    border_filter,
    tile_width
):
    """Tests completed to see if a feature should be considered valid.

    Checks if the feature is above the confidence threshold if there is a confidence score available (only applies in
    predicted crown case).  Filters out features with areas too small which are often crowns that are from an
    adjacent tile that have a bit spilt over. Removes features within a border of the edge, border size is given by
    border_filter proportion of the tile width.

    """
    valid_feature = True

    if "Confidence_score" in feature_instance.properties:
        if feature_instance.properties["Confidence_score"] < conf_threshold:
            valid_feature = False

    if feature_instance.crown_area < area_threshold:
        valid_feature = False

    # variables stand for tile width and edge buffer
    TW = tile_width
    if valid_feature and border_filter[0]:
        EB = tile_width * border_filter[1]

        # Go through each coordinate pair in feautre
        for coords in feature_instance.geometry['coordinates'][0]:
            # if coordinate is out of bounds, skip it
            if (-EB <= coords[0] <= EB or -EB <= coords[1] <= EB
                    or TW - EB <= coords[0] <= TW + EB
                    or TW - EB <= coords[1] <= TW + EB):
                valid_feature = False
                break

    return valid_feature


def feat_threshold_tests2(
    feature_instance,
    conf_threshold,
    area_threshold,
    border_filter,
    tile_width,
    tile_origin
):
    """Tests completed to see if a feature should be considered valid.

    Checks if the feature is above the confidence threshold if there is a
    confidence score available (only applies in predicted crown case).  Filters
    out features with areas too small which are often crowns that are from an
    adjacent tile that have a bit spilt over. Removes features within a border
    of the edge, border size is given by border_filter proportion of the tile
    width.

    """
    valid_feature = True

    if "Confidence_score" in feature_instance.properties:
        if feature_instance.properties["Confidence_score"] < conf_threshold:
            valid_feature = False

    if feature_instance.crown_area < area_threshold:
        valid_feature = False

    # variables stand for tile width and edge buffer
    TW = tile_width
    TO = tile_origin

    if valid_feature and border_filter[0]:
        EB = border_filter[1]
        # Go through each coordinate pair in feautre
        for coords in feature_instance.geometry['coordinates'][0]:
            # if coordinate is out of bounds, skip it
            # print(coords[0], TO[0], TW, EB)
            # print(coords[1], TO[1], TW, EB)
            if (coords[0] <= TO[0] + EB
                    or coords[1] <= TO[1] + EB
                    or TO[0] + TW - EB <= coords[0]
                    or TO[1] + TW - EB <= coords[1]):
                valid_feature = False
                break

    return valid_feature


def initialise_feats(
    directory,
    file,
    lidar_filename,
    lidar_img,
    area_threshold,
    conf_threshold,
    border_filter,
    tile_width,
    EPSG,
):
    """Creates a list of all the features as objects of the class."""
    with open(directory + "/" + file) as feat_file:
        feat_json = json.load(feat_file)
    feats = feat_json["features"]

    all_feats = []
    count = 0
    for feat in feats:
        feat_obj = Feature(file, directory, count, feat, lidar_filename, lidar_img, EPSG)

        if feat_threshold_tests(feat_obj, conf_threshold, area_threshold, border_filter, tile_width):
            all_feats.append(feat_obj)
            count += 1
        else:
            continue

    return all_feats


def initialise_feats2(
    directory,
    file,
    lidar_img,
    area_threshold,
    conf_threshold,
    border_filter,
    tile_width,
    tile_origin,
    epsg
):
    """Creates a list of all the features as objects of the class."""
    with open(directory + "/" + file) as feat_file:
        feat_json = json.load(feat_file)
    feats = feat_json["features"]

    all_feats = []
    count = 0
    for feat in feats:
        feat_obj = GeoFeature(file, directory, count, feat, lidar_img, epsg)

        if feat_threshold_tests2(feat_obj, conf_threshold, area_threshold,
                                 border_filter, tile_width, tile_origin):
            all_feats.append(feat_obj)
            count += 1
        else:
            continue

    return all_feats


def save_feats(tile_directory, all_feats):
    """Collating all the information for the features back into a geojson to save."""

    adjusted_directory = tile_directory + "/adjusted/"
    Path(adjusted_directory).mkdir(parents=True, exist_ok=True)

    geofile = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {
                "name": "urn:ogc:def:crs:EPSG::" + all_feats[0].EPSG
            },
        },
        "features": [],
    }

    for feat in all_feats:
        geofile["features"].append({
            "type": "Feature",
            "properties": feat.properties,
            "geometry": feat.geometry,
        })

    output_geo_file = adjusted_directory + feat.filename.replace(".geojson", "_adjusted.geojson")
    with open(output_geo_file, "w") as dest:
        json.dump(geofile, dest)


def find_intersections(all_test_feats, all_pred_feats):
    """Finds the greatest intersection between predicted and manual crowns and then updates objects."""

    for pred_feat in all_pred_feats:
        for test_feat in all_test_feats:
            if shape(test_feat.geometry).intersects(shape(pred_feat.geometry)):
                try:
                    intersection = (shape(pred_feat.geometry).intersection(shape(test_feat.geometry))).area
                except ValueError:
                    continue

                # calculate the IoU
                # union_area = pred_feat.crown_area + test_feat.crown_area - intersection
                union_area = (shape(pred_feat.geometry).union(shape(test_feat.geometry))).area
                IoU = intersection / union_area

                # update the objects so they only store greatest intersection value
                if IoU > test_feat.GIoU:
                    test_feat.GIoU = IoU
                    test_feat.GIoU_other_feat_num = pred_feat.number

                if IoU > pred_feat.GIoU:
                    pred_feat.GIoU = IoU
                    pred_feat.GIoU_other_feat_num = test_feat.number


def feats_height_filt(all_feats, min_height, max_height):
    """Stores the numbers of all the features between min and max height."""
    tall_feat = []

    for feat in all_feats:
        # print(feat.height)
        if feat.height >= min_height and feat.height < max_height:
            tall_feat.append(feat.number)

    return tall_feat


def get_heights(all_feats, min_height, max_height):
    """Find the heights of the trees

    """
    heights = []
    test_nums = feats_height_filt(all_feats, min_height, max_height)

    for feat in all_feats:
        # if the pred feat is not tall enough then skip it
        if feat.number not in test_nums:
            continue
        else:
            heights.append(feat.height)
    return heights


def positives_test(all_test_feats, all_pred_feats, min_IoU, min_height, max_height):
    """Determines number of true postives, false positives and false negatives.

    Store the numbers of all test features which have true positives arise.
    """

    test_feats_tps = []

    tps = 0
    fps = 0

    tall_test_nums = feats_height_filt(all_test_feats, min_height, max_height)
    tall_pred_nums = feats_height_filt(all_pred_feats, min_height, max_height)

    for pred_feat in all_pred_feats:
        # if the pred feat is not tall enough then skip it
        if pred_feat.number not in tall_pred_nums:
            continue
        # if the number has remained at -1 it means the pred feat does not intersect
        # with any test feat and hence is a false positive.
        if pred_feat.GIoU_other_feat_num == -1:
            fps += 1
            continue

        # test to see if the two crowns overlap with each other the most and if
        # they are above the required GIoU. Then need the height of the test feature
        # to also be above the threshold to allow it to be considered
        matching_test_feat = all_test_feats[pred_feat.GIoU_other_feat_num]
        if (pred_feat.number == matching_test_feat.GIoU_other_feat_num and pred_feat.GIoU > min_IoU
                and matching_test_feat.number in tall_test_nums):
            tps += 1
            test_feats_tps.append(matching_test_feat.number)
        else:
            fps += 1

    fns = len(tall_test_nums) - len(test_feats_tps)

    return tps, fps, fns


def prec_recall(total_tps: int, total_fps: int, total_fns: int):
    """Calculate the precision and recall by standard formulas."""

    precision = total_tps / (total_tps + total_fps)
    recall = total_tps / (total_tps + total_fns)

    return precision, recall


def f1_cal(precision, recall):
    """Calculate the F1 score."""

    return (2 * precision * recall) / (precision + recall)


def site_f1_score(
    tile_directory=None,
    test_directory=None,
    pred_directory=None,
    lidar_img=None,
    IoU_threshold=0.5,
    height_threshold=0,
    area_fraction_limit=0.0005,
    conf_threshold=0,
    border_filter=tuple,
    scaling=list,
    EPSG=None,
    save=False,
):
    """Calculating all the intersections of shapes in a pair of files and the area of the corresponding polygons.

    Args:
        tile_directory: path to the folder containing all of the tiles
        test_directory: path to the folder containing just the test files
        pred_directory: path to the folder containing the predictions and the reprojections
        lidar_img: path to the lidar image of an entire region
        IoU_threshold: minimum value of IoU such that the intersection can be considered a true positive
        height_threshold: minimum height of the features to be considered
        area_fraction_limit: proportion of the tile for which crowns with areas less than this will be ignored
        conf_threshold: minimun confidence of a predicted feature so that it is considered
        border_filter: bool of whether to remove border crowns, proportion of border to be used
        in relation to tile size
        scaling: x and y scaling used when tiling the image
        EPSG: area code of tree location
        save: bool to tell program whether the filtered crowns should be saved
    """

    if EPSG is None:
        raise ValueError("Set the EPSG value")

    test_entries = os.listdir(test_directory)
    total_tps = 0
    total_fps = 0
    total_fns = 0

    for file in test_entries:
        if ".geojson" in file:
            # work out the area threshold to ignore these crowns in the tiles
            tile_width = get_tile_width(file) * scaling[0]
            area_threshold = ((tile_width)**2) * area_fraction_limit

            test_lidar = tile_directory + "/" + file
            all_test_feats = initialise_feats(
                test_directory,
                file,
                test_lidar,
                lidar_img,
                area_threshold,
                conf_threshold,
                border_filter,
                tile_width,
                EPSG,
            )

            pred_file_path = "Prediction_" + file
            pred_lidar = tile_directory + "/predictions/" + pred_file_path
            all_pred_feats = initialise_feats(
                pred_directory,
                pred_file_path,
                pred_lidar,
                lidar_img,
                area_threshold,
                conf_threshold,
                border_filter,
                tile_width,
                EPSG,
            )

            if save:
                save_feats(tile_directory, all_test_feats)
                save_feats(tile_directory, all_pred_feats)

            find_intersections(all_test_feats, all_pred_feats)
            tps, fps, fns = positives_test(all_test_feats, all_pred_feats, IoU_threshold, height_threshold)

            print("tps:", tps)
            print("fps:", fps)
            print("fns:", fns)
            print("")

            total_tps = total_tps + tps
            total_fps = total_fps + fps
            total_fns = total_fns + fns

    try:
        prec, rec = prec_recall(total_tps, total_fps, total_fns)
        f1_score = f1_cal(prec, rec)  # noqa: F841
        print("Precision  ", "Recall  ", "F1")
        print(prec, rec, f1_score)
    except ZeroDivisionError:
        print("ZeroDivisionError: Height threshold is too large.")


def site_f1_score2(
    tile_directory=None,
    test_directory=None,
    pred_directory=None,
    lidar_img=None,
    IoU_threshold=0.5,
    min_height=0,
    max_height=100,
    area_threshold=25,
    conf_threshold=0,
    border_filter=tuple,
    save=False
):
    """Calculating all the intersections of shapes in a pair of files and the
    area of the corresponding polygons.

    Args:
        tile_directory: path to the folder containing all of the tiles
        test_directory: path to the folder containing just the test files
        pred_directory: path to the folder containing the predictions and the reprojections
        lidar_img: path to the lidar image of an entire region
        IoU_threshold: minimum value of IoU such that the intersection can be considered a true positive
        min_height: minimum height of the features to be considered
        max_height: minimum height of the features to be considered
        area_threshold: min crown area to consider in m^2
        conf_threshold: minimun confidence of a predicted feature so that it is considered
        border_filter: bool to remove border crowns, buffer in from border to be used (in m)
        in relation to tile size
        scaling: x and y scaling used when tiling the image
        save: bool to tell program whether the filtered crowns should be saved
    """

    test_entries = os.listdir(test_directory)
    total_tps = 0
    total_fps = 0
    total_fns = 0
    heights = []
    # total_tests = 0

    for file in test_entries:
        if ".geojson" in file:
            print(file)

            # work out the area threshold to ignore these crowns in the tiles
            # tile_width = get_tile_width(file) * scaling[0]
            # area_threshold = ((tile_width)**2) * area_fraction_limit

            tile_width = get_tile_width(file)
            tile_origin = get_tile_origin(file)
            epsg = get_epsg(file)

            test_file = file.replace(".geojson", "_geo.geojson")
            all_test_feats = initialise_feats2(tile_directory, test_file,
                                               lidar_img, area_threshold,
                                               conf_threshold, border_filter,
                                               tile_width, tile_origin, epsg)

            new_heights = get_heights(all_test_feats, min_height, max_height)
            heights.extend(new_heights)

            pred_file = "Prediction_" + file
            all_pred_feats = initialise_feats2(pred_directory, pred_file,
                                               lidar_img, area_threshold,
                                               conf_threshold, border_filter,
                                               tile_width, tile_origin, epsg)

            if save:
                save_feats(tile_directory, all_test_feats)
                save_feats(tile_directory, all_pred_feats)

            find_intersections(all_test_feats, all_pred_feats)
            tps, fps, fns = positives_test(all_test_feats, all_pred_feats,
                                           IoU_threshold, min_height, max_height)

            print("tps:", tps)
            print("fps:", fps)
            print("fns:", fns)
            print("")

            total_tps += tps
            total_fps += fps
            total_fns += fns

    try:
        prec, rec = prec_recall(total_tps, total_fps, total_fns)
        # not used!
        f1_score = f1_cal(prec, rec)    # noqa: F841
        med_height = median(heights)
        print("Precision  ", "Recall  ", "F1")
        print(prec, rec, f1_score)
        print(" ")
        print("Total_trees=", len(heights))
        print("med_height=", med_height)
    except ZeroDivisionError:
        print("ZeroDivisionError: Height threshold is too large.")
    return prec, rec, f1_score


if __name__ == "__main__":
    print("to do")
