import os
import numpy as np
import json
from shapely.geometry import shape, Polygon
import rasterio
import rasterio.drivers
from rasterio.mask import mask


# Initialising the feature class
class Feature:

    def __init__(self, filename, directory, number, feature, lidar_filename,
                 lidar_img, EPSG):
        self.filename = filename
        self.directory = directory
        self.number = number
        self.properties = feature['properties']
        self.geometry = feature['geometry']
        self.GIoU = 0
        self.EPSG = EPSG
        self.lidar_filename = lidar_filename
        self.lidar_img = lidar_img
        self.GIoU_other_feat_num = -1
        self.PolyArea()
        self.TreeHeight()

    def get_tuple_coords(self, coords):
        """
    Changes the coordinates from a list of lists to a list of tuples
    """
        coord_tuples = []

        for entry in coords:
            coord_tuples.append((entry[0], entry[1]))

        return coord_tuples

    def PolyArea(self):
        "Calculates the area of the feature from scaled geojson"
        polygon = Polygon(self.get_tuple_coords(
            self.geometry['coordinates'][0]))

        self.crown_area = polygon.area

    def TreeHeight(self):
        """
    Crops the lidar tif to the features and then calculcates the 95% greatest height to account for error at the top end.
    If no lidar file is inputted than the height is given as 0
    """
        if self.lidar_img == None:
            self.height = 0
        else:
            with open(self.lidar_filename) as lidar_file:
                lidar_json = json.load(lidar_file)

            # Want coord tuples for the unmoved crown coordinates so using the lidar copied crown file
            lidar_coords = lidar_json['features'][
                self.number]['geometry']['coordinates'][0]
            # print(lidar_coords)
            geo = [{
                'type': 'Polygon',
                'coordinates': [self.get_tuple_coords(lidar_coords)]
            }]

            with rasterio.open(self.lidar_img) as src:
                out_image, out_transform = mask(src, geo, crop=True)
            out_meta = src.meta.copy()

            # remove all the values that are nodata values and recorded as negatives
            fixed_array = (out_image[out_image > 0])

            # the lidar data can have missed out areas or have noise meaning the array is empty
            # hence we will give this feature height 0 so it is still used in calculating F1
            # scores in general but ignored if any height restriction is used
            if len(fixed_array) != 0:
                sorted_array = np.sort(fixed_array)
                self.height = sorted_array[int(len(sorted_array) * 0.95)]
            else:
                self.height = 0


# Regular functions now
def get_tile_area(file):
    """
  Splitting up the file name to get width and buffer then squaring the result to get area
  """
    filename = file.replace(".geojson", "")
    filename_split = filename.split("_")

    area = (2 * int(filename_split[-1]) + int(filename_split[-2]))**2
    return area


def initialise_feats(directory, file, lidar_filename, lidar_img, area_threshold,
                     EPSG):
    """
  Creates a list of all the features as objects of the class.
  It filters out features with areas too small which are often crowns
  that are from an adjacent tile that have a bit of split over
  """
    with open(directory + file) as feat_file:
        feat_json = json.load(feat_file)
    feats = feat_json["features"]
    # print("features:", feats)

    all_feats = []
    count = 0
    for feat in feats:
        feat_obj = Feature(file, directory, count, feat, lidar_filename,
                           lidar_img, EPSG)
        if feat_obj.crown_area > area_threshold:
            all_feats.append(feat_obj)
            count += 1
        else:
            continue

    return all_feats


def find_intersections(all_test_feats, all_pred_feats):
    """
  Finds the greatest intersection between the predicted and manual crowns and then
  updates the objects respectively
  """

    for pred_feat in all_pred_feats:
        for test_feat in all_test_feats:
            if shape(test_feat.geometry).intersects(shape(pred_feat.geometry)):
                try:
                    intersection = (shape(pred_feat.geometry).intersection(
                        shape(test_feat.geometry))).area
                except Exception:
                    continue

                # calculate the IoU
                union_area = pred_feat.crown_area + test_feat.crown_area - intersection
                IoU = intersection / union_area

                # update the objects so they only store greatest intersection value
                if IoU > test_feat.GIoU:
                    test_feat.GIoU = IoU
                    test_feat.GIoU_other_feat_num = pred_feat.number

                if IoU > pred_feat.GIoU:
                    pred_feat.GIoU = IoU
                    pred_feat.GIoU_other_feat_num = test_feat.number


def feats_tall_enough(all_feats, min_height):
    """
  Stores the numbers of all the features above the minimun height
  """
    tall_feat = []

    for feat in all_feats:
        if feat.height >= min_height:
            tall_feat.append(feat.number)

    return tall_feat


def positives_test(all_test_feats, all_pred_feats, min_IoU, min_height):
    """
  Works out how many true postives, false positives and false negatives we have.
  """
    # Store the numbers of all test features which have true positives arise
    test_feats_tps = []

    tps = 0
    fps = 0

    tall_test_nums = feats_tall_enough(all_test_feats, min_height)
    tall_pred_nums = feats_tall_enough(all_pred_feats, min_height)

    for pred_feat in all_pred_feats:
        # if the pred feat is not all enough then skip it
        if pred_feat.number not in tall_pred_nums:
            continue
        # if the number has remained at -1 it means the pred feat does not intersect
        # with any test feat and hence is a false positive.
        if pred_feat.GIoU_other_feat_num == -1:
            fps += 1
            continue

        # test to see if the two crowns both overlap with each other the most and if
        # they are above the required GIoU. Then need the height of the test feature
        # to also be above the threshold to allow it to be considered
        matching_test_feat = all_test_feats[pred_feat.GIoU_other_feat_num]
        if (pred_feat.number == matching_test_feat.GIoU_other_feat_num
                and pred_feat.GIoU > min_IoU
                and matching_test_feat.number in tall_test_nums):
            tps += 1
            test_feats_tps.append(matching_test_feat.number)
        else:
            fps += 1

    fns = len(tall_test_nums) - len(test_feats_tps)

    return tps, fps, fns


def prec_recall_func(total_tps: int, total_fps: int, total_fns: int):
    """Calculate the precision and recall by standard formulas
  """

    precision = total_tps / (total_tps + total_fps)
    recall = total_tps / (total_tps + total_fns)

    return precision, recall


def f1_cal(precision: float, recall: float):
    """Calculate the F1 score

  Args:
    precision
    recall
  
  retu
  """

    return (2 * precision * recall) / (precision + recall)


def site_F1_score(
    tile_directory=None,
    test_directory=None,
    pred_directory=None,
    lidar_img=None,
    IoU_threshold=0,
    height_threshold=0,
    area_fraction_limit=0.0005,
    scaling=list,
    EPSG=None,
):
    """
  Code to calculate all the intersections of shapes in a pair of files and the area of the corresponding polygons
  Output the test_count so 
  """

    if EPSG == None:
        raise ValueError('Set the EPSG value')

    test_entries = os.listdir(test_directory)
    total_tps = 0
    total_fps = 0
    total_fns = 0

    for file in test_entries:
        if ".geojson" in file:
            print(file)

            # work out the area threshold to ignore these crowns in the tiles
            tile_area = get_tile_area(file)
            area_threshold = tile_area * area_fraction_limit * scaling[
                0] * scaling[1]
            # print("Area Threshold:", area_threshold)

            test_lidar = tile_directory + file.replace(".geojson",
                                                       "_lidar.geojson")
            all_test_feats = initialise_feats(test_directory, file, test_lidar,
                                              lidar_img, area_threshold, EPSG)

            pred_file_path = "Prediction_" + file.replace(
                '.geojson', '_' + EPSG + '.geojson')
            pred_lidar = tile_directory + "reprojected/" + pred_file_path
            all_pred_feats = initialise_feats(pred_directory, pred_file_path,
                                              pred_lidar, lidar_img,
                                              area_threshold, EPSG)

            find_intersections(all_test_feats, all_pred_feats)
            tps, fps, fns = positives_test(all_test_feats, all_pred_feats,
                                           IoU_threshold, height_threshold)

            print("tps:", tps)
            print("fps:", fps)
            print("fns:", fns)

            total_tps = total_tps + tps
            total_fps = total_fps + fps
            total_fns = total_fns + fns

            # print("height:", all_test_feats[0].height)
            # print("area:", all_test_feats[0].crown_area)
            # print("len:", len(all_test_feats))
            print("")

    try:
        prec, rec = prec_recall_func(total_tps, total_fps, total_fns)
        f1_score = f1_cal(prec, rec)
        print(f1_score)
    except:
        print("ZeroDivisionError: Height threshold is too large.")
