import json
import os

from detectree2.models.evaluation import Feature

# from detectree2.models.F1_calculator import (find_intersections, get_tile_width,
#                                              initialise_feats, positives_test, site_f1_score)  # noaq: F401


def test_site_f1_score():
    """Computes the F1 score on manufactured data from tests/input directory.

    The data in the test and pred dirs contains manufactured data to test IoU It will give a zero value for one of the
    predictions and a value of 1 for the other. Small refactor of evaluation.py is needed.
    Improve evaluation functions. Function is currently incomplete.
    """
    # TODO: Complete
    site_dir = "detectree2/tests/input/"
    test_directory = os.path.join(site_dir, "iou_data/test")
    pred_directory = os.path.join(site_dir, "iou_data/pred")  # noqa: F401, F841
    # tile_directory = os.path.join(site_dir, "tiles-ref/")  # noqa: F401, F841
    IoU_threshold = (0, )  # noqa: N803, F401, F841
    height_threshold = 0  # noqa: F401, F841

    test_entries = os.listdir(test_directory)
    for file in test_entries:
        if ".geojson" in file:
            pass
            # TODO:
            # The data in the test and pred dirs contains manufactured data to test IoU
            # It will give a zero value for one of the predictions and a value of 1 for the other
            # Improve evaluation functions.


def test_tree_feature():
    """
    Tests the Feature class with some dummy data from detectree2-data.

    Function is incomplete. Only returns known crown_area. Crowns are boxes so this is easy to evaluate.
    """
    # TODO: Remove detectree2-data dependency
    site_dir = "detectree2/tests/input/"
    test_dir = os.path.join(site_dir, "iou_data/test")
    EPSG = "32622"  # can get from rasterio?
    test_entries = os.listdir(test_dir)
    for file in test_entries:
        # if ".geojson" in file:
        with open(os.path.join(test_dir, file)) as feat_file:
            feat_json = json.load(feat_file)
        feats = feat_json["features"]

        # only one feature in each file, and areas are the same in both files
        feat_obj = Feature(
            file,
            test_dir,
            number=0,
            feature=feats[0],
            lidar_filename=None,
            lidar_img=None,
            EPSG=EPSG,
        )

    assert feat_obj.crown_area == 360000
