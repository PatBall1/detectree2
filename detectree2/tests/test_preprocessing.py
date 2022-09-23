import os
import pathlib
import tempfile
import unittest

import geopandas as gpd
import pytest
import rasterio
import requests


@pytest.mark.order(1)
class TestCase(unittest.TestCase):

    @pytest.mark.dependency()
    def test_tiling(self):

        data_dir = 'test_data'
        site_path = os.path.abspath(data_dir)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        # site_path = "detectree2-data"

        url = ("https://github.com/ma595/detectree2-data/raw/main/raw_images/"
               "/Paracou_20220426_RGB10cm_mosa_rect_cropsmall.tif")

        r = requests.get(url)
        img_path = os.path.join(site_path, 'sample_image_small.tif')
        with open(img_path, 'wb') as f:
            f.write(r.content)

        # img_path = os.path.join(site_path, "raw_images/Paracou_20220426_RGB10cm_mosa_rect_cropsmall.tif")
        # crown_path = os.path.join(site_path, "UpdatedCrowns8.gpkg")

        url = ("https://github.com/ma595/detectree2-data/raw/main/crowns/paracou/UpdatedCrowns8.gpkg")

        r = requests.get(url)
        crown_path = os.path.join(site_path, "crowns.gpkg")
        with open(crown_path, 'wb') as f:
            f.write(r.content)
        # crown_path = os.path.join(site_path, "crowns/paracou/220619_AllSpLabelled.gpkg")

        # Read in the tiff file
        data = rasterio.open(img_path)
        # Read in crowns
        crowns = gpd.read_file(crown_path)
        crowns = crowns.to_crs(data.crs.data)

        # Set tiling parameters
        buffer = 15
        tile_width = 40
        tile_height = 40
        threshold = 0.2

        from detectree2.preprocessing.tiling import tile_data_train

        out_dir = os.path.join(site_path, "paracou-out/tiles")

        tile_data_train(data, out_dir, buffer, tile_width, tile_height, crowns, threshold)

    # TODO: install pytest-depends to automatically order

    @pytest.mark.order(2)
    def test_image_details(self):
        site_path = os.path.abspath("test_data")
        out_dir = os.path.join(site_path, "paracou-out/tiles")

        # get the first file in the directory
        file = sorted(os.listdir(os.path.join(site_path, out_dir)))[0]
        file_root = pathlib.Path(file).stem
        print(file_root)
        from detectree2.preprocessing.tiling import image_details
        xbox_coords, ybox_coords = image_details(file_root)
        self.assertEqual(xbox_coords, (286523, 286593))
        self.assertEqual(ybox_coords, (583696, 583766))

    # @pytest.mark.dependency(depends=["test_tiling"]) # SKIPS tests - does not resolve order
    @pytest.mark.order(3)
    def test_to_traintest_folders(self):
        site_path = os.path.abspath("detectree2-data")
        tiles_path = os.path.join(site_path, 'ref-out/paracou-out/tiles')  # data exists without running test_tiling
        out_path = os.path.join(site_path, 'paracou-out/train_test_tiles')
        test_frac = 0.1
        folds = 3
        from detectree2.preprocessing.tiling import to_traintest_folders

        to_traintest_folders(tiles_path, out_path, test_frac, folds, seed=1)

        # TODO: now check that the outputs match to detectree2-data repo reference.


suite = unittest.TestLoader().loadTestsFromTestCase(TestCase)
