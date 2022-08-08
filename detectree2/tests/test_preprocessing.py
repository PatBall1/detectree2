import os
import unittest

import geopandas as gpd
import rasterio


class TestCase(unittest.TestCase):

    def test_upper(self):
        self.assertEqual("foo".upper(), "FOO")

    def test_tiling(self):
        site_path = "detectree2-data"
        img_path = os.path.join(site_path, "cropped_high_res_small.tif")
        crown_path = os.path.join(site_path, "UpdatedCrowns8.gpkg")

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

        out_dir = os.path.join(site_path, "out/tiles")

        tile_data_train(data, out_dir, buffer, tile_width, tile_height, crowns, threshold)

        return True

    def test_to_traintest_folders(self):
        root_path = 'detectree2-data'
        tiles_path = os.path.join(root_path, 'out/tiles')
        out_path = os.path.join(root_path, 'out/train_test_tiles')
        test_frac = 0.1
        folds = 5
        from detectree2.preprocessing.tiling import to_traintest_folders

        to_traintest_folders(tiles_path, out_path, test_frac, folds, seed=1)

        return True


suite = unittest.TestLoader().loadTestsFromTestCase(TestCase)
