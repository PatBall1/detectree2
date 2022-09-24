import os
import pathlib
import unittest

import geopandas as gpd
import pytest
import rasterio
import requests


@pytest.mark.order(1)
class TestCase(unittest.TestCase):

    @pytest.mark.dependency()
    def test_tiling(self):
        """Calls tile_data_train function in tiling.py with example (though reduced) input.

        Cropped image from Paracou_20220426_RGB10cm_mosa_rect.tif from detectree2 google drive. Image currently stored
        in ma595/detectree2-data. Use HTTP request to download data (86MB) from github link and store in
        detectree2/tests/test_data/. Though tile_data_train is deterministic, the threshold setting will lead to
        different output. Subsequent functions assume a setting of 0.2.  Function is executed first using
        pytest.mark.order(). Other better approaches may be available.
        """
        # TODO: Consider alternative storing solution
        # TODO: Extend for other inputs (currently only tests paracou)
        # TODO: Use pathlib instead of join
        # TODO: Why are absolute paths needed here? For imagePath later on?
        # TODO: Consider deleting data after test?
        # TODO: pytest fixtures for test_dir / out_dir variables.
        test_dir = 'test_data'
        test_dir = os.path.abspath(test_dir)
        out_dir = os.path.join(test_dir, 'paracou_out')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        # get the tif file if it isn't downloaded already
        img_path = os.path.join(test_dir, 'raw', 'sample_image_small.tif')
        if not os.path.exists(img_path):
            os.makedirs(os.path.join(test_dir, 'raw'))
            url = ("https://github.com/ma595/detectree2-data/raw/main/raw_images/"
                   "/Paracou_20220426_RGB10cm_mosa_rect_cropsmall.tif")
            r = requests.get(url)
            with open(img_path, 'wb') as f:
                f.write(r.content)

        # get the crowns file (could be stored in repo)
        url = ("https://github.com/ma595/detectree2-data/raw/main/crowns/paracou/220619_AllSpLabelled.gpkg")
        r = requests.get(url)
        crown_path = os.path.join(test_dir, 'raw', "crowns.gpkg")
        with open(crown_path, 'wb') as f:
            f.write(r.content)

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

        out_dir = os.path.join(out_dir, "tiles")

        tile_data_train(data, out_dir, buffer, tile_width, tile_height, crowns, threshold)

    # TODO: install pytest-depends to automatically order

    @pytest.mark.order(2)
    def test_image_details(self):
        """Test the image_details function.

        Tile_data_train function is deterministic so will capture the same tiles with same bounding boxes around trees
        regardless of the input.  Therefore we know the first file in the list has the xbox and ybox coords given
        cbelow. The image_details function takes these from the generated tile file names. Function will need to be
        modified if data input changes.
        """
        # TODO: use pathlib instead of join.
        site_path = os.path.abspath("test_data")
        out_dir = os.path.join(site_path, "paracou_out/tiles")

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
        """Test to_traintest_folders splitting."""
        # TODO: Use pathlib instead of join.
        # TODO: The repetition of directories is bad practice.

        site_path = os.path.abspath("test_data")
        tiles_path = os.path.join(site_path, 'paracou_out/tiles')  # data exists without running test_tiling
        out_path = os.path.join(site_path, 'paracou_out/train_test_tiles')
        test_frac = 0.1
        folds = 3
        from detectree2.preprocessing.tiling import to_traintest_folders

        to_traintest_folders(tiles_path, out_path, test_frac, folds, seed=1)

        # TODO: now check that the outputs match to some known reference. Could be a list of files not the files
        # themselves.


suite = unittest.TestLoader().loadTestsFromTestCase(TestCase)
