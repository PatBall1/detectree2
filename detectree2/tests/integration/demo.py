import os
import unittest

import geopandas as gpd
import pytest
import rasterio

from detectree2.models.train import MyTrainer, register_train_data, setup_cfg


class TestCase(unittest.TestCase):

    @pytest.mark.dependency()
    def test_tiling(self):
        site_path = os.path.abspath("detectree2-data")
        img_path = os.path.join(site_path, "raw_images/Paracou_20220426_RGB10cm_mosa_rect_cropsmall.tif")
        crown_path = os.path.join(site_path, "crowns/paracou/220619_AllSpLabelled.gpkg")
        out_dir = os.path.join(site_path, "paracou-out/tiles")

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
        tile_data_train(data, out_dir, buffer, tile_width, tile_height, crowns, threshold)

    @pytest.mark.dependency(depends=["TestCase::test_tiling"])
    def test_to_traintest_folders(self):
        root_path = 'detectree2-data'
        tiles_path = os.path.join(root_path, 'paracou-out/tiles')
        out_path = os.path.join(root_path, 'paracou-out/train_test_tiles')
        test_frac = 0.1
        folds = 3
        from detectree2.preprocessing.tiling import to_traintest_folders

        to_traintest_folders(tiles_path, out_path, test_frac, folds, seed=1)

    @pytest.mark.dependency(depends=["TestCase::test_to_traintest_folders"])
    def test_train(self):
        """Integration test: Training on Paracou dataset for a single step.

        Runs on CPU.
        """

        val_fold = 1
        root_path = os.path.abspath("detectree2-data")
        train_location = os.path.join(root_path, 'paracou-out/train_test_tiles/train/')
        register_train_data(train_location, "Paracou", val_fold)
        model = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
        # model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
        trains = ('Paracou_train', )
        tests = ('Paracou_val', )
        out_dir = os.path.join(root_path, "paracou-out/train_outputs-1")
        cfg = setup_cfg(model, trains, tests, ims_per_batch=1, eval_period=10, max_iter=1, out_dir=out_dir)
        cfg.MODEL.DEVICE = 'cpu'
        trainer = MyTrainer(cfg, patience=10)
        trainer.resume_or_load(resume=False)
        trainer.train()


suite = unittest.TestLoader().loadTestsFromTestCase(TestCase)
