import os
import unittest

import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer


class TestCase(unittest.TestCase):

    def test_upper(self):
        self.assertEqual("foo".upper(), "FOO")

    def test_detectron2(self):
        # !wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
        im = cv2.imread("./input/input.jpg")

        # Create config
        cfg = get_cfg()

        cfg.MODEL.DEVICE = 'cpu'

        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))

        # cfg.merge_file = 'https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'
        # cfg.merge_from_file("./detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

        # set threshold for this model
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

        # cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"

        # Here we just get the pre-trained weights straight from facebook hosting website

        cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl'

        # Create predictor
        predictor = DefaultPredictor(cfg)

        # Make prediction
        outputs = predictor(im)

        self.assertEqual(1, 1)

    def test_train(self):
        """Integration test: test training detectree2 on tree data for a single step.

            Run on CPU.  
        """
        from detectree2.models.train import (MyTrainer, register_train_data,
                                             setup_cfg)

        val_fold = 1
        root = '/rds/project/rds-5mCMIDBOkPU/ma595/detectree2/data/tiles/'
        train_location = os.path.join(root, 'paracou/tiles/train')
        register_train_data(train_location, "Paracou", val_fold)
        # 10.5281/zenodo.5515408
        model = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
        trains = 'Paracou_train'
        tests = 'Paracou_val'
        out_dir = os.path.join(root, "0308_train_outputs")
        cfg = setup_cfg(model, trains, tests, eval_period=1, max_iter=1, out_dir=out_dir)
        cfg.MODEL.DEVICE = 'cpu'
        trainer = MyTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()


suite = unittest.TestLoader().loadTestsFromTestCase(TestCase)
