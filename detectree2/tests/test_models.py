# import os
import unittest

import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


class TestCase(unittest.TestCase):

    def test_detectron2(self):
        # !wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
        im = cv2.imread("./detectree2/tests/input/input.jpg")
        cfg = get_cfg()
        cfg.MODEL.DEVICE = 'cpu'
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
        cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl'  # noqa: E501
        predictor = DefaultPredictor(cfg)
        outputs = predictor(im)
        print(outputs)
        self.assertEqual(1, 1)

    # @pytest.mark.dependency(
    #     depends=["tests/test_preprocessing::test_tiling", "tests/test_preprocessing::test_to_traintest_folders"],
    #     scope='session'
    # )
    # def test_train(self):
    #     """Integration test: Training on Paracou dataset for a single step.

    #     Run on CPU.
    #     """
    #     from detectree2.models.train import (MyTrainer, register_train_data,
    #                                          setup_cfg)
    #     val_fold = 1
    #     site_path = os.path.abspath("detectree2-data")
    #     train_location = os.path.join(site_path, 'ref-out/paracou/train_test_tiles/train/')
    #     register_train_data(train_location, "Paracou", val_fold)
    #     # 10.5281/zenodo.5515408
    #     model = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    #     trains = ('Paracou_train',)
    #     tests = ('Paracou_val',)
    #     out_dir = os.path.join(site_path, "paracou-out/train_outputs")
    #     cfg = setup_cfg(model, trains, tests, warm_iter=1, workers=1, eval_period=10, max_iter=1, out_dir=out_dir)
    #     cfg.MODEL.DEVICE = 'cpu'
    #     trainer = MyTrainer(cfg, patience=10)
    #     trainer.resume_or_load(resume=False)
    #     trainer.train()


suite = unittest.TestLoader().loadTestsFromTestCase(TestCase)
