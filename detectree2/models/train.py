"""Train a model.

Classes and functions to train a model based on othomosaics and corresponding
manual crown data.
"""
import datetime
import glob
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import detectron2.data.transforms as T  # noqa:N812
import detectron2.utils.comm as comm
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer  # noqa:F401
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    DatasetMapper,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.structures import BoxMode
from detectron2.utils.events import get_event_storage  # noqa:F401
from detectron2.utils.events import EventStorage
from detectron2.utils.logger import log_every_n_seconds
from detectron2.utils.visualizer import ColorMode, Visualizer

# from IPython.display import display
# from PIL import Image


class LossEvalHook(HookBase):
    """Do inference and get the loss metric.

    Class to:
    - Do inference of dataset like an Evaluator does
    - Get the loss metric like the trainer does
    https://github.com/facebookresearch/detectron2/blob/master/detectron2/evaluation/evaluator.py
    https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/train_loop.py
    See https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b

    Attributes:
        model:
        period:
        data_loader:
        patience: number of evaluation periods to wait for improvement
    """

    def __init__(self, eval_period, model, data_loader, patience):
        """Inits LossEvalHook."""
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
        self.patience = patience
        self.iter = 0
        self.max_ap = 0
        self.best_iter = 0

    def _do_loss_eval(self):
        """Copying inference_on_dataset from evaluator.py.

        Returns:
            _type_: _description_
        """
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(idx + 1, total, seconds_per_img,
                                                                                    str(eta)),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        # print(self.trainer.cfg.DATASETS.TEST)
        # Combine the AP50s of the different datasets
        if len(self.trainer.cfg.DATASETS.TEST) > 1:
            APs = []
            for dataset in self.trainer.cfg.DATASETS.TEST:
                APs.append(self.trainer.test(self.trainer.cfg, self.trainer.model)[dataset]["segm"]["AP50"])
            AP = sum(APs) / len(APs)
        else:
            AP = self.trainer.test(self.trainer.cfg, self.trainer.model)["segm"]["AP50"]
        print("Av. AP50 =", AP)
        self.trainer.APs.append(AP)
        self.trainer.storage.put_scalar("validation_loss", mean_loss)
        self.trainer.storage.put_scalar("validation_ap", AP)
        comm.synchronize()

        return losses

    def _get_loss(self, data):
        """Calculate loss in train_loop.

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
            if self.max_ap < self.trainer.APs[-1]:
                self.iter = 0
                self.max_ap = self.trainer.APs[-1]
                self.trainer.checkpointer.save("model_" + str(len(self.trainer.APs)))
                self.best_iter = self.trainer.iter
            else:
                self.iter += 1
        if self.iter == self.patience:
            self.trainer.early_stop = True
            print("Early stopping occurs in iter {}, max ap is {}".format(self.best_iter, self.max_ap))
        self.trainer.storage.put_scalars(timetest=12)

    def after_train(self):
        # Select the model with the best AP50
        index = self.trainer.APs.index(max(self.trainer.APs)) + 1
        # Error in demo:
        # AssertionError: Checkpoint /__w/detectree2/detectree2/detectree2-data/paracou-out/train_outputs-1/model_1.pth
        # not found!
        # Therefore sleep is attempt to allow CI to pass, but it often still fails.
        time.sleep(15)
        self.trainer.checkpointer.load(self.trainer.cfg.OUTPUT_DIR + '/model_' + str(index) + '.pth')


# See https://jss367.github.io/data-augmentation-in-detectron2.html for data augmentation advice
class MyTrainer(DefaultTrainer):
    """Summary.

    Args:
        DefaultTrainer (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __init__(self, cfg, patience):  # noqa: D107
        self.patience = patience
        # self.resize = resize
        super().__init__(cfg)

    def train(self):
        """Run training.

        Args:
            start_iter, max_iter (int): See docs above

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """

        start_iter = self.start_iter
        max_iter = self.max_iter
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter
        self.early_stop = False
        self.APs = []

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
                    if self.early_stop:
                        break
                # self.iter == max_iter can be used by `after_train` to
                # tell whether the training successfully finished or failed
                # due to exceptions.
                self.iter += 1
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()
        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
            assert hasattr(self, "_last_eval_results"), "No evaluation results obtained during training!"
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("eval", exist_ok=True)
            output_folder = "eval"
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(
            -1,
            LossEvalHook(
                self.cfg.TEST.EVAL_PERIOD,
                self.model,
                build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST, DatasetMapper(self.cfg, True)),
                self.patience,
            ),
        )
        return hooks


def build_train_loader(cls, cfg):
    """Summary.

    Args:
        cfg (_type_): _description_

    Returns:
        _type_: _description_
    """
    augmentations = [
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3),
        T.RandomSaturation(0.8, 1.4),
        T.RandomRotation(angle=[90, 90], expand=False),
        T.RandomLighting(0.7),
        T.RandomFlip(prob=0.4, horizontal=True, vertical=False),
        T.RandomFlip(prob=0.4, horizontal=False, vertical=True),
    ]

    if cfg.RESIZE:
        augmentations.append(T.Resize((1000, 1000)))
    elif cfg.RESIZE == "random":
        augmentations.append(T.Resize((1000, 1000)))
        augmentations.append(T.ResizeScale(800 / 1000, 1333 / 800, 1000, 1000))
    return build_detection_train_loader(
        cfg,
        mapper=DatasetMapper(
            cfg,
            is_train=True,
            augmentations=augmentations,
        ),
    )


def get_tree_dicts(directory: str, classes: List[str] = None) -> List[Dict]:
    """Get the tree dictionaries.

    Args:
        directory: Path to directory
        classes: Signifies which column (if any) corresponds to the class labels

    Returns:
        List of dictionaries corresponding to segmentations of trees. Each dictionary includes
        bounding box around tree and points tracing a polygon around a tree.
    """
    # filepath = '/content/drive/MyDrive/forestseg/paracou_data/Panayiotis_Outputs/220303_AllSpLabelled.gpkg'
    # datagpd = gpd.read_file(filepath)
    # List_Genus = datagpd.Genus_Species.to_list()
    # Genus_Species_UniqueList = list(set(List_Genus))

    #
    if classes is not None:
        # list_of_classes = crowns[variable].unique().tolist()
        list_of_classes = ["CIRAD", "CNES", "INRA"]
        classes = list_of_classes
    else:
        classes = ["tree"]
    # classes = Genus_Species_UniqueList #['tree'] # genus_species list
    dataset_dicts = []
    # for root, dirs, files in os.walk(train_location):
    #    for file in files:
    #        if file.endswith(".geojson"):
    #            print(os.path.join(root, file))

    for filename in [file for file in os.listdir(directory) if file.endswith(".geojson")]:
        json_file = os.path.join(directory, filename)
        with open(json_file) as f:
            img_anns = json.load(f)
        # Turn off type checking for annotations until we have a better solution
        record: Dict[str, Any] = {}

        # filename = os.path.join(directory, img_anns["imagePath"])
        filename = img_anns["imagePath"]

        # Make sure we have the correct height and width
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["height"] = height
        record["width"] = width
        record["image_id"] = filename[0:400]
        record["annotations"] = {}
        # print(filename[0:400])

        objs = []
        for features in img_anns["features"]:
            anno = features["geometry"]
            # pdb.set_trace()
            # GenusSpecies = features['properties']['Genus_Species']
            px = [a[0] for a in anno["coordinates"][0]]
            py = [np.array(height) - a[1] for a in anno["coordinates"][0]]
            # print("### HERE IS PY ###", py)
            poly = [(x, y) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]
            # print("#### HERE ARE SOME POLYS #####", poly)
            if classes != ["tree"]:
                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": classes.index(features["properties"]["PlotOrg"]),  # id
                    # "category_id": 0,  #id
                    "iscrowd": 0,
                }
            else:
                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": 0,  # id
                    "iscrowd": 0,
                }
            # pdb.set_trace()
            objs.append(obj)
            # print("#### HERE IS OBJS #####", objs)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def combine_dicts(root_dir: str, val_dir: int, mode: str = "train") -> List[Dict]:
    """Join tree dicts from different directories.

    Args:
        root_dir:
        val_dir:

    Returns:
        Concatenated array of dictionaries over all directories
    """
    train_dirs = [os.path.join(root_dir, dir) for dir in os.listdir(root_dir)]
    if mode == "train":
        del train_dirs[(val_dir - 1)]
        tree_dicts = []
        for d in train_dirs:
            tree_dicts += get_tree_dicts(d)
    elif mode == "val":
        tree_dicts = get_tree_dicts(train_dirs[(val_dir - 1)])
    elif mode == "full":
        tree_dicts = []
        for d in train_dirs:
            tree_dicts += get_tree_dicts(d)
    return tree_dicts


def get_filenames(directory: str):
    """Get the file names if no geojson is present.

    Allows for predictions where no delinations have been manually produced.

    Args:
        directory (str): directory of images to be predicted on
    """
    dataset_dicts = []
    files = glob.glob(directory + "*.png")
    for filename in [file for file in files]:
        file = {}
        filename = os.path.join(directory, filename)
        file["file_name"] = filename

        dataset_dicts.append(file)
    return dataset_dicts


def register_train_data(train_location, name: str = "tree", val_fold=None):
    """Register data for training and (optionally) validation.

    Args:
        train_location: directory containing training folds
        name: string to name data
        val_fold: fold assigned for validation and tuning. If not given,
        will take place on all folds.
    """
    if val_fold is not None:
        for d in ["train", "val"]:
            DatasetCatalog.register(name + "_" + d, lambda d=d: combine_dicts(train_location, val_fold, d))
            MetadataCatalog.get(name + "_" + d).set(thing_classes=["tree"])
    else:
        DatasetCatalog.register(name + "_" + "full", lambda d=d: combine_dicts(train_location, 0, "full"))
        MetadataCatalog.get(name + "_" + "full").set(thing_classes=["tree"])


def remove_registered_data(name="tree"):
    for d in ["train", "val"]:
        DatasetCatalog.remove(name + "_" + d)
        MetadataCatalog.remove(name + "_" + d)


def register_test_data(test_location, name="tree"):
    d = "test"
    DatasetCatalog.register(name + "_" + d, lambda d=d: get_tree_dicts(test_location))
    MetadataCatalog.get(name + "_" + d).set(thing_classes=["tree"])


def load_json_arr(json_path):
    lines = []
    with open(json_path, "r") as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


def setup_cfg(
    base_model: str = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
    trains=("trees_train", ),
    tests=("trees_val", ),
    update_model=None,
    workers=2,
    ims_per_batch=2,
    gamma=0.1,
    backbone_freeze=3,
    warm_iter=120,
    momentum=0.9,
    batch_size_per_im=1024,
    base_lr=0.001,
    max_iter=1000,
    num_classes=1,
    eval_period=100,
    out_dir="/content/drive/Shareddrives/detectree2/train_outputs",
    resize=True,
):
    """Set up config object # noqa: D417.

    Args:
        base_model: base pre-trained model from detectron2 model_zoo
        trains: names of registered data to use for training
        tests: names of registered data to use for evaluating models
        update_model: updated pre-trained model from detectree2 model_garden
        workers:
        ims_per_batch:
        gamma:
        backbone_freeze:
        warm_iter:
        momentum:
        batch_size_per_im:
        base_lr:
        max_iter:
        num_classes:
        eval_period:
        out_dir:
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(base_model))
    cfg.DATASETS.TRAIN = trains
    cfg.DATASETS.TEST = tests
    cfg.DATALOADER.NUM_WORKERS = workers
    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.GAMMA = gamma
    cfg.MODEL.BACKBONE.FREEZE_AT = backbone_freeze
    cfg.SOLVER.WARMUP_ITERS = warm_iter
    cfg.SOLVER.MOMENTUM = momentum
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = batch_size_per_im
    cfg.SOLVER.WEIGHT_DECAY = 0.001
    cfg.SOLVER.BASE_LR = base_lr
    cfg.OUTPUT_DIR = out_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    if update_model is not None:
        cfg.MODEL.WEIGHTS = update_model
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(base_model)

    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.TEST.EVAL_PERIOD = eval_period
    cfg.RESIZE = resize
    return cfg


def predictions_on_data(directory=None,
                        predictor=DefaultTrainer,
                        trees_metadata=None,
                        save=True,
                        scale=1,
                        geos_exist=True,
                        num_predictions=0):
    """Prediction produced from a test folder and outputted to predictions folder."""

    test_location = directory + "/test"
    pred_dir = test_location + "/predictions"

    Path(pred_dir).mkdir(parents=True, exist_ok=True)

    if geos_exist:
        dataset_dicts = get_tree_dicts(test_location)
    else:
        dataset_dicts = get_filenames(test_location)

    # Works out if all items in folder should be predicted on
    if num_predictions == 0:
        num_to_pred = len(dataset_dicts)
    else:
        num_to_pred = num_predictions

    for d in random.sample(dataset_dicts, num_to_pred):
        img = cv2.imread(d["file_name"])
        # cv2_imshow(img)
        outputs = predictor(img)
        v = Visualizer(
            img[:, :, ::-1],
            metadata=trees_metadata,
            scale=scale,
            instance_mode=ColorMode.SEGMENTATION,
        )  # remove the colors of unsegmented pixels
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # image = cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
        # display(Image.fromarray(image))

        # Creating the file name of the output file
        file_name_path = d["file_name"]
        # Strips off all slashes so just final file name left
        file_name = os.path.basename(os.path.normpath(file_name_path))
        file_name = file_name.replace("png", "json")

        output_file = pred_dir + "/Prediction_" + file_name

        if save:
            # Converting the predictions to json files and saving them in the specfied output file.
            evaluations = instances_to_coco_json(outputs["instances"].to("cpu"), d["file_name"])
            with open(output_file, "w") as dest:
                json.dump(evaluations, dest)


if __name__ == "__main__":
    train_location = "/content/drive/Shareddrives/detectree2/data/Paracou/tiles/train/"
    register_train_data(train_location, "Paracou", 1)  # folder, name, validation fold

    name = "Paracou2019"
    train_location = "/content/drive/Shareddrives/detectree2/data/Paracou/tiles2019/train/"
    dataset_dicts = combine_dicts(train_location, 1)
    trees_metadata = MetadataCatalog.get(name + "_train")
    # dataset_dicts = get_tree_dicts("./")
    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=trees_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        image = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
        # display(Image.fromarray(image))
    # Set the base (pre-trained) model from the detectron2 model_zoo
    model = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    # Set the names of the registered train and test sets
    # pretrained model?
    # trained_model = "/content/drive/Shareddrives/detectree2/models/220629_ParacouSepilokDanum_JB.pth"
    trains = (
        "Paracou_train",
        "Paracou2019_train",
        "ParacouUAV_train",
        "Danum_train",
        "SepilokEast_train",
        "SepilokWest_train",
    )
    tests = (
        "Paracou_val",
        "Paracou2019_val",
        "ParacouUAV_val",
        "Danum_val",
        "SepilokEast_val",
        "SepilokWest_val",
    )
    out_dir = "/content/drive/Shareddrives/detectree2/220703_train_outputs"

    # update_model arg can be used to load in trained  model
    cfg = setup_cfg(model, trains, tests, eval_period=100, max_iter=3000, out_dir=out_dir)
    trainer = MyTrainer(cfg, patience=4)
    trainer.resume_or_load(resume=False)
    trainer.train()
