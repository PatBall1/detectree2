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
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (DatasetCatalog, DatasetMapper, MetadataCatalog,
                             build_detection_test_loader,
                             build_detection_train_loader)
from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.structures import BoxMode
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.utils.logger import log_every_n_seconds
from detectron2.utils.visualizer import ColorMode, Visualizer
from IPython.display import display
from PIL import Image


class LossEvalHook(HookBase):
  """Do inference and get the loss metric
    Class to:
    - Do inference of dataset like an Evaluator does
    - Get the loss metric like the trainer does
    https://github.com/facebookresearch/detectron2/blob/master/detectron2/evaluation/evaluator.py
    https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/train_loop.py
    See https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b
    Attributes:
        model:
        period
        data loader
    """
  """The algorithm of early-stopping is from <deep learning> of Goodfellow section 7.8.
    The main calculation of early_stopping is in after_step and 
    then the best weight recorded is loaded in the current model
    """


  def __init__(self, eval_period, model, data_loader, patience, out_dir):
    self._model = model
    self._period = eval_period
    self._data_loader = data_loader
    self.patience = patience
    self.iter = 0
    self.max_value = 0
    self.best_iter = 0
    #self.checkpointer = DetectionCheckpointer(self._model, save_dir=out_dir)

  def _do_loss_eval(self):
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
        total_seconds_per_img = (time.perf_counter() -
                                 start_time) / iters_after_start
        eta = datetime.timedelta(seconds=int(total_seconds_per_img *
                                             (total - idx - 1)))
        log_every_n_seconds(
            logging.INFO,
            "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                idx + 1, total, seconds_per_img, str(eta)),
            n=5,
        )
      loss_batch = self._get_loss(inputs)
      losses.append(loss_batch)
    mean_loss = np.mean(losses)
    self.trainer.storage.put_scalar("validation_loss", mean_loss, smoothing_hint = False)
    
    comm.synchronize()

    #return losses
  def _get_loss(self, data):
    """How loss is calculated on train_loop
        Args:
            data (_type_): _description_
        Returns:
            _type_: _description_
        """
    #
    metrics_dict = self._model(data)
    metrics_dict = {
        k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
        for k, v in metrics_dict.items()
    }
    total_losses_reduced = sum(loss for loss in metrics_dict.values())
    return total_losses_reduced


  '''early stop see <deep learning> of goodfellow'''
  def after_step(self):
    next_iter = self.trainer.iter + 1
    is_final = next_iter == self.trainer.max_iter
    if is_final or (self._period > 0 and next_iter % self._period == 0):
      if len(self.trainer.cfg.DATASETS.TEST) > 1:
          APs = []
          AP_datasets = self.trainer.test(
                    self.trainer.cfg,
                    self.trainer.model)
          for dataset in self.trainer.cfg.DATASETS.TEST:
            APs.append(AP_datasets[dataset]['segm']['AP50'])
          AP = sum(APs) / len(APs)
      else:
          AP = self.trainer.test(self.trainer.cfg, self.trainer.model)['segm']['AP50']
      print("Av. AP50 =", AP)
      self.trainer.values.append(AP)
      self.trainer.storage.put_scalar("validation_AP", AP, smoothing_hint = False)
      if self.trainer.metrix == 'AP50':
        if len(self.trainer.cfg.DATASETS.TEST) > 1:
          APs = []
          AP_datasets = self.trainer.test_train(
                    self.trainer.cfg,
                    self.trainer.model)
          for dataset in self.trainer.cfg.DATASETS.TEST:
            APs.append(AP_datasets[dataset]['segm']['AP50'])
          AP = sum(APs) / len(APs)
        else:
          AP = self.trainer.test_train(self.trainer.cfg, self.trainer.model)['segm']['AP50']
        self.trainer.storage.put_scalar("training_AP", AP, smoothing_hint = False)
      elif self.trainer.metrix == 'loss':
        self._do_loss_eval()
      else:
        if len(self.trainer.cfg.DATASETS.TEST) > 1:
          APs = []
          AP_datasets = self.trainer.test_train(
                    self.trainer.cfg,
                    self.trainer.model)
          for dataset in self.trainer.cfg.DATASETS.TRAIN:
            APs.append(AP_datasets[dataset]['segm']['AP50'])
          AP = sum(APs) / len(APs)
        else:
          AP = self.trainer.test(self.trainer.cfg, self.trainer.model)['segm']['AP50']
        self.trainer.storage.put_scalar("training_AP", AP, smoothing_hint = False)
        loss = self._do_loss_eval()
      if self.max_value < self.trainer.values[-1]:
        self.iter = 0
        self.max_value = self.trainer.values[-1]
        #self.checkpointer.save('model_' + str(len(self.trainer.values)))
        torch.save(self._model.state_dict(), self.trainer.cfg.OUTPUT_DIR + '/Model_' + str(len(self.trainer.values)) + '.pth')
        self.best_iter = self.trainer.iter
      else:
        self.iter += 1
    if self.iter == self.patience:
      self.trainer.early_stop = True
      print("Early stopping occurs in iter {}, max ap is {}".format(self.best_iter, self.max_value))
    self.trainer.storage.put_scalars(timetest=12)  

  def after_train(self):
    print('train done !!!')
    if len(self.trainer.values) != 0:
      index = self.trainer.values.index(max(self.trainer.values)) + 1
      print(self.trainer.early_stop,"best model is", index)
      trainer.cfg.MODEL.WEIGHTS = self.trainer.cfg.OUTPUT_DIR + '/Model_' + str(index) + '.pth'
    else:
      print('train fails')

# See https://jss367.github.io/data-augmentation-in-detectron2.html for data augmentation advice
class MyTrainer(DefaultTrainer):
  """_summary_
    Args:
        DefaultTrainer (_type_): _description_
    Returns:
        _type_: _description_
    """
  # add a judge on if early-stopping in train function
  # train is inherited from TrainerBase https://detectron2.readthedocs.io/en/latest/_modules/detectron2/engine/train_loop.html
  def __init__(self, cfg, patience = 5, training_metrix = 'loss'):
    self.patience = patience
    self.metrix = training_metrix
    super().__init__(cfg)
  def train(self):
    """
    Run training.

    Returns:
        OrderedDict of results, if evaluation is enabled. Otherwise None.
    """
    """
    Args:
        start_iter, max_iter (int): See docs above
    """
    start_iter = self.start_iter
    max_iter = self.max_iter
    logger = logging.getLogger(__name__)
    logger.info("Starting training from iteration {}".format(start_iter))

    self.iter = self.start_iter = start_iter
    self.max_iter = max_iter
    self.early_stop = False
    self.values = []
    self.reweight = False ### used to decide when to increase the weight of classification loss

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
        assert hasattr(
            self, "_last_eval_results"
        ), "No evaluation results obtained during training!"
        verify_results(self.cfg, self._last_eval_results)
        return self._last_eval_results


  def run_step(self):
    self._trainer.iter = self.iter
    """
    Implement the standard training logic described above.
    """
    assert self._trainer.model.training, "[SimpleTrainer] model was changed to eval mode!"
    start = time.perf_counter()
    """
    If you want to do something with the data, you can wrap the dataloader.
    """
    data = next(self._trainer._data_loader_iter)
    data_time = time.perf_counter() - start

    """
    If you want to do something with the losses, you can wrap the model.
    """
    loss_dict = self._trainer.model(data)
    if isinstance(loss_dict, torch.Tensor):
        losses = loss_dict
        loss_dict = {"total_loss": loss_dict}
    else:
        # loss_dict['cls'] = torch.tensor(0)
        # loss_dict['loss_rpn_cls'] = torch.tensor(0)
        # if self.iter > 1000:
        #   self.reweight = True
        # if self.reweight:
        #   loss_dict['loss_mask'] *= 4
        loss_dict['loss_mask'] *= 0.8
        losses = sum(loss_dict.values())

    """
    If you need to accumulate gradients or do something similar, you can
    wrap the optimizer with your custom `zero_grad()` method.
    """
    self.optimizer.zero_grad()
    losses.backward()

    self._trainer._write_metrics(loss_dict, data_time)

    """
    If you need gradient clipping/scaling or other processing, you can
    wrap the optimizer with your custom `step()` method. But it is
    suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
    """
    self._trainer.optimizer.step()

  
  def build_hooks(self):
    """
    Build a list of default hooks, including timing, evaluation,
    checkpointing, lr scheduling, precise BN, writing events.
    Returns:
        list[HookBase]:
    """
    cfg = self.cfg.clone()
    cfg.defrost()
    cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

    ret = [
        hooks.IterationTimer(),
        hooks.LRScheduler(),
        hooks.PreciseBN(
            # Run at the same freq as (but before) evaluation.
            cfg.TEST.EVAL_PERIOD,
            self.model,
            # Build a new data loader to not affect training
            self.build_train_loader(cfg),
            cfg.TEST.PRECISE_BN.NUM_ITER,
        )
        if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
        else None,
    ]

    # Do PreciseBN before checkpointer, because it updates the model and need to
    # be saved by checkpointer.
    # This is not always the best: if checkpointing has a different frequency,
    # some checkpoints may have more precise statistics than others.
    if comm.is_main_process():
        ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

    # def test_and_save_results():
    #     self._last_eval_results = self.test(self.cfg, self.model)
    #     return self._last_eval_results

    # # Do evaluation after checkpointer, because then if it fails,
    # # we can use the saved checkpoint to debug.
    # ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

    if comm.is_main_process():
        # Here the default print/log frequency of each writer is used.
        # run writers in the end, so that evaluation metrics are written
        ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
    ret.insert(
        -1,
        LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0],
            DatasetMapper(self.cfg, True)),
            self.patience,
            self.cfg.OUTPUT_DIR,
        ),
    )
    return ret


  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    if output_folder is None:
      os.makedirs("eval_2", exist_ok=True)
      output_folder = "eval_2"
    return COCOEvaluator(dataset_name, cfg, output_dir = output_folder)


  def build_train_loader(cls, cfg):
    """_summary_
        Args:
            cfg (_type_): _description_
        Returns:
            _type_: _description_
        """
    for i, datas in enumerate(DatasetCatalog.get(cfg.DATASETS.TRAIN[0])):
      location = datas['file_name']
      size = cv2.imread(location).shape[0]
      break
    return build_detection_train_loader(
        cfg,
        mapper=DatasetMapper(
            cfg,
            is_train=True,
            augmentations=[
                #T.Resize((800, 800)),
                #T.Resize((random_size, random_size)),
                T.ResizeScale(0.6, 1.4, size, size),
                #T.RandomCrop('relative',(0.5,0.5)),
                T.RandomBrightness(0.8, 1.8),
                T.RandomContrast(0.6, 1.3),
                T.RandomSaturation(0.8, 1.4),
                T.RandomRotation(angle=[90, 90], expand=False),
                T.RandomLighting(0.7),
                T.RandomFlip(prob=0.4, horizontal=True, vertical=False),
                T.RandomFlip(prob=0.4, horizontal=False, vertical=True),
            ],
        ),
    )

  @classmethod
  def test_train(cls, cfg, model, evaluators=None):
      """
      Evaluate the given model. The given model is expected to already contain
      weights to evaluate.

      Args:
          cfg (CfgNode):
          model (nn.Module):
          evaluators (list[DatasetEvaluator] or None): if None, will call
              :meth:`build_evaluator`. Otherwise, must have the same length as
              ``cfg.DATASETS.TEST``.

      Returns:
          dict: a dict of result metrics
      """
      logger = logging.getLogger(__name__)
      if isinstance(evaluators, DatasetEvaluator):
          evaluators = [evaluators]
      if evaluators is not None:
          assert len(cfg.DATASETS.TRAIN) == len(evaluators), "{} != {}".format(
              len(cfg.DATASETS.TRAIN), len(evaluators)
          )

      results = OrderedDict()
      for idx, dataset_name in enumerate(cfg.DATASETS.TRAIN):
          data_loader = cls.build_test_loader(cfg, dataset_name)
          # When evaluators are passed in as arguments,
          # implicitly assume that evaluators can be created before data_loader.
          if evaluators is not None:
              evaluator = evaluators[idx]
          else:
              try:
                  evaluator = cls.build_evaluator(cfg, dataset_name)
              except NotImplementedError:
                  logger.warn(
                      "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                      "or implement its `build_evaluator` method."
                  )
                  results[dataset_name] = {}
                  continue
          results_i = inference_on_dataset(model, data_loader, evaluator)
          results[dataset_name] = results_i
          if comm.is_main_process():
              assert isinstance(
                  results_i, dict
              ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                  results_i
              )
              logger.info("Evaluation results for {} in csv format:".format(dataset_name))
              print_csv_format(results_i)

      if len(results) == 1:
          results = list(results.values())[0]
      return results


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

    for filename in [
            file for file in os.listdir(directory) if file.endswith(".geojson")
    ]:
        json_file = os.path.join(directory, filename)
        with open(json_file) as f:
            img_anns = json.load(f)
        # Turn off type checking for annotations until we have a better solution
        record: dict[str, Any] = {}

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
                    "bbox": [np.min(px),
                             np.min(py),
                             np.max(px),
                             np.max(py)],
                    "bbox_mode":
                        BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id":
                        classes.index(features["properties"]["PlotOrg"]
                                      ),    # id
                    # "category_id": 0,  #id
                    "iscrowd":
                        0,
                }
            else:
                obj = {
                    "bbox": [np.min(px),
                             np.min(py),
                             np.max(px),
                             np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": 0,    # id
                    "iscrowd": 0,
                }
            # pdb.set_trace()
            objs.append(obj)
            # print("#### HERE IS OBJS #####", objs)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def combine_dicts(root_dir: str,
                  val_dir: int,
                  mode: str = "train") -> List[Dict]:
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
        return tree_dicts
    else:
        tree_dicts = get_tree_dicts(train_dirs[(val_dir - 1)])
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


def register_train_data(train_location, name="tree", val_fold=1):
    for d in ["train", "val"]:
        DatasetCatalog.register(
            name + "_" + d,
            lambda d=d: combine_dicts(train_location, val_fold, d))
        MetadataCatalog.get(name + "_" + d).set(thing_classes=["tree"])


def remove_registered_data(name="tree"):
    for d in ["train", "val"]:
        DatasetCatalog.remove(name + "_" + d)
        MetadataCatalog.remove(name + "_" + d)


def register_test_data(test_location, name="tree"):
    d = "test"
    DatasetCatalog.register(name + "_" + d,
                            lambda d=d: get_tree_dicts(test_location))
    MetadataCatalog.get(name + "_" + d).set(thing_classes=["tree"])


def load_json_arr(json_path):
    lines = []
    with open(json_path, "r") as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


def setup_cfg(
    base_model="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
    trains=("trees_train",),
    tests=("trees_val",),
    update_model=None,
    workers=2,
    ims_per_batch=1,
    base_lr=0.0003,
    max_iter=1000,
    num_classes=1,
    eval_period=100,
    out_dir="/content/drive/Shareddrives/detectree2/train_outputs"):
  """
  To set up config object
  """
  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file(base_model))
  cfg.DATASETS.TRAIN = trains 
  cfg.DATASETS.TEST = tests
  cfg.DATALOADER.NUM_WORKERS = workers
  cfg.OUTPUT_DIR = out_dir
  cfg.SOLVER.IMS_PER_BATCH = 2
  cfg.SOLVER.GAMMA = 0.1
  cfg.MODEL.BACKBONE.FREEZE_AT = 3
  cfg.SOLVER.WARMUP_ITERS = 120
  cfg.SOLVER.MOMENTUM = 0.9
  cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 128
  cfg.SOLVER.WEIGHT_DECAY = 0
  cfg.SOLVER.BASE_LR = 0.001
  cfg.betas = (0.9, 0.999)
  os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
  if update_model is not None:
    cfg.MODEL.WEIGHTS = update_model # DOESN'T WORK
  else:
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(base_model)
  
  cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
  cfg.SOLVER.BASE_LR = base_lr
  cfg.SOLVER.MAX_ITER = max_iter
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
  cfg.TEST.EVAL_PERIOD = eval_period
  cfg.MODEL.BACKBONE.FREEZE_AT = 2
  cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = 'diou'
  return cfg

def predictions_on_data(directory=None,
                        predictor=DefaultTrainer,
                        trees_metadata=None,
                        save=True,
                        scale=1,
                        geos_exist=True,
                        num_predictions=0):
    """
    Prediction produced from a test folder and outputted to predictions folder
    """

    test_location = directory + "test"
    pred_dir = directory + "predictions"

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
        )    # remove the colors of unsegmented pixels
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        image = cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
        display(Image.fromarray(image))

        # Creating the file name of the output file
        file_name_path = d["file_name"]
        # Strips off all slashes so just final file name left
        file_name = os.path.basename(os.path.normpath(file_name_path))
        file_name = file_name.replace("png", "json")

        output_file = pred_dir + "/Prediction_" + file_name
        print(output_file)

        if save:
            # Converting the predictions to json files and saving them in the specfied output file.
            evaluations = instances_to_coco_json(outputs["instances"].to("cpu"),
                                                 d["file_name"])
            with open(output_file, "w") as dest:
                json.dump(evaluations, dest)


if __name__ == "__main__":
    train_location = "/content/drive/Shareddrives/detectree2/data/Paracou/tiles/train/"
    register_train_data(train_location, "Paracou",
                        1)    # folder, name, validation fold

    name = "Paracou2019"
    train_location = "/content/drive/Shareddrives/detectree2/data/Paracou/tiles2019/train/"
    dataset_dicts = combine_dicts(train_location, 1)
    trees_metadata = MetadataCatalog.get(name + "_train")
    #dataset_dicts = get_tree_dicts("./")
    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1],
                                metadata=trees_metadata,
                                scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        image = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
        display(Image.fromarray(image))
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
    cfg = setup_cfg(model,
                    trains,
                    tests,
                    eval_period=100,
                    max_iter=3000,
                    out_dir=out_dir)
    trainer = MyTrainer(cfg, patience=4)
    trainer.resume_or_load(resume=False)
    trainer.train()
