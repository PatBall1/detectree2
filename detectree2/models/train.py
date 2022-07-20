import os
import json
import logging
import time
import datetime
import cv2
import random
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.logger import log_every_n_seconds, setup_logger
import detectron2.utils.comm as comm
from detectron2.structures import BoxMode
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.data import (
    MetadataCatalog,
    DatasetCatalog,
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
)
import detectron2.data.transforms as T
from IPython.display import display, clear_output



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

  def __init__(self, eval_period, model, data_loader):
    self._model = model
    self._period = eval_period
    self._data_loader = data_loader

  def _do_loss_eval(self):
    """Copying inference_on_dataset from evaluator.py

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
    self.trainer.storage.put_scalar("validation_loss", mean_loss)
    comm.synchronize()

    return losses

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

  def after_step(self):
    next_iter = self.trainer.iter + 1
    is_final = next_iter == self.trainer.max_iter
    if is_final or (self._period > 0 and next_iter % self._period == 0):
      self._do_loss_eval()
    self.trainer.storage.put_scalars(timetest=12)


# See https://jss367.github.io/data-augmentation-in-detectron2.html for data augmentation advice
class MyTrainer(DefaultTrainer):
  """_summary_

    Args:
        DefaultTrainer (_type_): _description_

    Returns:
        _type_: _description_
    """

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    if output_folder is None:
      os.makedirs("eval_2", exist_ok=True)
      output_folder = "eval_2"
    return COCOEvaluator(dataset_name, cfg, True, output_folder)

  def build_hooks(self):
    hooks = super().build_hooks()
    hooks.insert(
        -1,
        LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0],
                                        DatasetMapper(self.cfg, True)),
        ),
    )
    return hooks

  def build_train_loader(cls, cfg):
    """_summary_

        Args:
            cfg (_type_): _description_

        Returns:
            _type_: _description_
        """
    return build_detection_train_loader(
        cfg,
        mapper=DatasetMapper(
            cfg,
            is_train=True,
            augmentations=[
                T.Resize((800, 800)),
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


def get_tree_dicts(directory, classes=None):
  """
    directory points to files
    classes signifies which column (if any) corresponds to the class labels
  """
  # filepath = '/content/drive/MyDrive/forestseg/paracou_data/Panayiotis_Outputs/220303_AllSpLabelled.gpkg'
  # datagpd = gpd.read_file(filepath)
  # List_Genus = datagpd.Genus_Species.to_list()
  # Genus_Species_UniqueList = list(set(List_Genus))

  #
  if classes is not None:
    # list_of_classes = crowns[variable].unique().tolist()
    # list_of_classes = ['Pradosia_cochlearia','Eperua_falcata','Dicorynia_guianensis','Eschweilera_sagotiana','Eperua_grandiflora','Symphonia_sp.1','Sextonia_rubra','Vouacapoua_americana','Sterculia_pruriens','Tapura_capitulifera','Pouteria_eugeniifolia','Recordoxylon_speciosum','Chrysophyllum_prieurii','Platonia_insignis','Chrysophyllum_pomiferum','Parkia_nitida','Goupia_glabra','Carapa_surinamensis','Licania_alba','Bocoa_prouacensis','Lueheopsis_rugosa']
    list_of_classes = ["CIRAD", "CNES", "INRA"]
    classes = list_of_classes
  else:
    classes = ["tree"]
  # classes = Genus_Species_UniqueList #['tree'] # genus_species list
  dataset_dicts = []
  #for root, dirs, files in os.walk(train_location):
  #    for file in files:
  #        if file.endswith(".geojson"):
  #            print(os.path.join(root, file))

  for filename in [
      file for file in os.listdir(directory) if file.endswith(".geojson")
  ]:
    json_file = os.path.join(directory, filename)
    with open(json_file) as f:
      img_anns = json.load(f)

    record = {}

    filename = os.path.join(directory, img_anns["imagePath"])
    # Make sure we have the correct height and width
    height, width = cv2.imread(filename).shape[:2]

    record["file_name"] = filename
    record["height"] = height
    record["width"] = width
    record["image_id"] = filename[0:400]
    #print(filename[0:400])

    objs = []
    for features in img_anns["features"]:
      anno = features["geometry"]
      # pdb.set_trace()
      # GenusSpecies = features['properties']['Genus_Species']
      # print("##### HERE IS AN ANNO #####", anno)...weirdly sometimes (but not always) have to make 1000 into a np.array
      px = [a[0] for a in anno["coordinates"][0]]
      py = [np.array(height) - a[1] for a in anno["coordinates"][0]]
      # print("### HERE IS PY ###", py)
      poly = [(x, y) for x, y in zip(px, py)]
      poly = [p for x in poly for p in x]
      # print("#### HERE ARE SOME POLYS #####", poly)
      if classes != ['tree']:
        obj = {
            "bbox": [np.min(px), np.min(py),
                     np.max(px), np.max(py)],
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": [poly],
            "category_id": classes.index(features["properties"]["PlotOrg"]
                                        ),    # id
        # "category_id": 0,  #id
            "iscrowd": 0,
        }
      else:
        obj = {
            "bbox": [np.min(px), np.min(py),
                     np.max(px), np.max(py)],
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


def combine_dicts(folder, val_folder, mode='train'):
  """
  function to join tree dicts from different directories
  """
  train_dirs = [os.path.join(folder, file) for file in os.listdir(folder)]
  if mode == 'train':
    del train_dirs[(val_folder - 1)]
    tree_dicts = []
    for d in train_dirs:
      tree_dicts = tree_dicts + get_tree_dicts(d)
    return tree_dicts
  else:
    tree_dicts = get_tree_dicts(train_dirs[(val_folder - 1)])
    return tree_dicts


def register_train_data(train_location, name= "tree", val_fold=1):
  for d in ["train", "val"]:
    DatasetCatalog.register(name +"_" + d,
                            lambda d=d: combine_dicts(train_location, val_fold, d))
    MetadataCatalog.get(name +"_" + d).set(thing_classes=['tree'])

def remove_registered_data(name= "tree"):
  for d in ['train', 'val']:
    DatasetCatalog.remove(name +"_" + d)
    MetadataCatalog.remove(name +"_" + d)

def register_test_data(test_location, name= "tree"):
  d="test"
  DatasetCatalog.register(name +"_" + d,
                          lambda d=d: get_tree_dicts(test_location))
  MetadataCatalog.get(name +"_" + d).set(thing_classes=['tree'])


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
    ims_per_batch=2,
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
  os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
  if update_model is not None:
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(update_model)
  else:
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(base_model)
  
  cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
  cfg.SOLVER.BASE_LR = base_lr
  cfg.SOLVER.MAX_ITER = max_iter
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
  cfg.TEST.EVAL_PERIOD = eval_period
  return cfg


def get_filenames(directory):
  """
  Used to get the file names if no geojson is present, i.e allows for predictions
  where no delinations have been manually produced
  """
  dataset_dicts = []
  for filename in [file for file in os.listdir(directory)]:
    file = {}
    filename = os.path.join(directory, filename)
    file["file_name"] = filename

    dataset_dicts.append(file)
  return dataset_dicts


def predictions_on_data(
  directory = None,
  predictor = DefaultTrainer,
  trees_metadata = None,
  save = True,
  scale = 1,
  geos_exist = True,
  num_predictions = 0
  ):
  """
  Prediction produced from a test folder and outputted to predictions folder
  """

  test_location = directory + "test"
  pred_dir = directory + "predictions"

  Path(pred_dir).mkdir(parents=True, exist_ok=True)

  if geos_exist:
    dataset_dicts= get_tree_dicts(test_location)
  else:
    dataset_dicts = get_filenames(test_location)
  
  # Works out if all items in folder should be predicted on
  if num_predictions == 0:
    num_to_pred = len(dataset_dicts)
  else:
    num_to_pred = num_predictions

  for d in random.sample(dataset_dicts,num_to_pred):
    img = cv2.imread(d["file_name"])
    # cv2_imshow(img)
    outputs = predictor(img)
    v = Visualizer(img[:, :, ::-1], metadata=trees_metadata, scale=scale, instance_mode=ColorMode.SEGMENTATION)   # remove the colors of unsegmented pixels
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    image = cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
    display(Image.fromarray(image))

    ### Creating the file name of the output file
    file_name_path = d["file_name"]
    file_name = os.path.basename(os.path.normpath(file_name_path))  #Strips off all slashes so just final file name left
    file_name = file_name.replace("png","json")
    
    output_file = pred_dir + "/Prediction_" + file_name
    print(output_file)

    if save: 
      ## Converting the predictions to json files and saving them in the specfied output file.
      evaluations= instances_to_coco_json(outputs["instances"].to("cpu"),d["file_name"])
      with open(output_file, "w") as dest:
        json.dump(evaluations,dest)


    
if __name__ == "__main__":
  train_location = "/content/drive/Shareddrives/detectree2/data/Paracou/tiles/train/"
  register_train_data(train_location, "Paracou", 1) # folder, name, validation fold

  name = "Paracou2019"
  train_location = "/content/drive/Shareddrives/detectree2/data/Paracou/tiles2019/train/"
  dataset_dicts = combine_dicts(train_location, 1)
  trees_metadata = MetadataCatalog.get(name + "_train")
  #dataset_dicts = get_tree_dicts("./")
  for d in dataset_dicts:
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=trees_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    image = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
    display(Image.fromarray(image))
  # Set the base (pre-trained) model from the detectron2 model_zoo
  model = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
  # Set the names of the registered train and test sets
  # pretrained model?
  # trained_model = "/content/drive/Shareddrives/detectree2/models/220629_ParacouSepilokDanum_JB.pth"
  trains = ("Paracou_train", "Paracou2019_train", "ParacouUAV_train", "Danum_train", "SepilokEast_train", "SepilokWest_train")
  tests = ("Paracou_val", "Paracou2019_val", "ParacouUAV_val", "Danum_val", "SepilokEast_val", "SepilokWest_val")
  out_dir = "/content/drive/Shareddrives/detectree2/220703_train_outputs"

  cfg = setup_cfg(model, trains, tests, eval_period=100, max_iter=3000, out_dir=out_dir) # update_model arg can be used to load in trained  model
  trainer = MyTrainer(cfg) 
  trainer.resume_or_load(resume=False)
  trainer.train()

