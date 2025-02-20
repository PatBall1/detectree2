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
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import cv2
import detectron2.data.transforms as T  # noqa:N812
import detectron2.utils.comm as comm
import numpy as np
import rasterio
import torch
import torch.nn as nn
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
from detectron2.data import detection_utils as utils
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.structures import BoxMode
from detectron2.utils.events import get_event_storage  # noqa:F401
from detectron2.utils.events import EventStorage
from detectron2.utils.logger import log_every_n_seconds
from detectron2.utils.visualizer import ColorMode, Visualizer

from detectree2.preprocessing.tiling import load_class_mapping


class FlexibleDatasetMapper(DatasetMapper):
    """
    A flexible dataset mapper that extends the standard DatasetMapper to handle
    multi-band images and custom augmentations.

    This class is designed to work with datasets that may contain images with
    more than three channels (e.g., multispectral images) and allows for custom
    augmentations to be applied. It also handles semantic segmentation data if
    provided in the dataset.

    Args:
        cfg (CfgNode): Configuration object containing dataset and model configurations.
        is_train (bool): Flag indicating whether the mapper is being used for training. Default is True.
        augmentations (list, optional): List of augmentations to be applied. Default is an empty list.

    Attributes:
        cfg (CfgNode): Stores the configuration object for later use.
        is_train (bool): Indicates whether the mapper is in training mode.
        logger (Logger): Logger instance for logging messages.
    """

    def __init__(self, cfg, is_train=True, augmentations=None):
        if augmentations is None:
            augmentations = []

        # Initialize the base DatasetMapper class with provided parameters
        super().__init__(is_train=is_train,
                         augmentations=augmentations,
                         image_format=cfg.INPUT.FORMAT,
                         use_instance_mask=cfg.MODEL.MASK_ON,
                         use_keypoint=cfg.MODEL.KEYPOINT_ON,
                         instance_mask_format=cfg.INPUT.MASK_FORMAT,
                         keypoint_hflip_indices=None,
                         precomputed_proposal_topk=None,
                         recompute_boxes=False)
        self.cfg = cfg
        self.is_train = is_train
        self.logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        self.logger.info(f"[FlexibleDatasetMapper] Augmentations used in {mode}: {augmentations}")

    def __call__(self, dataset_dict):
        """
        Process a single dataset dictionary, applying the necessary transformations and augmentations.

        Args:
            dataset_dict (dict): A dictionary containing data for a single dataset item, including
                                 file names and metadata.

        Returns:
            dict: The processed dataset dictionary, or None if there was an error.
        """
        if dataset_dict is None:
            self.logger.warning("Received None for dataset_dict, skipping this entry.")
            return None

        if self.cfg.IMGMODE == "rgb":
            return super().__call__(dataset_dict)

        try:
            # Handle multi-band image loading using rasterio
            with rasterio.open(dataset_dict["file_name"]) as src:
                img = src.read()
                if img is None:
                    raise ValueError(f"Image data is None for file: {dataset_dict['file_name']}")
                # Transpose image dimensions to match expected format (H, W, C)
                img = np.transpose(img, (1, 2, 0)).astype("float32")

            # Size check similar to utils.check_image_size
            if img.shape[:2] != (dataset_dict.get("height"), dataset_dict.get("width")):
                self.logger.warning(
                    f"""Image size {img.shape[:2]} does not match expected size {(dataset_dict.get('height'),
                                                                                dataset_dict.get('width'))}.""")

            # Otherwise, handle custom multi-band logic
            aug_input = T.AugInput(img)
            transforms = self.augmentations(aug_input)  # Apply the augmentations
            img = aug_input.image

            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))

            # Handle semantic segmentation if present
            if "sem_seg_file_name" in dataset_dict:
                sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
                dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

            if not self.is_train:
                # If not in training mode, remove annotations and segmentation file names
                dataset_dict.pop("annotations", None)
                dataset_dict.pop("sem_seg_file_name", None)
                return dataset_dict

            if "annotations" in dataset_dict:
                # Apply the transformations to the annotations
                self._transform_annotations(dataset_dict, transforms, img.shape[:2])

            return dataset_dict

        except Exception as e:
            file_name = dataset_dict.get('file_name', 'unknown') if dataset_dict else 'unknown'
            self.logger.error(f"Error processing {file_name}: {e}")
            return None


class LossEvalHook(HookBase):
    """
    A custom hook for evaluating loss during training and managing model checkpoints based on evaluation metrics.

    This hook is designed to:
    - Perform inference on a dataset similarly to an Evaluator.
    - Calculate and log the loss metric during training.
    - Save the best model checkpoint based on a specified evaluation metric (e.g., AP50).
    - Implement early stopping if the evaluation metric does not improve over a specified number of evaluations.

    Attributes:
        _model: The model to evaluate.
        _period: Number of iterations between evaluations.
        _data_loader: The data loader used for evaluation.
        patience: Number of evaluation periods to wait before early stopping.
        iter: Tracks the number of evaluations since the last improvement in the evaluation metric.
        max_ap: The best evaluation metric (e.g., AP50) achieved during training.
        best_iter: The iteration at which the best evaluation metric was achieved.
    """

    def __init__(self, eval_period, model, data_loader, patience):
        """
        Initialize the LossEvalHook.

        Args:
            eval_period (int): The number of iterations between evaluations.
            model (torch.nn.Module): The model to evaluate.
            data_loader (torch.utils.data.DataLoader): The data loader for evaluation.
            patience (int): The number of evaluation periods to wait for improvement before early stopping.
        """
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
        self.patience = patience
        self.iter = 0
        self.max_ap = 0
        self.best_iter = 0

    def _do_loss_eval(self):
        """
        Perform inference on the dataset and calculate the average loss.

        This method is adapted from `inference_on_dataset` in Detectron2's evaluator.
        It also calculates and logs the AP50 metric and updates the best model checkpoint if needed.

        Returns:
            list: A list of loss values for each batch in the dataset.
        """
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                # Reset the start time after the warm-up phase
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                # Log progress and estimated time remaining
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(idx + 1, total, seconds_per_img,
                                                                                    str(eta)),
                    n=5,
                )
            # Calculate loss for the current batch
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)

        mean_loss = np.mean(losses)

        # Calculate the average AP50 across datasets if multiple datasets are used for testing
        if len(self.trainer.cfg.DATASETS.TEST) > 1:
            APs = []
            for dataset in self.trainer.cfg.DATASETS.TEST:
                APs.append(self.trainer.test(self.trainer.cfg, self.trainer.model)[dataset]["segm"]["AP50"])
            AP = sum(APs) / len(APs)
        else:
            AP = self.trainer.test(self.trainer.cfg, self.trainer.model)["segm"]["AP50"]

        print("Av. segm AP50 =", AP)

        # Store the calculated loss and AP50 in the trainer's storage
        self.trainer.APs.append(AP)
        self.trainer.storage.put_scalar("validation_loss", mean_loss)
        self.trainer.storage.put_scalar("validation_ap", AP)
        comm.synchronize()

        return losses

    def _get_loss(self, data):
        """
        Compute the loss for a given batch of data.

        Args:
            data (dict): A batch of input data.

        Returns:
            float: The total loss for the batch.
        """
        metrics_dict = self._model(data)
        # Detach and move to CPU for logging
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def after_step(self):
        """
        Hook to be called after each training iteration to evaluate the model and manage checkpoints.

        - Evaluates the model at regular intervals.
        - Saves the best model checkpoint based on the AP50 metric.
        - Implements early stopping if the AP50 does not improve after a set number of evaluations.
        """
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
            # Check if the current AP50 is the best so far
            if self.max_ap < self.trainer.APs[-1]:
                self.iter = 0
                self.max_ap = self.trainer.APs[-1]
                # Save the current best model
                self.trainer.checkpointer.save("model_" + str(len(self.trainer.APs)))
                self.best_iter = self.trainer.iter
            else:
                self.iter += 1
        if self.iter == self.patience:
            # Early stopping condition met
            self.trainer.early_stop = True
            print("Early stopping occurs in iter {}, max ap is {}".format(self.best_iter, self.max_ap))
        self.trainer.storage.put_scalars(timetest=12)

    def after_train(self):
        """
        Hook to be called after training is complete to load the best model checkpoint based on AP50.

        - Selects and loads the model checkpoint with the best AP50.
        """
        if not self.trainer.APs:
            print("No APs were recorded during training. Skipping model selection.")
            return
        # Select the model with the best AP50
        index = self.trainer.APs.index(max(self.trainer.APs)) + 1
        # Error handling for checkpoint loading, with a sleep to ensure file availability in CI environments
        time.sleep(15)
        self.trainer.checkpointer.load(self.trainer.cfg.OUTPUT_DIR + '/model_' + str(index) + '.pth')


# See https://jss367.github.io/data-augmentation-in-detectron2.html for data augmentation advice
class MyTrainer(DefaultTrainer):
    """
    Custom Trainer class that extends the DefaultTrainer.

    This trainer adds flexibility for handling different image types (e.g., RGB and multi-band images)
    and custom training behavior, such as early stopping and specialized data augmentation strategies.

    Args:
        cfg (CfgNode): Configuration object containing the model and dataset configurations.
        patience (int): Number of evaluation periods to wait for improvement before early stopping.
    """

    def __init__(self, cfg, patience):  # noqa: D107
        self.patience = patience
        super().__init__(cfg)

    def train(self):
        """
        Run the training loop.

        This method overrides the DefaultTrainer's train method to include early stopping and
        custom logging of Average Precision (AP) metrics.

        Returns:
            OrderedDict: Results from evaluation, if evaluation is enabled. Otherwise, None.
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
        # Verify the results if testing is enabled and this is the main process
        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
            assert hasattr(self, "_last_eval_results"), "No evaluation results obtained during training!"
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """
        self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            self.start_iter = self.iter + 1

        if self.cfg.MODEL.WEIGHTS:
            checkpoint = torch.tensor(
                self.checkpointer._load_file(
                    self.checkpointer.path_manager.get_local_path(
                        urlparse(self.cfg.MODEL.WEIGHTS)._replace(
                            query="").geturl()))['model']['backbone.bottom_up.stem.conv1.weight']).to(
                                self.model.backbone.bottom_up.stem.conv1.weight.device)
            input_channels_in_checkpoint = checkpoint.shape[1]
            input_channels_in_model = self.model.backbone.bottom_up.stem.conv1.weight.shape[1]
            if input_channels_in_checkpoint != input_channels_in_model:
                logger = logging.getLogger("detectree2")
                if input_channels_in_checkpoint != 3:
                    logger.warning(
                        "Input channel modification only works if checkpoint was trained on RGB images (3 channels). The first three channels will be copied and then repeated in the model."
                    )
                logger.warning(
                    "Mismatch in input channels in checkpoint and model, meaning fvcommon would not have been able to automatically load them. Adjusting weights for 'backbone.bottom_up.stem.conv1.weight' manually."
                )
                with torch.no_grad():
                    self.model.backbone.bottom_up.stem.conv1.weight[:, :
                                                                    input_channels_in_checkpoint] = checkpoint[:, :
                                                                                                               input_channels_in_checkpoint]
                multiply_conv1_weights(self.model)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Build the evaluator for the model.

        Args:
            cfg (CfgNode): Configuration object.
            dataset_name (str): Name of the dataset to evaluate.
            output_folder (str, optional): Directory to save evaluation results. Defaults to "eval".

        Returns:
            COCOEvaluator: An evaluator for COCO-style datasets.
        """
        if output_folder is None:
            os.makedirs("eval", exist_ok=True)
            output_folder = "eval"
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        """
        Build the training hooks, including the custom LossEvalHook.

        This method adds a custom hook for evaluating the model's loss during training, with support for
        early stopping based on the AP50 metric.

        Returns:
            list: A list of hooks to be used during training.
        """
        hooks = super().build_hooks()

        # Determine the appropriate resize strategy based on the configuration
        if self.cfg.RESIZE == "random":
            size = None
            # Attempt to determine the image size from the training dataset
            for i, datas in enumerate(DatasetCatalog.get(self.cfg.DATASETS.TRAIN[0])):
                location = datas['file_name']
                try:
                    # Attempt to read the image with OpenCV (for RGB images)
                    img = cv2.imread(location)
                    if img is not None:
                        size = img.shape[0]
                    else:
                        # Fall back to rasterio for multi-band images
                        with rasterio.open(location) as src:
                            size = src.height  # Assuming square images
                except Exception as e:
                    # Handle any errors that occur during loading
                    print(f"Error loading image {location}: {e}")
                    continue
                break
            # Define augmentation based on the determined size
            augmentations = [T.ResizeShortestEdge([size, size], size + 300)]
        else:
            # Use fixed size resizing as a default
            augmentations = [T.ResizeShortestEdge([1000, 1000], 1333)]

        # Insert the custom LossEvalHook before the last hook (typically the evaluation hook)
        hooks.insert(
            -1,
            LossEvalHook(
                self.cfg.TEST.EVAL_PERIOD,
                self.model,
                build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST,
                                            FlexibleDatasetMapper(self.cfg, True, augmentations=augmentations)),
                self.patience,
            ),
        )
        return hooks

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Build the training data loader with support for custom augmentations and image types.

        This method configures the data loader to apply specific augmentations depending on the image mode
        (RGB or multi-band) and resize strategy defined in the configuration.

        Args:
            cfg (CfgNode): Configuration object.

        Returns:
            DataLoader: A data loader for the training dataset.
        """

        # Define basic augmentations including rotation and flipping
        augmentations = [
            T.RandomRotation(angle=[0, 360], expand=False),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        ]

        # Additional augmentations for RGB images
        if cfg.IMGMODE == "rgb":
            augmentations.extend([
                T.RandomBrightness(0.7, 1.5),
                T.RandomLighting(0.7),
                T.RandomContrast(0.6, 1.3),
                T.RandomSaturation(0.8, 1.4)
            ])

        # Add resizing augmentations based on the resize strategy
        if cfg.RESIZE == "fixed":
            augmentations.append(T.ResizeShortestEdge([1000, 1000], 1333))
        elif cfg.RESIZE == "random":
            size = None
            for i, datas in enumerate(DatasetCatalog.get(cfg.DATASETS.TRAIN[0])):
                location = datas['file_name']
                try:
                    # Try to read with cv2 (for RGB images)
                    img = cv2.imread(location)
                    if img is not None:
                        size = img.shape[0]
                    else:
                        # Fall back to rasterio for multi-band images
                        with rasterio.open(location) as src:
                            size = src.height  # Assuming square images
                except Exception as e:
                    # Handle any errors that occur during loading
                    print(f"Error loading image {location}: {e}")
                    continue
                break

            if size:
                print("ADD RANDOM RESIZE WITH SIZE = ", size)
                augmentations.append(T.ResizeScale(0.6, 1.4, size, size))
            else:
                raise ValueError("Failed to determine image size for random resize")
        elif cfg.RESIZE == "rand_fixed":
            augmentations.append(T.ResizeScale(0.6, 1.4, 1000, 1000))

        return build_detection_train_loader(
            cfg,
            mapper=FlexibleDatasetMapper(
                cfg,
                is_train=True,
                augmentations=augmentations,
            ),
        )

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Build the test data loader.

        This method configures the data loader for evaluation, using the FlexibleDatasetMapper
        to handle custom augmentations and image types.

        Args:
            cfg (CfgNode): Configuration object.
            dataset_name (str): Name of the dataset to load for testing.

        Returns:
            DataLoader: A data loader for the test dataset.
        """
        return build_detection_test_loader(cfg, dataset_name, mapper=FlexibleDatasetMapper(cfg, is_train=False))


def get_tree_dicts(directory: str, class_mapping: Optional[Dict[str, int]] = None) -> List[Dict[str, Any]]:
    """Get the tree dictionaries.

    Args:
        directory: Path to directory
        classes: List of classes to include
        classes_at: Signifies which column (if any) corresponds to the class labels

    Returns:
        List of dictionaries corresponding to segmentations of trees. Each dictionary includes
        bounding box around tree and points tracing a polygon around a tree.
    """

    dataset_dicts = []

    for filename in [file for file in os.listdir(directory) if file.endswith(".geojson")]:
        json_file = os.path.join(directory, filename)
        with open(json_file) as f:
            img_anns = json.load(f)

        record: Dict[str, Any] = {}
        filename = img_anns["imagePath"]

        # Make sure we have the correct height and width
        # If image path ends in .png use cv2 to get height and width else if image path ends in .tif use rasterio
        if filename.endswith(".png"):
            height, width = cv2.imread(filename).shape[:2]
        elif filename.endswith(".tif"):
            with rasterio.open(filename) as src:
                height, width = src.shape

        record["file_name"] = filename
        record["height"] = height
        record["width"] = width
        record["image_id"] = filename[0:400]
        record["annotations"] = {}
        # print(filename[0:400])

        objs = []
        for features in img_anns["features"]:
            anno = features["geometry"]
            px = [a[0] for a in anno["coordinates"][0]]
            py = [np.array(height) - a[1] for a in anno["coordinates"][0]]
            poly = [(x, y) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            # If class mapping is provided, use it; otherwise, default to "tree"
            if class_mapping:
                category_id = class_mapping[features["properties"]["status"]]
            else:
                category_id = 0  # Default to "tree" if no class mapping is provided

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": category_id,
                "iscrowd": 0,
            }

            objs.append(obj)

        record["annotations"] = objs if objs else []
        dataset_dicts.append(record)

    return dataset_dicts


def combine_dicts(root_dir: str,
                  val_dir: int,
                  mode: str = "train",
                  class_mapping: Optional[Dict[str, int]] = None) -> List[Dict[str, Any]]:
    """
    Combine dictionaries from different directories based on the specified mode.

    This function aggregates tree dictionaries from multiple directories within a root directory.
    Depending on the mode, it either combines dictionaries from all directories,
    all except a specified validation directory, or only from the validation directory.

    Args:
        root_dir (str): The root directory containing subdirectories with tree dictionaries.
        val_dir (int): The index (1-based) of the validation directory to exclude or use depending on the mode.
        mode (str, optional): The mode of operation. Can be "train", "val", or "full".
                              "train" excludes the validation directory,
                              "val" includes only the validation directory,
                              and "full" includes all directories. Defaults to "train".
        class_mapping: A dictionary mapping class labels to category indices (optional).

    Returns:
        List of combined dictionaries from the specified directories.
    """
    # Get the list of directories within the root directory
    train_dirs = [
        os.path.join(root_dir, dir) for dir in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, dir))
    ]
    # Handle the different modes for combining dictionaries
    if mode == "train":
        # Exclude the validation directory from the list of directories
        del train_dirs[(val_dir - 1)]
        tree_dicts = []
        for d in train_dirs:
            # Combine dictionaries from all directories except the validation directory
            tree_dicts += get_tree_dicts(d, class_mapping=class_mapping)
    elif mode == "val":
        # Use only the validation directory
        tree_dicts = get_tree_dicts(train_dirs[(val_dir - 1)], class_mapping=class_mapping)
    elif mode == "full":
        # Combine dictionaries from all directories, including the validation directory
        tree_dicts = []
        for d in train_dirs:
            tree_dicts += get_tree_dicts(d, class_mapping=class_mapping)
    return tree_dicts


def get_filenames(directory: str):
    """Get the file names from the directory, handling both RGB (.png) and multispectral (.tif) images.

    Args:
        directory (str): Directory of images to be predicted on.

    Returns:
        tuple: A tuple containing:
            - dataset_dicts (list): List of dictionaries with 'file_name' keys.
            - mode (str): 'rgb' if .png files are used, 'ms' if .tif files are used.
    """
    dataset_dicts = []

    # Get list of .png and .tif files
    png_files = glob.glob(os.path.join(directory, "*.png"))
    tif_files = glob.glob(os.path.join(directory, "*.tif"))

    if png_files and tif_files:
        # Both .png and .tif files are present, select only .png files
        files = png_files
        mode = "rgb"
    elif png_files:
        # Only .png files are present
        files = png_files
        mode = "rgb"
    elif tif_files:
        # Only .tif files are present
        files = tif_files
        mode = "ms"
    else:
        # No image files found
        files = []
        mode = None

    for filename in files:
        file = {}
        file["file_name"] = filename
        dataset_dicts.append(file)
    return dataset_dicts, mode


def register_train_data(train_location, name: str = "tree", val_fold=None, class_mapping_file=None):
    """Register data for training and (optionally) validation.

    Args:
        train_location: Directory containing training folds.
        name: Name to register the dataset.
        val_fold: Validation fold index (optional).
        class_mapping_file: Path to the class mapping file (json or pickle).
    """
    # Load the class mapping from file if provided
    class_mapping = None
    if class_mapping_file:
        class_mapping = load_class_mapping(class_mapping_file)
        thing_classes = list(class_mapping.keys())  # Convert dictionary to list of class names
        print(f"Class mapping loaded: {class_mapping}")  # Debugging step
    else:
        thing_classes = ["tree"]

    if val_fold is not None:
        for d in ["train", "val"]:
            DatasetCatalog.register(name + "_" + d,
                                    lambda d=d: combine_dicts(train_location, val_fold, d, class_mapping=class_mapping))
            MetadataCatalog.get(name + "_" + d).set(thing_classes=thing_classes)
    else:
        DatasetCatalog.register(name + "_" + "full",
                                lambda d=d: combine_dicts(train_location, 0, "full", class_mapping=class_mapping))
        MetadataCatalog.get(name + "_" + "full").set(thing_classes=thing_classes)


def get_classes(out_dir):
    """Function that will read the classes that are recorded during tiling.

    Args:
        out_dir: directory where classes.txt is located

    Returns:
        list of classes
    """
    list = []
    classes_txt = out_dir + 'classes.txt'
    # open file and read the content in a list
    with open(classes_txt, 'r') as fp:
        for line in fp:
            # remove linebreak from a current name
            # linebreak is the last character of each line
            x = line[:-1]
            # add current item to the list
            list.append(x)
    return (list)


def remove_registered_data(name="tree"):
    """Remove registered data from catalog.

    Args:
        name: string of named registered data
    """
    for d in ["train", "val"]:
        DatasetCatalog.remove(name + "_" + d)
        MetadataCatalog.remove(name + "_" + d)


def register_test_data(test_location, name="tree"):
    """Register data for testing.

    Args:
        test_location: directory containing test data
        name: string to name data
    """
    d = "test"

    class_mapping = None
    if class_mapping_file:
        class_mapping = load_class_mapping(class_mapping_file)
        thing_classes = list(class_mapping.keys())  # Convert dictionary to list of class names
        print(f"Class mapping loaded: {class_mapping}")  # Debugging step
    else:
        thing_classes = ["tree"]

    DatasetCatalog.register(name + "_" + d, lambda d=d: get_tree_dicts(test_location, class_mapping))
    MetadataCatalog.get(name + "_" + d).set(thing_classes=thing_classes)


def load_json_arr(json_path):
    """Load json array.

    Args:
        json_path: path to json file
    """
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
    base_lr=0.0003389,
    weight_decay=0.001,
    max_iter=1000,
    eval_period=100,
    out_dir="./train_outputs",
    resize="fixed",  # "fixed" or "random" or "rand_fixed"
    imgmode="rgb",
    num_bands=3,
    class_mapping_file=None,
):
    """Set up config object # noqa: D417.

    Args:
        base_model: base pre-trained model from detectron2 model_zoo
        trains: names of registered data to use for training
        tests: names of registered data to use for evaluating models
        update_model: updated pre-trained model from detectree2 model_garden
        workers: number of workers for dataloader
        ims_per_batch: number of images per batch
        gamma: gamma for learning rate scheduler
        backbone_freeze: backbone layer to freeze
        warm_iter: number of iterations for warmup
        momentum: momentum for optimizer
        batch_size_per_im: batch size per image
        base_lr: base learning rate
        weight_decay: weight decay for optimizer
        max_iter: maximum number of iterations
        num_classes: number of classes
        eval_period: number of iterations between evaluations
        out_dir: directory to save outputs
        resize: resize strategy for images
        imgmode: image mode (rgb or multispectral)
        num_bands: number of bands in the image
        class_mapping_file: path to class mapping file
    """

    # Load the class mapping if provided
    if class_mapping_file:
        class_mapping = load_class_mapping(class_mapping_file)
        num_classes = len(class_mapping)  # Set the number of classes based on the mapping
    else:
        num_classes = 1  # Default to 1 class if no mapping is provided

    # Validate the resize parameter
    if resize not in {"fixed", "random", "rand_fixed"}:
        raise ValueError(f"Invalid resize option '{resize}'. Must be 'fixed', 'random', or 'rand_fixed'.")

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
    cfg.SOLVER.WEIGHT_DECAY = weight_decay
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
    cfg.INPUT.MIN_SIZE_TRAIN = 1000
    cfg.IMGMODE = imgmode  # "rgb" or "ms" (multispectral)
    if num_bands > 3:
        # Adjust PIXEL_MEAN and PIXEL_STD for the number of bands
        default_pixel_mean = cfg.MODEL.PIXEL_MEAN
        default_pixel_std = cfg.MODEL.PIXEL_STD
        # Extend or truncate the PIXEL_MEAN and PIXEL_STD based on num_bands
        cfg.MODEL.PIXEL_MEAN = (default_pixel_mean * (num_bands // len(default_pixel_mean)) +
                                default_pixel_mean[:num_bands % len(default_pixel_mean)])
        cfg.MODEL.PIXEL_STD = (default_pixel_std * (num_bands // len(default_pixel_std)) +
                               default_pixel_std[:num_bands % len(default_pixel_std)])
    return cfg


def predictions_on_data(
    directory=None,
    predictor=DefaultPredictor,
    trees_metadata=None,
    save=True,
    scale=1,
    geos_exist=True,
    num_predictions=0,
):
    """Make predictions on test data and output them to the predictions folder.

    Args:
        directory (str): Directory containing test data.
        predictor (DefaultPredictor): The predictor object.
        trees_metadata: Metadata for trees.
        save (bool): Whether to save the predictions.
        scale (float): Scale of the image for visualization.
        geos_exist (bool): Determines if geojson files exist.
        num_predictions (int): Number of predictions to make.

    Returns:
        None
    """
    pred_dir = os.path.join(directory, "predictions")
    Path(pred_dir).mkdir(parents=True, exist_ok=True)

    test_location = os.path.join(directory, "test")

    if geos_exist:
        dataset_dicts = get_tree_dicts(test_location)
        if len(dataset_dicts) > 0:
            sample_file = dataset_dicts[0]["file_name"]
            _, mode = get_filenames(os.path.dirname(sample_file))
        else:
            mode = None
    else:
        dataset_dicts, mode = get_filenames(test_location)

    # Decide how many items to predict on
    num_to_pred = len(dataset_dicts) if num_predictions == 0 else num_predictions

    for d in random.sample(dataset_dicts, num_to_pred):
        file_name = d["file_name"]
        file_ext = os.path.splitext(file_name)[1].lower()
        if file_ext == ".png":
            # RGB image, read with cv2
            img = cv2.imread(file_name)
            if img is None:
                print(f"Failed to read image {file_name} with cv2.")
                continue
            # Convert BGR to RGB for visualization
            img_vis = img[:, :, ::-1]
        elif file_ext == ".tif":
            # Multispectral image, read with rasterio
            with rasterio.open(file_name) as src:
                img = src.read()
                # Transpose to match expected format (H, W, C)
                img = np.transpose(img, (1, 2, 0))
            # For visualization, convert to RGB if possible
            img_vis = img[:, :, :3] if img.shape[2] >= 3 else img
        else:
            print(f"Unsupported file extension {file_ext} for file {file_name}")
            continue

        outputs = predictor(img)
        v = Visualizer(
            img_vis,
            metadata=trees_metadata,
            scale=scale,
            instance_mode=ColorMode.SEGMENTATION,
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # Create the output file name
        file_name_only = os.path.basename(file_name)
        file_name_json = os.path.splitext(file_name_only)[0] + ".json"
        output_file = os.path.join(pred_dir, f"Prediction_{file_name_json}")

        if save:
            # Save predictions to JSON file
            evaluations = instances_to_coco_json(outputs["instances"].to("cpu"), file_name)
            with open(output_file, "w") as dest:
                json.dump(evaluations, dest)


def multiply_conv1_weights(model):
    """
    Modify the weights of the first convolutional layer (conv1) to accommodate a different number of input channels.

    This function adjusts the weights of the `conv1` layer in the model's backbone to support a custom number
    of input channels. It creates a new weight tensor with the desired number of input channels,
    and initializes it by repeating the weights of the original channels.

    Args:
        model (torch.nn.Module): The model containing the convolutional layer to modify.

    """
    with torch.no_grad():
        # Retrieve the original weights of the conv1 layer
        old_weights = model.backbone.bottom_up.stem.conv1.weight
        num_input_channels = model.backbone.bottom_up.stem.conv1.weight.shape[1]  # The number of input channels

        # Create a new weight tensor with the desired number of input channels
        # The shape is (out_channels, in_channels, height, width)
        new_weights = torch.zeros((old_weights.size(0), num_input_channels, *old_weights.shape[2:]))

        # Initialize the new weights by repeating the original weights across the new channels
        # This example repeats the first 3 channels if num_input_channels > 3
        for i in range(num_input_channels):
            new_weights[:, i, :, :] = old_weights[:, i % 3, :, :]

        # Create a new conv1 layer with the updated number of input channels
        model.backbone.bottom_up.stem.conv1 = nn.Conv2d(num_input_channels,
                                                        old_weights.size(0),
                                                        kernel_size=7,
                                                        stride=2,
                                                        padding=3,
                                                        bias=False)

        # Copy the modified weights into the new conv1 layer
        model.backbone.bottom_up.stem.conv1.weight.copy_(new_weights)


def get_latest_model_path(output_dir: str) -> str:
    """
    Find the model file with the highest index in the specified output directory.

    Args:
        output_dir (str): The directory where the model files are stored.

    Returns:
        str: The path to the model file with the highest index.
    """
    # Regular expression to match model files with the pattern "model_X.pth"
    model_pattern = re.compile(r"model_(\d+)\.pth")

    # List all files in the output directory
    files = os.listdir(output_dir)

    # Find all files that match the pattern and extract their indices
    model_files = []
    for f in files:
        match = model_pattern.search(f)
        if match:
            model_files.append((f, int(match.group(1))))

    if not model_files:
        raise FileNotFoundError(f"No model files found in the directory {output_dir}")

    # Sort the files by index in descending order and select the highest one
    latest_model_file = max(model_files, key=lambda x: x[1])[0]

    # Return the full path to the latest model file
    return os.path.join(output_dir, latest_model_file)


if __name__ == "__main__":
    # Define paths to training data and optional class mapping file
    train_location = "/path/to/your/train/location"
    class_mapping_file = "/path/to/your/class_to_idx.json"  # Optional, can be None

    # Register the training and validation datasets using the class mapping
    # If class_mapping_file is not provided, defaults to "tree"
    register_train_data(train_location, "MyDataset", val_fold=1, class_mapping_file=class_mapping_file)

    # Set up model configuration, using the class mapping to determine the number of classes
    cfg = setup_cfg(
        base_model="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
        trains=("MyDataset_train", ),
        tests=("MyDataset_val", ),
        max_iter=3000,
        out_dir="/path/to/output",
        class_mapping_file=class_mapping_file  # Optional
    )

    # Train the model
    trainer = MyTrainer(cfg, patience=4)
    trainer.resume_or_load(resume=False)
    trainer.train()
