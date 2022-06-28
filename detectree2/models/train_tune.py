# Script to train while logging parameters on WandB
import wandb
import detectron2
from detectron2.utils.logger import setup_logger
import pandas as pd
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import json
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

if __name__ == "__main__":
  setup_logger()
  wandb.login()
