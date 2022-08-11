# Script to train while logging parameters on WandB
import json
import os
import random

import cv2
import detectron2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from PIL import Image

if __name__ == "__main__":
    setup_logger()
    wandb.login()
