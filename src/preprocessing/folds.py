import os
import glob
import numpy as np
import random
import shutil
from pathlib import Path
import math

Path("./data/train/train/").mkdir(parents=True, exist_ok=True)
# Path("./data/val").mkdir(parents=True, exist_ok=True)
Path("./data/train/val/").mkdir(parents=True, exist_ok=True)

valfold = 2

folders = glob.glob("./data/train/fold_*")

for i in range(0, len(folders)):
    if i == (valfold - 1):
        files = glob.glob(folders[i] + "/*")
        for file in files:
            shutil.copy(file, "./data/train/val/")
    else:
        files = glob.glob(folders[i] + "/*")
        for file in files:
            shutil.copy(file, "./data/train/train/")

# Need to reset train / val folder for k-fold cross validation
