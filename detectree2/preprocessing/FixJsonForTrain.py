import glob
import json
import os
import random
import re
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

entries = os.listdir(
    "/content/drive/MyDrive/forestseg/paracou_data/Panayiotis_Outputs/ForTrainingSegmentation/train"
)
print(entries)

for i in entries:
    if "geojson" in i:
        filename = (
            "/content/drive/MyDrive/forestseg/paracou_data/Panayiotis_Outputs/ForTrainingSegmentation/train/"
            + i
        )
        print(i)
        print(filename)
        j = i[:-8]
        print(j)
        k = j + ".png"
        Dict = {"imagePath": k}
        print(Dict)
        with open(filename, "r") as f:
            data = json.load(f)
            data.update(Dict)
        with open(filename, "w") as f:
            json.dump(data, f)

entries = os.listdir(
    "/content/drive/MyDrive/forestseg/paracou_data/Panayiotis_Outputs/ForTrainingSegmentation/test"
)
print(entries)

for i in entries:
    if "geojson" in i:
        filename = (
            "/content/drive/MyDrive/forestseg/paracou_data/Panayiotis_Outputs/ForTrainingSegmentation/test/"
            + i
        )
        print(i)
        print(filename)
        j = i[:-8]
        print(j)
        k = j + ".png"
        Dict = {"imagePath": k}
        print(Dict)
        with open(filename, "r") as f:
            data = json.load(f)
            data.update(Dict)
        with open(filename, "w") as f:
            json.dump(data, f)
