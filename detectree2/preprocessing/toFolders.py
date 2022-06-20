import os
import glob
import numpy as np
import random
import shutil
from pathlib import Path

random.seed(10)
Path("./data/paracou/train").mkdir(parents=True, exist_ok=True)
# Path("./data/val").mkdir(parents=True, exist_ok=True)
Path("./data/paracou/test").mkdir(parents=True, exist_ok=True)

# First split between train and test
split = np.array([4, 1])
summed = np.sum(split)
percs = 100 * split / summed
percs = np.cumsum(percs)

filenames = glob.glob("./output/Paracou/*.png")
jsonnames = glob.glob("./output/Paracou/*.geojson")

stemname = Path(filenames[0]).stem.split("_", 1)[0]

indices = [item.split("_", 1)[-1].split(".", 1)[0] for item in filenames]

num = list(range(0, len(indices)))
random.shuffle(num)

# glob.glob("./data/*" + indices[0] + ".json")

for i in range(0, len(indices)):
    print(i)
    if num[i] < np.percentile(num, percs[0]):
        shutil.copy(filenames[i], "./data/paracou/train/")
        shutil.copy("./output/Paracou/" + stemname + "_" + indices[i] + ".geojson", "./data/paracou/train/")
    # elif num[i] < np.percentile(num, percs[1]):
    #    shutil.copy(filenames[i], "./data/val/")
    #    shutil.copy("./data/" + stemname + "_" + indices[i] + ".json", "./data/val/")
    else:
        shutil.copy(filenames[i], "./data/paracou/test/")
        shutil.copy("./output/Paracou/" + stemname + "_" + indices[i] + ".geojson", "./data/paracou/test/")


# MAD ignore folds for a second. 

# folds = 3

# filenames = glob.glob("./data/paracou/train/*.png")
# jsonnames = glob.glob("./data/paracou/train/*.geojson")

# stemname = Path(filenames[0]).stem.split("_", 1)[0]

# indices = [item.split("_", 1)[-1].split(".", 1)[0] for item in filenames]

# random.shuffle(indices)
# ind_split = np.array_split(indices, folds)

# for i in range(0, folds):
#     Path("./data/paracou/train/fold_" + str(i + 1) + "/").mkdir(parents=True, exist_ok=True)
#     for ind in ind_split[i]:
#         print(ind)
#         shutil.move(
#             "./data/paracou/train/" + stemname + "_" + ind + ".png",
#             "./data/paracou/train/fold_" + str(i + 1) + "/",
#         )
#         shutil.move(
#             "./data/paracou/train/" + stemname + "_" + ind + ".geojson",
#             "./data/paracou/train/fold_" + str(i + 1) + "/",
#         )

