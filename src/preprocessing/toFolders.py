import os
import glob
import numpy as np
import random
import shutil
from pathlib import Path

random.seed(10)
Path("./data/train").mkdir(parents=True, exist_ok=True)
# Path("./data/val").mkdir(parents=True, exist_ok=True)
Path("./data/test").mkdir(parents=True, exist_ok=True)

# First split between train and test
split = np.array([4, 1])
summed = np.sum(split)
percs = 100 * split / summed
percs = np.cumsum(percs)

filenames = glob.glob("./data/*.txt")
jsonnames = glob.glob("./data/*.json")

stemname = Path(filenames[0]).stem.split("_", 1)[0]

indices = [item.split("_", 1)[-1].split(".", 1)[0] for item in filenames]

num = list(range(0, len(indices)))
random.shuffle(num)

glob.glob("./data/*" + indices[0] + ".json")

for i in range(0, len(indices)):
    print(i)
    if num[i] < np.percentile(num, percs[0]):
        shutil.copy(filenames[i], "./data/train/")
        shutil.copy("./data/" + stemname + "_" + indices[i] + ".json", "./data/train/")
    # elif num[i] < np.percentile(num, percs[1]):
    #    shutil.copy(filenames[i], "./data/val/")
    #    shutil.copy("./data/" + stemname + "_" + indices[i] + ".json", "./data/val/")
    else:
        shutil.copy(filenames[i], "./data/test/")
        shutil.copy("./data/" + stemname + "_" + indices[i] + ".json", "./data/test/")


folds = 3

filenames = glob.glob("./data/train/*.txt")
jsonnames = glob.glob("./data/train/*.json")

stemname = Path(filenames[0]).stem.split("_", 1)[0]

indices = [item.split("_", 1)[-1].split(".", 1)[0] for item in filenames]

random.shuffle(indices)
ind_split = np.array_split(indices, folds)

for i in range(0, folds):
    Path("./data/train/fold_" + str(i + 1) + "/").mkdir(parents=True, exist_ok=True)
    for ind in ind_split[i]:
        print(ind)
        shutil.move(
            "./data/train/" + stemname + "_" + ind + ".txt",
            "./data/train/fold_" + str(i + 1) + "/",
        )
        shutil.move(
            "./data/train/" + stemname + "_" + ind + ".json",
            "./data/train/fold_" + str(i + 1) + "/",
        )

