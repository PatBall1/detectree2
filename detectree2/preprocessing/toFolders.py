import os
import glob
import numpy as np
import random
import shutil
from pathlib import Path

#random.seed(10)
#Path("./data/train").mkdir(parents=True, exist_ok=True)
## Path("./data/val").mkdir(parents=True, exist_ok=True)
#Path("./data/test").mkdir(parents=True, exist_ok=True)
#
## First split between train and test
#split = np.array([4, 1])
#summed = np.sum(split)
#percs = 100 * split / summed
#percs = np.cumsum(percs)
#
#filenames = glob.glob("./data/*.png")
#jsonnames = glob.glob("./data/*.json")
#
#stemname = Path(filenames[0]).stem.split("_", 1)[0]
#
#indices = [item.split("_", 1)[-1].split(".", 1)[0] for item in filenames]
#
#num = list(range(0, len(indices)))
#random.shuffle(num)
#
#glob.glob("./data/*" + indices[0] + ".json")
#
#for i in range(0, len(indices)):
#    print(i)
#    if num[i] < np.percentile(num, percs[0]):
#        shutil.copy(filenames[i], "./data/train/")
#        shutil.copy("./data/" + stemname + "_" + indices[i] + ".json", "./data/train/")
#    # elif num[i] < np.percentile(num, percs[1]):
#    #    shutil.copy(filenames[i], "./data/val/")
#    #    shutil.copy("./data/" + stemname + "_" + indices[i] + ".json", "./data/val/")
#    else:
#        shutil.copy(filenames[i], "./data/test/")
#        shutil.copy("./data/" + stemname + "_" + indices[i] + ".json", "./data/test/")
#
#
#folds = 3
#
#filenames = glob.glob("./data/train/*.png")
#jsonnames = glob.glob("./data/train/*.json")
#
#stemname = Path(filenames[0]).stem.split("_", 1)[0]
#
#indices = [item.split("_", 1)[-1].split(".", 1)[0] for item in filenames]
#
#random.shuffle(indices)
#ind_split = np.array_split(indices, folds)
#
#for i in range(0, folds):
#    Path("./data/train/fold_" + str(i + 1) + "/").mkdir(parents=True, exist_ok=True)
#    for ind in ind_split[i]:
#        print(ind)
#        shutil.move(
#            "./data/train/" + stemname + "_" + ind + ".png",
#            "./data/train/fold_" + str(i + 1) + "/",
#        )
#        shutil.move(
#            "./data/train/" + stemname + "_" + ind + ".json",
#            "./data/train/fold_" + str(i + 1) + "/",
#        )


def to_traintest_folders(tiles_folder="./",
                         out_folder="./data/",
                         test_frac=0.2,
                         folds=1):
  """
  To send tiles to training (+validation) and test folder
  """

  Path(out_folder + "train").mkdir(parents=True, exist_ok=True)
  Path(out_folder + "test").mkdir(parents=True, exist_ok=True)

  # First split between train and test
  #split = np.array([4, 1])
  split = np.array([(1 - test_frac), test_frac])
  summed = np.sum(split)
  percs = 100 * split / summed
  percs = np.cumsum(percs)

  filenames = glob.glob(tiles_folder + "*.png")
  fileroots = [Path(item).stem for item in filenames]
  #jsonnames = glob.glob(tiles_folder + "*.geojson")
  #stemname = Path(filenames[0]).stem.split("_", 1)[0]
  #indices = [item.split("_", 1)[-1].split(".", 1)[0] for item in filenames]

  num = list(range(0, len(filenames)))
  random.shuffle(num)

  for i in range(0, len(filenames)):
    #print(i)
    if num[i] < np.percentile(num, percs[0]):
      shutil.copy(filenames[i], out_folder + "train/")
      shutil.copy(tiles_folder + fileroots[i] + ".geojson",
                  out_folder + "train/")
    # elif num[i] < np.percentile(num, percs[1]):
    #    shutil.copy(filenames[i], "./data/val/")
    #    shutil.copy("./data/" + stemname + "_" + indices[i] + ".geojson", "./data/val/")
    else:
      shutil.copy(filenames[i], out_folder + "test/")
      shutil.copy(tiles_folder + fileroots[i] + ".geojson",
                  out_folder + "test/")

  filenames = glob.glob(out_folder + "/train/*.png")
  #jsonnames = glob.glob(out_folder + "/train/*.geojson")
  fileroots = [Path(item).stem for item in filenames]
  #stemname = Path(filenames[0]).stem.split("_", 1)[0]

  #indices = [item.split("_", 1)[-1].split(".", 1)[0] for item in filenames]
  num = list(range(0, len(filenames)))
  random.shuffle(num)
  #random.shuffle(indices)
  ind_split = np.array_split(fileroots, folds)

  for i in range(0, folds):
    Path(out_folder + "/train/fold_" + str(i + 1) + "/").mkdir(parents=True,
                                                               exist_ok=True)
    for name in ind_split[i]:
      #print(ind)
      shutil.move(
          out_folder + "train/" + name + ".png",
          out_folder + "train/fold_" + str(i + 1) + "/",
      )
      shutil.move(
          out_folder + "train/" + name + ".geojson",
          out_folder + "train/fold_" + str(i + 1) + "/",
      )


if __name__ == "__main__":
  to_traintest_folders(folds=3)
