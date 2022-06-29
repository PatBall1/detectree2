import sys
import os
import json
import random
from detectree2.models.train import get_tree_dicts

# change to location of script
abspath = os.path.abspath(sys.argv[0]) 
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir("..")

output_root = './output2/paracou/'
train_path = os.path.join(output_root, 'train')
test_path = os.path.join(output_root, 'test')

train_dirs = os.listdir(train_path)
print(train_dirs)

# This allows us to improve plotting
train = False


# This doesn't seem necessary - imagePath is already set correctly in .geojson files
# Can be done with pathlib. Replace geojson extension with png. 
def update_json(geojson_path):
    for i in entries:
        if 'geojson' in i: 
            filename = geojson_path + i
            print(i)
            print(filename)
            j=i[:-8] # remove file extension
            print(j)
            k= j+ '.png'
            Dict = {"imagePath":  k }
            print(Dict)

            with open(filename,"r") as f:
                data = json.load(f)
                data.update(Dict)
            with open(filename,"w") as f:
                json.dump(data,f)

# update_json(train_path)
# update_json(test_path)

# combine all folds 
# k-fold cross validation implementation - the order on my machine is:
# ['fold_3', 'fold_5', 'fold_1', 'fold_2', 'fold_4'] so fold_3 is the hold out set
def combine_dicts(traind, val_folder):
    train_dirs = [os.path.join(traind, dir) for dir in os.listdir(traind)]
    print(train_dirs)
    del train_dirs[(val_folder-1)]
    print(train_dirs)
    tree_dicts = []
    for d in train_dirs:
        tree_dicts = tree_dicts + get_tree_dicts(d)
    return tree_dicts

dicts = combine_dicts(train_path, 1)  
len(dicts)

from detectron2.data import DatasetCatalog, MetadataCatalog

DatasetCatalog.register("trees_train", lambda d=train_path: combine_dicts(d, 1))
MetadataCatalog.get("trees_train").set(thing_classes=['tree'])
DatasetCatalog.register("trees_test", lambda d='test': get_tree_dicts(os.path.join(output_root, d))) 
# DatasetCatalog.register("trees_test", lambda d='fold_x': get_tree_dicts(os.path.join(train_path, d))) 
MetadataCatalog.get("trees_test").set(thing_classes=['tree'])

# trees_metadata = MetadataCatalog.get("trees_train")
# print(trees_metadata)

# now train:
from detectree2.models.train import MyTrainer
import logging

# from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo

if train:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("trees_train",)
    cfg.DATASETS.TEST = ("trees_test",)
    cfg.DATALOADER.NUM_WORKERS = 2
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    # load our pre-trained model if you like - Is this necessary?
    cfg.MODEL.WEIGHTS = 'data/model_final.pth'
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0005
    # cfg.SOLVER.WARMUP_ITERS = 500
    cfg.SOLVER.MAX_ITER = 1000
    # cfg.SOLVER.STEPS = (250,)
    # cfg.SOLVER.GAMMA = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.OUTPUT_DIR = output_root
    cfg.EVAL_PERIOD = 10

    ### From here is the important bit that hasn't been repeated further up
    #cfg.TEST.EVAL_PERIOD = 100

    trainer = MyTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer.train()

### Plot training and validation loss on the same plot to check how the training has gone

import json
import matplotlib.pyplot as plt

experiment_folder = output_root

def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

experiment_metrics = load_json_arr(os.path.join(experiment_folder, 'metrics.json'))

plt.plot(
    [x['iteration'] for x in experiment_metrics if 'validation_loss' in x], 
    [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x], label='Total Validation Loss', color='red')
plt.plot(
    [x['iteration'] for x in experiment_metrics if 'total_loss' in x], 
    [x['total_loss'] for x in experiment_metrics if 'total_loss' in x], label='Total Training Loss')

plt.legend(loc='upper right')
plt.title('Comparison of the training and validation loss of Mask R-CNN')
plt.ylabel('Total Loss')
plt.xlabel('Number of Iterations')
# plt.show()
plt.savefig(os.path.join(output_root,'training_loss_model.pdf'))