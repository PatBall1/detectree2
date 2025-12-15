# SwinT_detectron2
Swin Transformer for Object Detection by detectron2

This repo contains the supported code and configuration files to reproduce object detection results of [Swin Transformer](https://arxiv.org/pdf/2103.14030.pdf). It is based on [detectron2](https://github.com/facebookresearch/detectron2).

**You can find SwinV2 in this [repo](https://github.com/xiaohu2015/nndet2)**

## Results and Models

### RetinaNet

| Backbone | Pretrain | Lr Schd | box mAP | mask mAP | #params | FLOPs | config | log | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| Swin-T | ImageNet-1K | 3x | 44.6| - | - | - | [config](configs/SwinT/retinanet_swint_T_FPN_3x_.yaml) | - | [model](https://github.com/xiaohu2015/SwinT_detectron2/releases/download/v1.2/retinanet_swint_S_3x.pth) |

### Faster R-CNN

| Backbone | Pretrain | Lr Schd | box mAP | mask mAP | #params | FLOPs | config | log | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| Swin-T FPN| ImageNet-1K | 3x | 45.1| - | - | - | [config](configs/SwinT/faster_rcnn_swint_T_FPN_3x_.yaml) | - | [model](https://github.com/xiaohu2015/SwinT_detectron2/releases/download/v1.1/faster_rcnn_swint_T.pth) |

### Mask R-CNN

| Backbone | Pretrain | Lr Schd | box mAP | mask mAP | #params | FLOPs | config | log | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| Swin-T FPN| ImageNet-1K | 3x | 45.5 | 41.8 | - | - | [config](configs/SwinT/mask_rcnn_swint_T_FPN_3x.yaml) | - | [model](https://github.com/xiaohu2015/SwinT_detectron2/releases/download/v1.0/mask_rcnn_swint_T_coco17.pth) |

***The mask mAP (41.8 vs 41.6) is same as the mmdetection, but box mAP is worse (45.5 vs 46.0)***


## Usage
Please refer to [get_started.md](https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html) for installation and dataset preparation.

note: you need convert the original pretrained weights to d2 format by [convert_to_d2.py](convert_to_d2.py)

## References
- [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
- [detectron2](https://github.com/facebookresearch/detectron2)
