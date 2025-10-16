# Model garden

Welcome to the model garden. Here live the pretrained models (now hosted on 
[Zenodo](https://zenodo.org/records/15863800)). Please feel free to pick what
is ripe for your tree crown delineation problem.

Download with e.g.

```
!wget https://zenodo.org/records/15863800/files/250312_flexi.pth
```

and load with:

```
trained_model = "./250312_flexi.pth"
cfg = setup_cfg(update_model=trained_model)
```

## 220723_withParacouUAV.pth

A model trained with a range of tropical data including aeroplane and UAV
mounted cameras. Sites: Paracou, Danum, Sepilok.

* Appropriate tile size ~ 100 m

## 230103_randresize_full.pth

An updated model trained across a range of tropical sites with better scale
transferability owing to random resize augmentation during training.
Sites: Paracou, Danum, Sepilok.

* Appropriate tile size ~ 100 m (with some tolerance)

## urban_trees_Cambridge20230630.pth

A new model for mapping trees in urban environments (trained on Cambridge, UK).

* Appropriate tile size ~ 200 m

### Hyperparameters

- Learning rate: 0.01709
- Data loader workers: 6
- Gamma: 0.08866
- Backbone freeze at: 2
- Warmup iterations: 184
- Batch size per image: 623
- Weight decay: 0.006519
- AP50: 62.0

## 230717_base.pth

A model for mapping trees in tropical closed canopy systems trained on aerial (aeroplane) RGB imagery across three tropical sites. It is the "base" model in Harnessing temporal and spectral dimensionality to map and identify species of individual trees in diverse tropical forests.

- Training sites: Danum, Sepilok, Paracou
- Appropriate tile size ~ 100 m (with some tolerance)
- Most suitable for aeroplane collected RGB data ~1m resolution

### Hyperparameters
- Initial model: "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
- Base learning rate: 0.0003389
- weight_decay: 0.001
- Momentum: 0.9
- batch_size_per_im: 1024
- Gamma: 0.1
- Backbone freeze at: 3
- Warmup iterations: 120
- Augmentations: random flipping, resize, rotation, changes in lighting, brightness, contrast, and saturation

## 230729_05dates.pth

A model for mapping trees in tropical closed canopy systems. It took the above 230717_base.pth and was further trained on 5 dates of UAV RGB imagery (~5cm resolution) from Paracou. It is the "5 date" model in [*Harnessing temporal and spectral dimensionality to map and identify species of individual trees in diverse tropical forests*](https://doi.org/10.1101/2024.06.24.600405).

- Training sites: As above plus 5 dates of UAV imagery from Paracou
- Appropriate tile size ~ 100 m (with some tolerance)
- Suitable for high resolution UAV RGB data
- Hyperparameters
- Initial model: 230717_base.pth (above)
- Base learning rate: 0.0003389
- weight_decay: 0.001
- Momentum: 0.9
- batch_size_per_im: 1024
- Gamma: 0.1
- Backbone freeze at: 3
- Warmup iterations: 120
- Augmentations: random flipping, resize, rotation, changes in lighting, brightness, contrast, and saturation

## 250312_flexi.pth

An RGB model that is trained on both closed canopy systems and urban environments. This versatility allows the model to function effectively across a wider range of settings without incorrectly identifying as many trees in non-forested areas, whilst still maintaining the ability to distinguish between closed canopy trees. Since this is a generalised model, it is advisable to utilise one of the more specialised models if your specific use-case is clearly defined, as these will demonstrate superior accuracy in their targeted environments. Examples for [closed canopy](https://zenodo.org/records/15014174/files/Harapan.png) and [urban](https://zenodo.org/records/15014174/files/Cambridge.png).

-Training sites: Harapan, Danum, Paracou, Cambridge & Sepilok
-Appropriate tile size ~ 100 m (with some tolerance)
-Hyperparameters
-Base learning rate: 0.001
-weight_decay: 0.001
-Momentum: 0.9
-batch_size_per_im: 1024
-Gamma: 0.1
-Backbone freeze at: 2 until convergence, then 0 for 200 steps
-Warmup iterations: 120
-Augmentations: random flipping, resize, rotation, changes in lighting, brightness, contrast, and saturation

### Hyperparameters

- Base learning rate: 0.001
- weight_decay: 0.001
- Momentum: 0.9
- batch_size_per_im: 1024
- Gamma: 0.1
- Backbone freeze at: 2 until convergence, then 0 for 200 steps
- Warmup iterations: 120
- Augmentations: random flipping, resize, rotation, changes in lighting, brightness, contrast, and saturation

&nbsp;
&nbsp;

![model_garden](https://i.imgur.com/uc5fCoi.jpeg)
