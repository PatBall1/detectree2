# Model garden

Welcome to the model garden. Here lives the pretrained models. Please feel free
to pick what is ripe for your tree crown delineation problem.

Download with e.g.

```
!wget https://github.com/PatBall1/detectree2/raw/master/model_garden/230103_randresize_full.pth
```

and load with:

```
trained_model = "./230103_randresize_full.pth"
cfg = setup_cfg(update_model=trained_model)
```

## 220723_withParacouUAV.pth

A model trained with a range of tropical data including aeroplane and UAV
mounted cameras.

* Appropriate tile size ~ 100 m

## 230103_randresize_full.pth

An updated model trained across a range of tropical sites with better scale
transferability owing to random resize augmentation during training.

* Appropriate tile size ~ 100 m (with some flexibility)

## urban_trees.pth

A new model for mapping trees in urban environments. Available upon requests.

&nbsp;
&nbsp;

![model_garden](https://i.imgur.com/uc5fCoi.jpeg)
