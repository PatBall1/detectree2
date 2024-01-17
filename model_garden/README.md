# Model garden

Welcome to the model garden. Here live the pretrained models (now hosted on 
[Zenodo](https://zenodo.org/records/10522461)). Please feel free to pick what
is ripe for your tree crown delineation problem.

Download with e.g.

```
!wget https://zenodo.org/records/10522461/files/230103_randresize_full.pth
```

and load with:

```
trained_model = "./230103_randresize_full.pth"
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

&nbsp;
&nbsp;

![model_garden](https://i.imgur.com/uc5fCoi.jpeg)
