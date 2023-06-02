<p align="left">
<img width="350" alt="predictions" src= ./.github/logo.png#gh-light-mode-only>
<img width="350" alt="predictions" src= ./.github/logo_dark.png#gh-dark-mode-only>
</p>

DUE TO AN INFLUX OF NEW USERS WE HAVE BEEN HITTING BANDWIDTH LIMITS. THIS IS PRIMARILY FROM THE FILE SIZE OF THE PRE-TRAINED MODELS. IF YOU ARE USING THESE MODELS PLEASE AIM TO SAVE THEM LOCALLY AND POINT TO THEM WHEN YOU NEED THEM RATHER THAN DOWNLOADING THEM EACH TIME THEY ARE REQUIRED. WE WILL MOVE TO A MORE BANDWIDTH FRIENDLY SET UP SOON. IN THE MEANTIME, IF INSTALLING THE PACKAGE IS FAILING PLEASE RAISE IT AS AN ISSUE OR NOTIFY ME DIRECTLY ON BALL.JGC@GMAIL.COM.

 [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) [![Detectree CI](https://github.com/patball1/detectree2/actions/workflows/python-ci.yml/badge.svg)](https://github.com/patball1/detectree2/actions/workflows/python-ci.yml) [![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/) [![DOI](https://zenodo.org/badge/470698486.svg)](https://zenodo.org/badge/latestdoi/470698486)


<!-- <a href="https://github.com/hhatto/autopep8"><img alt="Code style: autopep8" src="https://img.shields.io/badge/code%20style-autopep8-000000.svg"></a> -->

Python package for automatic tree crown delineation based on Mask R-CNN. Pre-trained models can be picked in the [`model_garden`](https://github.com/PatBall1/detectree2/tree/master/model_garden).
A tutorial on how to prepare data, train models and make predictions is available [here](https://patball1.github.io/detectree2/tutorial.html). For questions, collaboration proposals and requests for example data email [James Ball](mailto:ball.jgc@gmail.com).

<sub>Code developed by James Ball, Seb Hickman, Thomas Koay, Oscar Jiang, Luran Wang, Panagiotis Ioannou, James Hinton and Matthew Archer in the [Forest Ecology and Conservation Group](https://coomeslab.org/) at the University of Cambridge.
The Forest Ecology and Conservation Group is led by Professor David Coomes and is part of the University of Cambridge [Conservation Research Institute](https://www.conservation.cam.ac.uk/).
MRes project repo available [here](https://github.com/shmh40/detectreeRGB).</sub>

## Citation

**Please cite**:

Ball, J.G.C., Hickman, S.H.M., Jackson, T.D., Koay, X.J., Hirst, J., Jay, W., Archer, M., Aubry-Kientz, M., Vincent, G. and Coomes, D.A. (2023),
Accurate delineation of individual tree crowns in tropical forests from aerial RGB imagery using Mask R-CNN.
*Remote Sens Ecol Conserv*. [https://doi.org/10.1002/rse2.332](https://doi.org/10.1002/rse2.332)

## Requirements

- Python 3.8+
- [gdal](https://gdal.org/download.html) geospatial libraries
- [PyTorch ≥ 1.8 and torchvision](https://pytorch.org/get-started/previous-versions/) versions that match
- For training models GPU access (with CUDA) is recommended

e.g.
```pip3 install torch torchvision torchaudio```

## Installation

### pip

```pip install git+https://github.com/PatBall1/detectree2.git```

Currently works on Google Colab (Pro version recommended). May struggle on clusters if geospatial libraries are not configured.
See [Installation Instructions](https://patball1.github.io/detectree2/installation.html) if you are having trouble.

### conda

*Under development*

## Getting started

Detectree2, based on the [Detectron2](https://github.com/facebookresearch/detectron2) Mask R-CNN architecture, locates trees in aerial images. It has been designed to delineate trees in challenging dense tropical forests for a range of ecological applications.

This [tutorial](https://patball1.github.io/detectree2/tutorial.html) takes you through the key steps. [Example Colab notebooks](https://github.com/PatBall1/detectree2/tree/master/notebooks/colab) are also available.

The standard workflow includes:

1) Tile the orthomosaics and crown data (for training, validation and testing)
2) Train (and tune) a model on the training tiles
3) Evaluate the model performance by predicting on the test tiles and comparing to manual crowns for the tiles
4) Using the trained model to predict the crowns over the entire region of interest

Training crowns are used to teach the network to delineate tree crowns.
<p align="center">
<img width="500" align="center" alt="predictions" src= ./report/figures/Workflow_Diagram2_a.png#gh-light-mode-only>
<img width="500" align="center" alt="predictions" src= ./report/figures/Workflow_Diagram2_b.png#gh-dark-mode-only>
</p>

Here is an example image of the predictions made by Detectree2.
<p align="center">
<img width="700" align="center" alt="predictions" src= ./report/figures/prediction_paracou.png >
</p>

## Independent validation

> *Detectree2 (F1 score: 0.57) outperformed DeepForest (F1 score: 0.52)*
>
> *Detectree2 could estimate tree crown areas accurately, highlighting its potential and robustness for tree detection and delineation*

Gan, Yi, Quan Wang, and Atsuhiro Iio. (2023).
Tree Crown Detection and Delineation in a Temperate Deciduous Forest from UAV RGB Imagery Using Deep Learning Approaches: Effects of Spatial Resolution and Species Characteristics. 
*Remote Sensing*. 15(3):778. [https://doi.org/10.3390/rs15030778](https://doi.org/10.3390/rs15030778)


## Applications

### Tracking tropical tree growth and mortality

<p align="center">
<img width="500" alt="predicting" src= ./report/figures/growth_mortality_bootstrap.png >
</p>

### Counting urban trees (Buffalo, NY)

<p align="center">
<img width="700" alt="predicting" src= ./report/figures/urban.png >
</p>

### Multi-temporal tree crown segmentation

<p align="center">
<img width="700" alt="predicting" src= ./report/figures/seg.gif >
</p>

### Liana detection and infestation mapping

*In development*

<p align="center">
<img width="700" alt="predicting" src= ./report/figures/Lianas_detect.jpg >
</p>

### Tree species identification and mapping

*In development*

## To do

- Functions for multiple labels vs single "tree" label

## Project Organization

```
├── LICENSE
├── Makefile
├── README.md
├── detectree2
│   ├── data_loading
│   ├── models
│   ├── preprocessing
│   ├── R
│   └── tests
├── docs
│   └── source
├── model_garden
├── notebooks
│   ├── colab
│   ├── colabJB
│   ├── colabJH
│   ├── colabKoay
│   ├── colabPan
│   ├── colabSeb
│   ├── exploratory
│   ├── mask_rcnn
│   │   ├── testing
│   │   └── training
│   ├── reports
│   └── turing
├── report
│   ├── figures
│   └── sections
└── requirements
```

## Code formatting

To automatically format your code, make sure you have `black` installed (`pip install black`) and call
```black .```
from within the project directory.

---

Copyright (c) 2022, James G. C. Ball
