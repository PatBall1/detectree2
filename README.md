# detectree2

 [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) [![Detectree CI](https://github.com/patball1/detectree2/actions/workflows/python-ci.yml/badge.svg)](https://github.com/patball1/detectree2/actions/workflows/python-ci.yml)[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

<!-- <a href="https://github.com/hhatto/autopep8"><img alt="Code style: autopep8" src="https://img.shields.io/badge/code%20style-autopep8-000000.svg"></a> -->

Python package for automatic tree crown delineation based on Mask R-CNN. Pre-trained models can be picked in the `model_garden`.
A tutorial on how to prepare data, train models and make predictions is available [here](https://patball1.github.io/detectree2/tutorial.html).

<sub>Code developed by Seb Hickman, James Ball, Thomas Koay, Panagiotis Ioannou, James Hinton and Matthew Archer in the [Forest Ecology and Conservation Group](https://coomeslab.org/) at the University of Cambridge.
The Forest Ecology and Conservation Group is led by Professor David Coomes and is part of the University of Cambridge [Conservation Research Institute](https://www.conservation.cam.ac.uk/).
Original MRes project repo at <https://github.com/shmh40/detectreeRGB>.</sub>

**Please cite**:

Accurate tropical forest individual tree crown delineation from RGB imagery using Mask R-CNN.
Sebastian H M Hickman, James G C Ball, Tobias D Jackson, Xian Jing Koay, James Hirst, William Jay, Melaine Aubry-Kientz, Gregoire Vincent, David A Coomes
*bioRxiv* 2022.07.10.499480; doi: [https://doi.org/10.1101/2022.07.10.499480](https://doi.org/10.1101/2022.07.10.499480)

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

### conda

```conda install detectree2 -c conda-forge```

## Getting started

Detectree2, based on the [Detectron2](https://github.com/facebookresearch/detectron2) Mask R-CNN architecture, locates trees in aerial images. It has been designed to delineate trees in challenging dense tropical forests for a range of ecological applications.

The standard workflow includes:

1) Tile the orthomosaics and crown data (for training, validation and testing)
2) Train (and tune) a model on the training tiles
3) Evaluate the model performance by predicting on the test tiles and comparing to manual crowns for the tiles
4) Using the trained model to predict the crowns over the entire region of interest

Training crowns are used to teach the network to delineate tree crowns
<p align="center">
<img width="500" align="center" alt="predictions" src= ./report/figures/Workflow_Diagram2_a.png#gh-light-mode-only>
<img width="500" align="center" alt="predictions" src= ./report/figures/Workflow_Diagram2_b.png#gh-dark-mode-only>
</p>

Here is an example image of the predictions made by Detectree2.
<p align="center">
<img width="700" align="center" alt="predictions" src= ./report/figures/prediction_paracou.png >
</p>

## To do

- Functions for multiple labels vs single "tree" label
- Implement early stopping
- Gather "pristine" training and testing tiles across all available sites
- Availability of pre-trained models (```model_garden```)

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

## Project Organization

```
├── LICENSE
├── Makefile           <- Makefile with commands like `make init` or `make lint-requirements`
├── README.md          <- The top-level README for developers using this project.
|
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
|   |                     the creator's initials, and a short `-` delimited description, e.g.
|   |                     `1.0_jqp_initial-data-exploration`.
|   ├── colab          <- Operational Google Colab notebooks.
|   ├── colabPan       <- Operational Google Colab notebooks updated by Panagiotis Ioannou.
│   ├── exploratory    <- Notebooks for initial exploration.
│   ├── reports        <- Polished notebooks for presentations or intermediate results.
│   └── turing         <- Notebooks developed by Seb Hickman (Cambridge) and 
|                         Alejandro Coca Castro (Turing Institute) for Environmental AI Book.
│
├── report             <- Generated analysis as HTML, PDF, LaTeX, etc.
│   ├── figures        <- Generated graphics and figures to be used in reporting
│   └── sections       <- LaTeX sections. The report folder can be linked to your overleaf
|                         report with github submodules.
│
├── requirements       <- Directory containing the requirement files.
│
├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data_loading   <- Scripts to download or generate data
│   │
│   ├── preprocessing  <- Scripts to turn raw data into clean data and features for modeling
|   |
│   ├── models         <- Scripts to train models and then use trained models to make
│   │                     predictions
│   │
│   └── tests          <- Scripts for unit tests of your functions
│
└── setup.cfg          <- setup configuration file for linting rules

NEW TREE:

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

Project template created by the [Cambridge AI4ER Cookiecutter](https://github.com/ai4er-cdt/ai4er-cookiecutter).
