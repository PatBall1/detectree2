# detectree2

 [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
 <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

<sub>Code developed by Seb Hickman, James Ball, Thomas Koay, Panagiotis Ioannou and James Hinton in the [Forest Ecology and Conservation Group](https://coomeslab.org/) at the University of Cambridge.
The Forest Ecology and Conservation Group is led by Professor David Coomes and is part of the University of Cambridge [Conservation Research Institute](https://www.conservation.cam.ac.uk/).
Original MSc project repo at https://github.com/shmh40/detectreeRGB.</sub>


## Requirements
- Python 3.8+
- [gdal](https://gdal.org/download.html) geospatial libraries
- [PyTorch ≥ 1.8 and torchvision](https://pytorch.org/get-started/previous-versions/) versions that match


e.g.
```pip3 install torch torchvision torchaudio```


## Installation

To install run:

```pip install git+https://github.com/PatBall1/detectree2.git```

Currently works on Google Colab (Pro version recommended). May struggle on clusters if geospatial libraries are not configured.

### Conda / mamba install
It is recommended to install mamba to speed up build process.

`mamba create --name detectenv --file conda-linux-64.lock`


### Update conda install

Re-generate Conda lock file(s) based on environment.yml

`conda-lock -k explicit --conda mamba`

Update Conda packages based on re-generated lock file

`mamba update --file conda-linux-64.lock`

## Getting started

Detectree2, based on the [Detectron2](https://github.com/facebookresearch/detectron2) Mask R-CNN architecture, locates trees in aerial images. It has been designed to delineate trees in challenging dense tropical forests for a range of ecological applications.

Here is an example image of the predictions made by Mask R-CNN.
<p align="center">
<img width="700" align="center" alt="predictions" src= https://github.com/patball1/detectree2/blob/master/report/figures/plot_13_285520_583300.jpg > 
</p>

## To do

- Functions for multiple labels vs single "tree" label
- Implement early stopping
- Gather "prisine" training and testing tiles across all available sites
- Availability of pre-trained model

## Applications

### Tracking tropical tree growth and mortality

### Counting urban trees (Buffalo, NY)

<p align="center">
<img width="700" alt="predicting" src= https://github.com/patball1/detectree2/blob/master/report/figures/urban.png > 
</p>

### Multi-temporal tree crown segmentation

<p align="center">
<img width="700" alt="predicting" src= https://github.com/patball1/detectree2/blob/master/report/figures/seg.gif > 
</p>

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
```

## Code formatting
To automatically format your code, make sure you have `black` installed (`pip install black`) and call
```black . ``` 
from within the project directory.

---

Project template created by the [Cambridge AI4ER Cookiecutter](https://github.com/ai4er-cdt/ai4er-cookiecutter).
