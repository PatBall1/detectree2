from setuptools import find_packages, setup

setup(
    name="detectree2",
    python_requires=">=3.7",  # detectron2 requirement
    version="0.0.1",
    author="James G. C. Ball",
    author_email="ball.jgc@gmail.com",
    description="Python package for automatic tree crown delineation based on Mask R-CNN",
    url="https://github.com/PatBall1/detectree2",
    # package_dir={"": "detectree2"},
    packages=find_packages(),
    license="MIT",
    test_suite="detectree2.tests.test_all.suite",
    install_requires=[
        "pyyaml==5.1",
        "GDAL>=1.11",
        "torch>=1.8",
        "torchvision>=0.9.0",
        "proj",
        "geos",
        "pypng",
        "pygeos",
        "geopandas",
        "opencv-python",
        "rasterio==1.3a3",
        "fiona",
        "pycrs",
        "descartes",
        "detectron2@git+https://github.com/facebookresearch/detectron2.git",
    ],
    extras_require={
        "docs": ["sphinx==5.1.0", "sphinx_rtd_theme", "nbsphinx"],
        "lint": ["flake8", "yapf", "flake8-docstrings", "pydocstyle", "isort", "mypy"],
        "test": ["pytest", "pytest-order"],
        "optional": [],
        "ci": ["detectree2[docs]", "detectree2[lint]", "detectree2[optional]", "detectree2[test]"]
    },
)
