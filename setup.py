from setuptools import find_packages, setup

setup(
    name="detectree2",
    version="2.1.1",
    author="James G. C. Ball",
    author_email="ball.jgc@gmail.com",
    description="Detectree packaging",
    url="https://github.com/PatBall1/detectree2",
    packages=find_packages(),
    test_suite="detectree2.tests.test_all.suite",
    python_requires=">=3.8",
    install_requires=[
        # Core
        "pyyaml>=5.1",
        "numpy>=1.20",
        "pandas>=1.3",
        "tqdm>=4.60",
        "opencv-python>=4.5",
        # Geospatial — shapely 2.x required by evaluation module (make_valid)
        "shapely>=2.0",
        "geopandas>=0.13",
        "rasterio>=1.2,<1.4",
        "fiona>=1.8,<1.10",
        "rtree>=0.9",
        # Evaluation utils
        "pycocotools>=2.0.4",
        # Swin backbone (direct VCS dependency)
        "swint @ git+https://github.com/xiaohu2015/SwinT_detectron2.git#egg=swint",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
