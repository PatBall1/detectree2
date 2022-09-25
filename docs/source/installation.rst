*******************
Building Detectree2 
*******************

==============================
Getting up and running quickly 
==============================

To get up and running quickly it is possible to install detectree2 and its dependencies with Conda. Simply do::

    conda install -c ma595 detectree2



=========
Using pip
=========

It is easy to install detectree2 on your own system. Simply pip install and all dependencies will be installed automatically. 

 See workflow `python-app.yaml <https://github.com/PatBall1/detectree2/tree/master/.github/workflows/python-app.yml>`_ workflow for a working CPU deployment. 

First install ``pytorch``, ``torchvision`` and ``torchaudio`` (compatible versions https://pypi.org/project/torchvision/). Follow `https://pytorch.org/get-started/locally/`_ to get compatible version for your system. Below we run through the process with pip but the conda approach is equally valid.

This can be done inside ``virtualenv`` (if root access is unavailable)::

    python3 -m venv ./venv # (check version of python is sufficiently high >=3.7, required by detectron2)
    . venv/bin/activate
    pip install --upgrade pip
    pip install wheel
    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

Then point to preinstalled GDAL header files (not necessary if include directory is already in your path)::

    export CPLUS_INCLUDE_PATH=/usr/include/gdal
    export C_INCLUDE_PATH=/usr/include/gdal

then::

    git clone git@github.com:PatBall1/detectree2.git
    cd detectree2
    pip install .  # (add -e flag to allow editable installs)

On other systems the process is more involved especially if root access is not available.


.. todo:: 

    * https://detectron2.readthedocs.io/en/latest/tutorials/install.html
    * http://www.tekroi.in/detectron2/projects/DensePose/setup.py
    * https://stackoverflow.com/questions/66738473/installing-pytorch-with-cuda-in-setup-py


Possible issues with GDAL 
-------------------------
GDAL presents a number of complexities. The issue is covered in `gdal/issue <https://github.com/PatBall1/detectree2/issues/1>`_ We must point to the location of the preinstalled GDAL headers, and the GDAL version must match the pip package version. https://github.com/OSGeo/gdal/issues/2293
For instance, on my cluster::

    gdal-config -v  # gives 3.0.4

So this means we must install the corresponding pip version: ``GDAL==3.0.4`` or lower. To avoid problems, the version of the system GDAL should be higher than the version bindings.
.. 
    https://gis.stackexchange.com/questions/188639/adding-gdal-as-dependency-to-python-package
In the event that GDAL does not exist on the system, install it as so (assuming root access)::

    sudo apt install libgdal-dev gdal-bin





===============================
Building Detectree2 using Conda
================================
Many of the aforementioned complexities with pip, GDAL and detectron2 can be solved with Conda. This is currently working for python 3.9.13, in branch `matt/conda <https://github.com/PatBall1/detectree2/tree/matt/conda>`_. The most important file is `environment.yaml <https://github.com/PatBall1/detectree2/blob/matt/conda/conda/environment.yaml>`_ which specifies the required dependencies. 

Install miniconda, and source (usually ``~/.miniconda/bin/activate`` if not in ``.bashrc`` already). Begin by installing ``mamba``::

    conda install mamba -c conda-forge

And then create the detectree2 environment:

    mamba env create -f envrironment.yaml 
    mamba activate detectree2env

Alternatively we may use a conda lock file which has transitive dependencies pinned. This improves reproducibility.::

    mamba create --name detectree2env --file conda-linux-64.lock

and if we modify our environment, we can update the lock file as so::

    conda-lock -k explicit --conda mamba

and then update conda packages based on the regenerated lock file::

    mamba update --file conda-linux-64.lock

The downside of this approach is that it takes much longer to install compared to pip, even with Mamba's improved dependency resolution. 

.. todo:: 

    * Determine how this can be integrated into current pip install without breaking ``colab`` pip deployment.
    * Investigate use of poetry as it is easier to package a distribution. But detectron2 is not PEP517 compliant. 
    * It is possible to combine Conda and Poetry, where Conda is used for packages like GDAL / detectron2 / openCV. 



To use detectree2, first install it using:

.. code-block:: console

   (.venv) $ pip install detectree2