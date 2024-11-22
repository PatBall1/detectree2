************
Installation
************
For a guide on installation and usage on a cluster, see the cluster dedicated page.

To use detectree2, install it using:

.. code-block:: console

   (.venv) $ pip install git+https://github.com/PatBall1/detectree2.git

Below we describe some issues that may arise.

===================
Building Detectree2 
===================

GDAL complexities
-----------------
GDAL presents a number of complexities. The issue is covered in `gdal/issue <https://github.com/PatBall1/detectree2/issues/1>`_ We must point to the location of the preinstalled GDAL headers, and the GDAL version must match the pip package version. https://github.com/OSGeo/gdal/issues/2293
For instance, on my cluster::

    gdal-config -v  # gives 3.0.4

So this means we must install the corresponding pip version: ``GDAL==3.0.4``. 

In the event that GDAL does not exist on the system, install it as so (assuming root access)::

    sudo apt install libgdal-dev gdal-bin


Using pip
---------

It is straightforward to install detectree2 on Colab. Simply pip install and all dependencies will be installed automatically. 

On other systems the process can be more involved especially if root access is not available. See workflow `python-app.yaml <https://github.com/PatBall1/detectree2/tree/master/.github/workflows/python-app.yml>`_ workflow for a working CPU deployment. 

First we need to install ``pytorch``, ``torchvision`` and ``torchaudio`` (compatible versions https://pypi.org/project/torchvision/).

This can be done inside ``virtualenv`` (if root access is unavailable)::

    python3 -m venv ./venv # (check version of python is sufficiently high >=3.7)
    . venv/bin/activate
    pip install --upgrade pip
    pip install wheel
    pip install opencv-python
    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

Then point to preinstalled GDAL header files::

    export CPLUS_INCLUDE_PATH=/usr/include/gdal
    export C_INCLUDE_PATH=/usr/include/gdal

then::

    pip install .  # (add -e flag to allow editable installs)

.. todo:: 

    * Pin torch and torchvision versions in setup.py
    * https://detectron2.readthedocs.io/en/latest/tutorials/install.html
    * http://www.tekroi.in/detectron2/projects/DensePose/setup.py
    * https://stackoverflow.com/questions/66738473/installing-pytorch-with-cuda-in-setup-py

Fixing detectron2 version
^^^^^^^^^^^^^^^^^^^^^^^^^
We can fix the version of ``detectron2`` by pointing to the pre-built wheel using pip::

    python -m pip install detectron2==0.6 -f \ https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

Or by changing the ``detectron2`` line in setup.py (which will build the latest version from source)::

    detectron2@https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/detectron2-0.6%2Bcu113-cp38-cp38-linux_x86_64.whl

It may be preferable to do this as errors have a tendency to be introduced into the ``detectron2`` codebase and may take a day or two to fix. 
We can also point to a specific working commit::

    pip install git+https://github.com/facebookresearch/detectron2.git@5aeb252b194b93dc2879b4ac34bc51a31b5aee13

    # or within setup.py (not tested):
    detectron2@git+https://github.com/facebookresearch/detectron2.git@5aeb252b194b93dc2879b4ac34bc51a31b5aee13

