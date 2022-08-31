==================
Contributing guide
==================

.. contents::


Contributing to Detectree2
==========================

Thanks for taking the time to contribute to detectree2!

Code of Conduct
---------------

All contributors are expected to abide by the `Code of Conduct <https://github.com/PatBall1/detectree2/blob/master/CODE_OF_CONDUCT.md>`_


Getting Started - Issues
------------------------
The "to do" task list for detectree2 starts at the `Issues <https://github.com/PatBall1/detectree2/issues>`_ page in Github.


Note: Before submitting a new issue, you should search the existing issues to make sure that it hasn't already been reported.


Filing a bug report
-------------------
Bug reports are important in helping identify things that aren't working in detectree2. Filing a bug report is straightforward: Head to the Issues page, click on the New issue button, and select the Bug Report template to get started.


Detectree2 Github tips
----------------------

There are many excellent resources for learning Git: 

* `<https://github.com/JamesFergusson/Introduction-to-Research-Computing/blob/master/05_BestPractice_GIT.md>`_
* `<https://www.bristol.ac.uk/acrc/research-software-engineering/training/>`_
   * `<https://chryswoods.com/introducing_git/>`_
   * `<https://chryswoods.com/git_collaboration/>`_
* `<https://ohshitgit.com/>`_

 


Git Rebase/Merge during PR
^^^^^^^^^^^^^^^^^^^^^^^^^^

It is strongly recommended to sync master with the feature branch during the submission of a PR. One can either         ``merge`` master or ``rebase`` on master to sync changes. Either is fine in practice, but for large projects, with many contributors it is considered good practice to ``rebase`` to keep the history linear.

However, I would **strongly recommend** the ``squash and merge`` approach when committing a PR to master (this is done using the github UI). This combines all of the commits into one commit in the base branch with an option to edit a commit summary. Therefore we do not need to worry about the effects of merging in other branches on the project history.  It is possible to make this the default behaviour in the repository settings. 

If for any reason you do not want to squash and merge the commits (i.e. to keep the PR's commit history in tact), I would suggest rebasing. 

TIP: Try to avoid merging the PR to a `dev` branch. This is considered bad practice since when it comes to merge to master the eventual `PR` can be large and difficult to understand. PRs should have a single focus. 

TIP: Delete the branch after merging to master. 

TIP: Many of the recommendations above can be made default in Github's settings:

.. todo::
    * Prohibit commits direct to master.
    * Automatically squash on merge.
    * Prevent merge unless all tests are passing. 
    * Only allow a merge if approved by assigned Reviewers. 



Detectree2 setup for developers
===============================

For projects with many contributors it is good to adhere to a programming style and testing framework. Settings for detectree2 are given in `setup.cfg`, in the project root. 

We adopt `GitHub actions` to deploy our software development workflows `detectree2/actions <https://github.com/PatBall1/detectree2/actions>`_, workflows are steered automatically using github actions in `.github/workflows <https://github.com/PatBall1/detectree2/tree/master/.github/workflows>`_ directory, a good example is: `python-app.yaml <https://github.com/PatBall1/detectree2/tree/master/.github/workflows/python-app.yml>`_.  The code checks are triggered automatically on pushing to a branch. The workflows detail the required dependencies for developing and testing detectree2, and should be consulted if anything in the following section is unclear. 

For reference, the relevant ``detectree2`` subprojects are:

* `detectree2-docker GitHub repo <https://github.com/ma595/detectree2-docker>`_
* `detectree2-data GitHub repo <https://github.com/ma595/detectree2-data>`_
* `anaconda distribution <https://anaconda.org/ma595/detectree2>`_


Setting up development environment
----------------------------------

Using conda or pip (below we show pip)::

    pip install flake8 flake8-docstrings mypy autopep8 isort

.. todo::
    * Create ``dev-environment.yaml``.

Programming style
-----------------

Detectree2 currently utilises the following tools for code style checks. It is recommended to run these locally before pushing code as the CI will not pass unless each test is successful. To see up-to-date exact commands, consult the relevant workflow. Note that different versions of python (+packages) may give different errors, so correcting errors may take a few attempts. 

- ``autopep8``: Ensure consistent formatting of Python files 
- ``mypy``: Validate Python type hints 
- ``flake8``: Multiple checks for - linting - syntax errors or anti-patterns - (lack of) executable flags on files - docstring validation - function complexity
- ``isort``: Checks that imports are correctly sorted

Other style choices:

* Line length = 120 characters
* Google style docstrings
* Function signatures and comments span 120 character length

Flake8
^^^^^^
Flake8 includes linting, syntax errors, and McCabe function complexity analysis. 

The are several instances where Flake8 errors have been purposely ignored using ``noqa: <CODE>`` annotations to allow flake8 CI to pass. This is not a permanent fix and the errors should eventually be addressed. For example: ``noqa: E501`` ensures that line lengths beyond (120 characters) are ignored by the linter and ``noqa: 901`` ignores the McCabe complexity measure. 

These can also be set globally in setup.cfg, but fewer the better. It is also possible to set ``continue-on-error`` in the flake8 workflow or ``--exit-zero`` flake8 argument to allow other checks to continue. In practice it was found that developers tend to ignore flake8 errors as a result of these two options, so the ``noqa`` solution is preferred. 

McCabe function complexity analysis is useful for detecting over-complex code (as determined by the amount of branching - `if`, `else` statements). A value of 10 is set as default. 

Docstrings
^^^^^^^^^^

We adopt google docstrings (`<https://google.github.io/styleguide/pyguide.html>`_)

Other dependencies include ``flake8-docstrings``, 

.. todo::

    * Remove ``pydocstyle``

Autopep8
^^^^^^^^

Autopep8 is an autoformatter (like black) with enforces the ``pep8`` standard. 


Static typing
^^^^^^^^^^^^^

Static typing is written for compatibility with python3.7 and above. The mypy syntax could be updated as the project moves towards more modern python3. `mypy` will attempt to type check all third-party libraries which might not be desirable. However, it is possible to install stubs for third-party libraries (i.e. ``pandas``, ``openCV``) if type-checking is desired, but it is easier to suppress all missing import errors libraries by adding  ``ignore_missing_imports = True`` in ``setup.cfg``.

Continuous integration
======================

The idea of Continuous integration (CI) is to frequently commit code to a shared repo. This has the effect of detecting errors sooner thereby reducing the amount of code a developer needs to debug when finding an error. Frequent updates also make it easier to merge changes from different members of the software development team. This is especially powerful when paired with the ability to build and test the code. Testing can include code linters, unit and integration tests. 

Building and testing code requires a server. CI using GitHub actions offers workflows that can build the repository code and run tests. We can run on GitHub's own virtual machines (using GitHub-hosted runners), or on machines that we host ourselves (or on compute clusters). The latter is desirable as GitHub does not currently support access to GPU resources.

Currently there are three files that steer workflows. The schedule is set at the top of the file. The workflows are found `here <https://github.com/PatBall1/detectree2/tree/master/.github/workflows>`_

- ``python-app.yml``: All style CI - builds the code on Ubuntu-20.04
- ``dockertest.yml``: All style CI - uses docker image for dependencies and installs detectree2 using pip.
- ``documentation.yml``: Generates documentation and hosts on github pages. Builds code first for sphinx-apidoc. 

The ``dockertest.yml`` workflow is an attempt to utilise docker to speed up deployment and testing of detectree2. It pulls the docker image: `ma595/detectree-cpu-20.04:latest <https://hub.docker.com/repository/docker/ma595/detectree-cpu-20.04>`_ (Python3.8) and installs detectree2 on top. A more up to date docker container, utilising python3.10 and ubuntu 22.04 has been successfully built but has yet to be integrated into the workflow, the file can be found in `github:ma595/detectree2-docker/Dockerfile-22.04 <https://github.com/ma595/detectree2-docker/blob/main/files/Dockerfile-22.04>`_.

All dockerfiles are in `github:ma595/detectree2-docker <https://github.com/ma595/detectree2-docker>`_, which uses `github:ma595/detectree2-data <https://github.com/ma595/detectree2-data>`_ to store the data required for the workflow.


.. todo::

    - Harmonise documentation.yml and python-app.yml into single file where appropriate. There is no good reason to separate.
    - Add GPU testing to workflow (currently unsupported on Github, but we can use CSD3's A100 resources).
    - Prevent merge unless all tests are passing
    - Build docker image as part of an action and push to dockerhub (or use github's docker features)
    - Check 22.04 docker image
    - Move dockerfiles into detectree2 project. 
    - Style check documentation.



Automatic Documentation
=======================
Documentation is generated automatically using Sphinx and GitHub actions in `documentation.yaml <https://github.com/PatBall1/detectree2/blob/master/.github/workflows/documentation.yaml>`_. 

Documentation can be generated locally to test rendering. It is better to develop locally rather than rely on the CI and hosted docs as a check, as it can take quite some time to build using the workflow. 

To generate locally it is necessary to install the following dependencies (either in pip or conda)::

    pip install sphinx sphinx_rtd_theme

Then generate api documentation, and build the html.::

    sphinx-apidoc -o ./docs/source/ detectree2/
    sphinx-build -b html docs/source/ docs/build/html

Then using your favourite browser open docs/build/html/index.html

.. todo::

    * Style checks on documentation. 

Tests
=====

Test-driven development stipulates that tests should be written as new features are introduced to the code. To run the tests simply do::

    # install Pytest if haven't already done so.
    pip install pytest
    # pytest should be run from the project root: 
    pytest . 

As of August 2022, an integration test has been written which demos the tiling, and training steps. The integration test will run the training on the CPU only. It is possible to use other 

A few unit tests have been implemented, the most interesting computes the area intersection over union (with dummy .geojson data containing square shapes with known areas). The test is still incomplete because much of the code in evaluation.py and F1_calculator is not sufficiently modular - a major refactor is required. 

TIP: Always write tests for newly introduced logic when contributing code.

.. todo:: 

    * Write more unit tests for existing code. 


Building Detectree2 
===================

GDAL complexities
-----------------
GDAL presents a number of complexities. We must point to the location of the preinstalled GDAL headers, and the GDAL version must match the pip package version. https://github.com/OSGeo/gdal/issues/2293
For instance, on my cluster::

    gdal-config -v  # gives 3.0.4

So this means we must install the corresponding pip version: ``GDAL==3.0.4``. 

In the event that GDAL does not exist on the system, install it as so (assuming root access)::

    sudo apt install libgdal-dev gdal-bin


Using pip
---------

It is relatively straightforward to install detectree2 on Colab. Simply pip install and all dependencies will be installed automatically. 

On other systems the process is more involved especially if root access is not available. See workflow `python-app.yaml <https://github.com/PatBall1/detectree2/tree/master/.github/workflows/python-app.yml>`_ workflow for a working CPU deployment. 

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
    https://detectron2.readthedocs.io/en/latest/tutorials/install.html
    http://www.tekroi.in/detectron2/projects/DensePose/setup.py
    https://stackoverflow.com/questions/66738473/installing-pytorch-with-cuda-in-setup-py

Fixing detectron2 version
^^^^^^^^^^^^^^^^^^^^^^^^
We can fix the version of ``detectron2`` by pointing to the pre-built wheel using pip::

    python -m pip install detectron2==0.6 -f \ https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

Or by changing the ``detectron2`` line in setup.py (which will build the latest version from source)::

    detectron2@https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/detectron2-0.6%2Bcu113-cp38-cp38-linux_x86_64.whl

It may be preferable to do this as errors have a tendency to be introduced into the ``detectron2`` codebase and may take a day or two to fix. 
We can also point to a specific working commit::

    pip install git+https://github.com/facebookresearch/detectron2.git@5aeb252b194b93dc2879b4ac34bc51a31b5aee13

    # or within setup.py (not tested):
    detectron2@git+https://github.com/facebookresearch/detectron2.git@5aeb252b194b93dc2879b4ac34bc51a31b5aee13




Building Detectree2 using Conda
-------------------------------
Many of the aforementioned complexities can be solved using Conda. This is currently working for python 3.9.13, in branch `matt/conda <https://github.com/PatBall1/detectree2/tree/matt/conda>`_. The most important file is `environment.yaml <https://github.com/PatBall1/detectree2/blob/matt/conda/conda/environment.yaml>`_ which specifies the required dependencies. 

Install miniconda, and source (usually ``~/.miniconda/bin/activate`` if not in ``.bashrc`` already). Begin by installing ``mamba``::

    conda install mamba -c conda-forge
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

Distributing detectree2 using Conda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the `matt/conda <https://github.com/PatBall1/detectree2/tree/matt/conda>`_ branch, the `conda/meta.yaml <https://github.com/PatBall1/detectree2/blob/matt/conda/conda/meta.yaml>`_ packages detectree2. An initial attempt can be found here: `ma595/detectree2 <https://anaconda.org/ma595/detectree2>`_. To install, do the following::

    conda install -c ma595 detectree2

To rebuild from meta.yaml::

    conda-build . -c conda-forge -c pytorch --output-folder ./conda-bld

Then upload to anaconda::

    anaconda login
    anaconda upload <path-to-tar.bz2>

.. todo::
    * Automate distribution of package to `anaconda <https://anaconda.org/ma595/detectree2>`_ using workflow. 

Python development environment
------------------------------

.. todo::
    * Setting up visual studio. ``dev-environment.yaml`` file? 



.. TODO list
.. ---------
.. .. todolist::