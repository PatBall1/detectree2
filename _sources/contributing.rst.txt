******************
Contributing guide
******************

.. contents::

==========================
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

There are many excellent resources for learning Git. See :doc:`using-git`, for a brief overview of Git and Github. For tips relating to detectree2 see `detectree2 tips`_. 


* `<https://github.com/JamesFergusson/Introduction-to-Research-Computing/blob/master/05_BestPractice_GIT.md>`_
* `<https://www.bristol.ac.uk/acrc/research-software-engineering/training/>`_
  
  * `<https://chryswoods.com/introducing_git/>`_
  * `<https://chryswoods.com/git_collaboration/>`_
  
* `<https://ohshitgit.com/>`_

 


Tips for creating a detectree2 PR
----------------------------------
.. _detectree2 tips:

This guide assumes that most developers do not have permission to contribute to the repo directly and therefore need to fork the project. For users with privileges to contribute to detectree2 directly the concepts are the same, but a PR can be initiated from a feature branch within the detectree2 project directly.

First consult :doc:`using-git` or any of the above resources if you are unclear on how to create a good PR. See `issue/6 <https://github.com/PatBall1/detectree2/pull/6#issuecomment-1189473815>`_ for some recommendatations on best practices when forking a project. 

Once a PR has been created, it is good practice to keep the PR's feature branch up-to-date with `upstream <https://github.com/PatBall1/detectree2>`_ master during the PR submission and review process.
This can be done simply with the ``Sync fork`` link on github which can synchronise `upstream master` to any remote branch, and then pull the changes using the command line to continue developing locally. This is the simplest way to keep the PR branch updated. 


It's also possible to do all this locally using Git. I.e. In your local clone do the following::

    git remote add upstream https://github.com/PatBall1/detectree2
    git fetch upstream
    git checkout master # if not already there
    git rebase upstream/master # or git merge upstream/master
    git checkout <feature-branch>
    git rebase master # or git merge master
    # now push to your remote repo
    git push --force-with-lease origin <feature-branch>



This process can be abbreviated to a couple of commands for advanced Git users. We can modify commits using ``git rebase -i`` (interactive rebase) to clean up the commit history. We can squash commits, remove unnecessary commits or edit commits. 

.. Which updates the local ``master`` branch and syncs to your remote fork's ``master``.  It is good practice to have the fork's master mirror the upstream master.

TIP: The process of rebasing on ``master`` may need to be done multiple times during a PR.

..  Once master is updated one can either ``merge`` master or ``rebase`` on master. This can be done using the command line during a PR or at the end using the github UI. 
 
At the end of the PR we can use GitHub's UI to commit. The available options are explained here: `Pull request merges <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/about-pull-request-merges>`_.

Using GitHub's UI to commit PR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Whether to squash commits or not is explained well in this article: 
`<https://blog.mergify.com/what-is-the-difference-between-a-merge-commit-a-squash/>`_ 


The github UI will give the ``squash and merge`` option when committing a PR to master, but one should proceed with caution. It squashes all of the commits down to one commit in the base branch with an option to edit a commit summary (please modify the commit summary from the default one provided with a more concise message of the PR's contributions). One could argue that this leads to a linear history (i.e. doesn't contain merges) but often it can result in large commits that are difficult to read, so if following this approach try to ensure that PRs have a single focus. Squashing also loses valuable information from individual commits.

.. Squashing also loses useful information, i.e. ``git blame`` cannot tell you which precise commit message corresponds to a particular line. (A general guide is that if a PR consists of logically separate parts then it makes sense to retain the commit history. But one could argue that the logically separate parts should in fact be separate PRs anyway). A further downside is that it is not possible to contribute to the head branch of a PR after you have squashed and merged the PR. Squashing can be done in Git without needing to rely on github's ``squash and merge`` button which eradicates all history. So commits like 'WIP', 'fix typo' can be removed manually and still keep the project history in tact. 

Alternatively, you can select the ``rebase and merge`` option - in this case all commits from the head branch are added onto the base branch individually without a merge commit. If you have conflicts and you still wish to rebase and merge, these need to be resolved locally using the command line as described `here <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/about-pull-request-merges>`_ and the next section. My advice is to ensure that PRs have a single focus and to reduce unnecessary style commits with pre-commit hooks. ``git rebase -i`` is an extremely useful command that every developer should be aware of to neaten history when ready to merge. 

Finally, GitHub provides the option to ``Create a merge commit`` which simply merges branch into main. I would advise against this as it creates complicated git histories that are difficult to read. 

For Detectree2, it's up to you which approach to take. Given that the project is quite small and there are few contributors I'd suggest using pre-commit hooks and interactive rebases to improve commit quality in favour of squashing. Squashing is also generally fine (and the easier approach of the two), but try to avoid doing it all the time.  

.. TIP: It is possible to make ``squash and merge`` the default behaviour in the repository settings. 

Using command line to rebase
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you want to rebase the commits but are unable to ``rebase and merge`` automatically on GitHub.com you must:

  - Rebase the PR branch onto master locally on the command line
  - Resolve any merge conflicts on the command line
  - Force push to the PR branch
     
There is plenty of useful information on this online in the `official docs <https://git-scm.com/docs/git-rebase>`_ and this `stackoverflow <https://stackoverflow.com/questions/7929369/how-to-rebase-local-branch-onto-remote-master>`_ post. This makes the GitHub UI process trivial and one can select ``rebase and merge``. Rebasing is more involved than merging but leads to a linear history.

TIP: If following this approach, make sure to use pre-commit hooks to improve quality of individual commits. 

General tips
^^^^^^^^^^^^

TIP: Try to avoid merging the PR to a `dev` branch. This is considered bad practice since when it comes to merge to master the eventual `PR` can be large and difficult to understand. PRs should have a single focus. 

TIP: Delete the branch after merging to master. 

TIP: Always nominate a collaborator to review the PR before merging. 

TIP: Do not merge unless all tests are passing. 


.. todo::
    * Many of the recommendations above can be made default in Github's settings:
     
        * Prohibit commits direct to master.
        * Automatically squash on merge.
        * Prevent merge unless all tests are passing. 
        * Only allow a merge if approved by assigned Reviewers. 
    * Change ``master`` to ``main``. GitHub provides a step-by-step walkthrough.

===============================
Detectree2 setup for developers
===============================

For projects with many contributors it is good practice to adhere to a programming style and testing framework. The programming style is enforced with a combination of pre-commit hooks and CI checks. Settings for detectree2's programming style components are given in `setup.cfg`, in the project root. 

.. The actual style guide is found in the `programming style`_ section.

We adopt `GitHub actions` to deploy software development workflows `detectree2/actions <https://github.com/PatBall1/detectree2/actions>`_, workflows are steered automatically using github actions in `.github/workflows <https://github.com/PatBall1/detectree2/tree/master/.github/workflows>`_ directory, a good example is: `python-app.yaml <https://github.com/PatBall1/detectree2/tree/master/.github/workflows/python-app.yml>`_.  The code checks are triggered automatically on pushing to a branch. The workflows detail the required dependencies for developing and testing detectree2, and should be consulted if anything in the following section is unclear. 

For reference, the relevant ``detectree2`` subprojects are:

    * `detectree2-docker GitHub repo <https://github.com/ma595/detectree2-docker>`_ For docker containers used in CI. 
    * `detectree2-data GitHub repo <https://github.com/ma595/detectree2-data>`_ For example data used in CI. 
    * `anaconda distribution <https://anaconda.org/ma595/detectree2>`_ detectree2 conda package. 

.. todo::
    * Publish model on model_zoo 

Setting up development environment
----------------------------------

Using conda or pip (below we show pip)::

    pip install flake8 flake8-docstrings mypy autopep8 isort

.. todo::
    * Create ``dev-environment.yaml``.

Programming style
-----------------
.. _programming style:

Detectree2 currently utilises the following tools to check code style and consistency. 

- ``yapf``: Autoformatter ensures consistent formatting of Python files.
- ``flake8``: Multiple checks for - linting - syntax errors or anti-patterns - (lack of) executable flags on files - docstring validation - function complexity.
- ``mypy``: Validate Python type hints.
- ``isort``: Checks that imports are correctly sorted.

A number of other style choices have been enforced across the project:

* Line length = 120 characters
* Google style docstrings
* Indent width = 4 spaces per tab.


This style is enforced both locally (using pre-commit hooks) and remotely (using github workflows). It is possible to use remote checks only, but it is often faster to check locally first than to wait for the github workflow to execute. 

Pre-commit hooks
^^^^^^^^^^^^^^^^
Pre-commit hooks ensure that each commit passes a minimum set of style checks. To set up pre-commit hooks do::

    pip install pre-commit
    pre-commit install
    pre-commit run --all-files # it is a good idea to do this over all files not just the files that have changed
    
    # or run on the files that have changed:
    
    git add -u  # e.g
    git commit -m "your message"

If tests do not pass, correct errors and then do ``git add -u`` again so that changes are staged.

If it is desirable to avoid pre-commit hooks::

    git commit -m "your message" --no-verify

With checks configured in the `.pre-commit-config.yaml` file in the project root. Note that a commit will not be made unless the tests pass. This generally has the effect of improving the quality of individual commits without needing to rely too much on server side checks for code quality.


As an alternative to running pre-commit hooks, one can still run the checks manually but the programmer must be careful that all checks pass. If everything is setup correctly, the CI should not permit a commit to ``master`` unless all tests are successful. To see up-to-date commands, consult the relevant workflow. Note that different versions of python (+packages) may give different errors to the CI, so correcting errors may take a few attempts. There may also be discrepancies between the client pre-commit hooks and server CI checks. It is best to update the pre-commit hooks if possible in this case. 



.. WARNING: ``Flake8`` will **not** detect infringements in the function signature style (and other aspects) if it still adheres to the PEP8 standard, and ``autopep8`` will not enforce it. The reviewers must ensure that the standards above are maintained, and update the style guide accordingly.

.. We therefore opt for a less strict autoformatter in favour of a style-guide. Strict autoformatters ensure consistency, but at the detriment to readability. 

    
.. todo::
    * Prevent commits unless all tests pass.
    * Convert setup.cfg to pyproject.toml (if using black - does not support setup.cfg)
    * Consider using style-guide instead of black (i.e. autopep8) - black ensures consistency at the detriment of readability. 
    * Add dev-requirements file. ``flake8``, ``flake8-docstrings``, ``mypy``, ``black``, ``isort``. 
    * Function arguments on individual lines may be preferred to make diffs slightly clearer. But I recommend writing a comprehensive style-guide (by extending the above) rather than using a strict autoformatter like black. 


Linting
-------
Flake8 includes linting, syntax errors, and McCabe function complexity analysis. 

The are several instances where Flake8 errors have been purposely ignored in Detectree2 using ``noqa: <CODE>`` annotations to allow flake8 CI to pass. This is not a permanent fix and the errors should eventually be addressed. For example: ``noqa: E501`` ensures that line lengths beyond (120 characters) are ignored by the linter and ``noqa: 901`` ignores the McCabe complexity measure. 

These can also be set globally in setup.cfg, but fewer the better. It is also possible to set ``continue-on-error`` in the flake8 workflow or ``--exit-zero`` flake8 argument to allow other checks to continue. In practice it was found that developers tend to ignore flake8 errors as a result of these two options, so the ``noqa`` solution is preferred. 

McCabe function complexity analysis is useful for detecting over-complex code (as determined by the amount of branching - `if`, `else` statements). A value of 10 is set as default. This is perhaps overkill and may be removed. 

Docstrings
----------

We adopt google docstrings (`<https://google.github.io/styleguide/pyguide.html>`_)

Other dependencies include ``flake8-docstrings``, 

.. todo::
    * Remove ``pydocstyle``

Autoformatters
--------------
We adopt ``yapf`` for this project, but others are listed for completion. 

YAPF
^^^^
From the `YAPF docs <https://github.com/google/yapf>`_:

    Most of the current formatters for Python --- e.g., autopep8, and pep8ify --- are made to remove lint errors from code. This has some obvious limitations. For instance, code that conforms to the PEP 8 guidelines may not be reformatted. But it doesn't mean that the code looks good.

YAPF is highly customisable and shares a similar philosophy to ``black``. It is possible to customise behaviour of any autoformatter like ``autopep8`` or ``black`` with  project modifications. 

Black
^^^^^

From the `Black docs <https://black.readthedocs.io/en/stable/>`_:

    Black is the uncompromising Python code formatter. By using it, you agree to cede control over minutiae of hand-formatting. In return, Black gives you speed, determinism, and freedom from pycodestyle nagging about formatting. You will save time and mental energy for more important matters.

It favours consistency, meaning it is guaranteed to give the same results across the team - a style guide is not needed. 


Autopep8
^^^^^^^^
Autopep8 is an autoformatter (like Black) with enforces the ``PEP8`` style guide. Autopep8 is a loose formatter, which will fix PEP8 errors but will not make the code uniform. It relies a little more on the programmer, whereas ``black``, which also produces PEP8 compatible code, is more opinionated in its approach.::

    pip install --upgrade autopep8 # if not already installed
    autopep8 --in-place --aggressive --aggressive <filename>

It is possible to configure vscode to autoformat with ``autopep8`` on save if desired. 

.. todo::
    * Consider configuring YAPF with pep8 settings to create unformity for project contributors.


Formatting aside
^^^^^^^^^^^^^^^^
The difference between the strict autoformatter ``yapf`` and the official demonstrated for function arguments with the example below. Both examples are PEP8 compliant and will pass ``flake8`` linting checks. The former is better for diffs and typing clarity, whereas the latter has fewer lines. 

.. code-block:: python3

    # black or yapf:
    def tile_data(
        data: DatasetReader,
        out_dir: str,
        buffer: int = 30,
        tile_width: int = 200,
        tile_height: int = 200,
        dtype_bool: bool = False
    ) -> None:

    
    # Python's official `style`

    def tile_data(data: DatasetReader, out_dir: str, buffer: int = 30, tile_width: int = 200, tile_height: int = 200,
                  dtype_bool: bool = False) -> None:


Static typing
-------------

From the `Mypy docs <http://mypy-lang.org/>`_:

    Mypy is an optional static type checker for Python that aims to combine the benefits of dynamic (or 'duck') typing and static typing. Mypy combines the expressive power and convenience of Python with a powerful type system and compile-time type checking.

The general idea is to add typing to functions that are most frequently used. It is not necessary to apply across the entire codebase.

The `mypy` syntax adopted in Detectree2 supports python3.7 and above, but could be updated as the project moves towards more modern python3 (I see no reason not to adopt python 3.10). `mypy` will attempt to type check all third-party libraries - which might not be desirable. It is possible to install stubs for third-party libraries (i.e. ``pandas``, ``openCV``) if type-checking is desired, but it is easier to suppress all missing import errors libraries by adding  ``ignore_missing_imports = True`` in ``setup.cfg``.

======================
Continuous integration
======================

The idea of Continuous integration (CI) is to frequently commit code to a shared repo. This has the effect of detecting errors sooner thereby reducing the amount of code a developer needs to debug when finding an error. Frequent updates also make it easier to merge changes from different members of the software development team. This is especially powerful when paired automated code building and testing. Testing can include code linters, as well as unit and integration tests. 

Building and testing code requires a server. CI using GitHub actions offers workflows that can build the repository code and run tests. We can run on GitHub's own virtual machines (using GitHub-hosted runners), or on machines that we host ourselves (or on compute clusters). The latter is desirable as GitHub does not currently support access to GPU resources.

Currently there are three files that steer workflows. The schedule is set at the top of the file. The workflows are found `here <https://github.com/PatBall1/detectree2/tree/master/.github/workflows>`_

- ``python-app.yml``: All style CI - builds the code on Ubuntu-20.04
- ``dockertest.yml``: All style CI - uses docker image for dependencies and installs detectree2 using pip.
- ``documentation.yml``: Generates documentation and hosts on github pages. Builds code first for sphinx-apidoc. 

The ``dockertest.yml`` workflow is an attempt to utilise docker to speed up deployment and testing of detectree2. It pulls the docker image: `ma595/detectree-cpu-20.04:latest <https://hub.docker.com/repository/docker/ma595/detectree-cpu-20.04>`_ (Python3.8) and installs detectree2 on top. A more up to date docker container, utilising python3.10 and ubuntu 22.04 has been successfully built but has yet to be integrated into the workflow, the file can be found in `github:ma595/detectree2-docker/Dockerfile-22.04 <https://github.com/ma595/detectree2-docker/blob/main/files/Dockerfile-22.04>`_.

All dockerfiles are in `github:ma595/detectree2-docker <https://github.com/ma595/detectree2-docker>`_, which uses `github:ma595/detectree2-data <https://github.com/ma595/detectree2-data>`_ to store the data required for the workflow.


.. todo::

    - Harmonise sphinx.yml and python-app.yml into single file where appropriate. There is no good reason to separate.
    - Add GPU testing to workflow (currently unsupported on Github, but we can use CSD3's A100 resources).
    - Prevent merge unless all tests are passing
    - Build docker image as part of an action and push to dockerhub (or use github's docker features)
    - Check 22.04 docker image
    - Move dockerfiles into detectree2 project. 
    - Style check documentation.


=======================
Automatic Documentation
=======================

Documentation is generated automatically using Sphinx and GitHub actions in `documentation.yaml <https://github.com/PatBall1/detectree2/blob/master/.github/workflows/documentation.yaml>`_. 

Documentation can be generated locally to test rendering. It is better to develop locally rather than rely on the CI and hosted docs as a check, as it can take quite some time to build using the workflow. 

To generate locally it is necessary to install the following dependencies (either in pip or conda)::

    pip install sphinx sphinx_rtd_theme

Then generate api documentation, and build the html.::

    sphinx-apidoc -o ./docs/source/ detectree2/
    sphinx-build -b html docs/source/ docs/build/html

Then using your favourite browser open docs/build/html/index.html. It's often necessary to delete the build output to remove old html.

.. todo::

    * Style checks on documentation. 

=====
Tests
=====

Test-driven development stipulates that tests should be written as new features are introduced to the code. To run the tests simply do::

    # install Pytest if haven't already done so.
    pip install pytest
    # pytest should be run from the project root: 
    pytest . 

It can sometimes be helpful to run individual tests::

    pytest -rP -v detectree2/tests/test_preprocessing.py -k test_to_traintest_folders

Which runs the *test_to_traintest_folders* function in the test_preprocessing.py module, and captures whatever output that may be produced.

A few unit tests have been implemented but there is definitely scope to add more. *test_preprocessing.py* executes a few functions like tile_data_train, and to_traintest_folders, but does not test if the output is correct. ``pytest.mark.order`` is used to force dependencies but better approaches may exist. Further TODOs are listed within the code.

*test_prediction.py* executes some quite trivial tests. *test_evaluation.py* computes the area intersection over union (with dummy .geojson data containing square shapes with known areas). The test is still incomplete as some of the code in evaluation.py needs refactoring slightly. 

As of August 2022, an integration test has been written which demos the tiling, and training steps. The integration test will run the training on the CPU only. It is possible to run tests on other systems using GitHub, but this will take more work. 

TIP: Always write tests for newly introduced logic when contributing code.

.. todo:: 

    * Write more unit tests for existing code. 

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
-----------------------------------

In the `matt/conda <https://github.com/PatBall1/detectree2/tree/matt/conda>`_ branch, the `conda/meta.yaml <https://github.com/PatBall1/detectree2/blob/matt/conda/conda/meta.yaml>`_ packages detectree2. An initial attempt can be found here: `ma595/detectree2 <https://anaconda.org/ma595/detectree2>`_. To install, do the following::

    conda install -c ma595 detectree2

To rebuild from meta.yaml::

    conda-build . -c conda-forge -c pytorch --output-folder ./conda-bld

Then upload to anaconda::

    anaconda login
    anaconda upload <path-to-tar.bz2>

.. todo::
    * Automate distribution of package to `anaconda <https://anaconda.org/ma595/detectree2>`_ using workflow. 

==============================
Python development environment
==============================

.. todo::
    * Setting up visual studio. 
    * Create ``dev-environment.yaml`` file.



.. TODO list
.. ---------
.. .. todolist::