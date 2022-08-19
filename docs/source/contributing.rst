******************
WIP - Contributing guide
******************


Contributing to Detectree2
==========================

Thanks for taking the time to contribute to detectree2!

Code of Conduct
---------------

All contributors are expected to abide by the `Code of Conduct <https://github.com/PatBall1/detectree2/blob/master/CODE_OF_CONDUCT.md>`_


Getting Started - Issues
------------------------
The “to do” task list for detectree2 starts at the `Issues <https://github.com/PatBall1/detectree2/issues>`_ page in Github.


Note: Before submitting a new issue, you should search the existing issues to make sure that it hasn't already been reported.


Filing a bug report
-------------------
Bug reports are important in helping identify things that aren't working in detectree2. Filing a bug report is straightforward: Head to the Issues page, click on the New issue button, and select the Bug Report template to get started.


Using Git / GitHub
==================

The following section provides a walkthrough of using git/GitHub.

Fork the repository:
Only the primary project team has permissions to directly edit the main detectree2 repository. As an external contributor, you will need to make a personal copy of the repository (a.k.a. “fork”) to begin making changes. To create a fork, click on the Fork button at the top right of the page of the repository on Github.

This will create a copy of the detectree2 repository under your own GitHub account. (Visit “Your Repositories” under your account.)

Clone the repository: 
To access and edit your fork (copy) of the repository on your computer, you will need to clone the repository.

git clone https://github.com/<YOUR-USERNAME>/detectree2.git detectree2-<YOUR-USERNAME>
cd detectree2-<YOUR-USERNAME>
git remote add upstream https://github.com/patball1/detectree2

Checkout a new branch:

While this step is optional for forked repositories, it's generally best practice to create a new branch for each thing you're working on. Run the following command:

get checkout -b <NEW-BRANCH>

where <NEW-BRANCH> is the name of the branch. (For future contributions, replace this with a name of your choosing). This command creates the new branch and checks out the branch.

Committing your changes: 

To save this changes to Git, you must commit the files. Run the command:

git commit -m "Add emoji to README title"

where the text inside the quotes is a description of the changes you've made in this commit.

Pushing your changes: 

To push your changes online for the first time, run:

git push origin --set-upstream update-readme

where --set-upstream update-readme creates the same update-readme branch online, in your fork on GitHub.

Creating a Pull Request: To submit your changes for review, you will need to create a pull request. When a new branch is pushed to GitHub, a prompt will appear to create a pull request:

Then fill out the form with (at minimum) a title and description and create the pull request. At this point, a project team member or maintainer can review your pull request, provide comments, and officially merge it when approved.


More advanced Git Practices
---------------------------

Rebase
If you're working on some changes for a long period of time, it's possible that other contributors may have submitted other changes on the same files you're working on (see section General Tips). To sync your branch, run:

git pull origin master --rebase

and follow the prompts to approve and/or resolve the changes that should be kept.

Rebasing often proceeds as follows (taken from gitlab's docs):

Fetch the latest changes from main:

git fetch origin master

Checkout your feature branch:

git checkout my-feature-branch

Rebase it against master:

git rebase origin/master

Then force push to your branch.

git push -f origin my-feature-branch

When you rebase:
1. Git imports all the commits submitted to main after the moment you created your feature branch until the present moment.

2. Git puts the commits you have in your feature branch on top of all the commits imported from main:

It's worth bearing in mind that this can lead to issues especially when working with other people. There is a high chance of overwriting commits from your colleagues resulting in lost work. The safer alternative is to use --force-with-lease.

git push --force-with-lease origin my-feature-branch

Updating your Fork
------------------
To update the master branch of your fork (so that new branches are created off of an up-to-date master branch), run:

git fetch upstream

Pull Request Conventions
------------------------
The pull request title is used to build the release notes. Write the title in past tense, describing the extent of the changes.

Pull Requests should have labels to identify which category belongs to in the release notes. Use the exclude notes label if the change doesn't make sense to document in release notes.

Pull Requests should be linked to issues, either manually or using keywords.


Detectree2 Github tips
----------------------




Git Rebase/Merge during PR
-------------------------------

It is strongly recommended to sync master with the feature branch during the submission of a PR. One can either merge master or rebase on master to sync changes. Either is fine in practice, but for large projects, with many contributors it is considered good practice to ``rebase`` to keep the history linear.

However, I would *strongly recommend* to ``squash and merge`` the commits when committing a PR (this is done using the github UI). This combines all of the commits into one commit in the base branch. Therefore we we do not need to worry about the effects of merging in other branches on the project history. 

It is possible to make this the default behaviour in the repository settings. 

If for any reason you do not want to squash and merge the commits (i.e. to keep the PR history in tact), I would strongly recommend rebasing. 

TIP: Try to avoid merging the PR to a `dev` branch. This is considered bad practice since when it comes to merge to master the eventual `PR` can be large and difficult to understand. PRs should have a single focus. 

TIP: Delete the branch after merging to master. 

TIP: Many of the recommendations above can be made default in Github's settings:
- Prevent commits direct to master.
- Squash on merge
- Prevent merge unless all tests are passing. 
- Only allow a merge if approved by assigned Reviewers. 



Current repository setup (TO CHANGE)
------------------------------------

Most of the below can be seen in the .github/workflows/*.yaml files. Refer to these files, if there is anything unclear about style. The code checks are triggered automatically on pushing to PR branch. This is steered using github actions with settings for each component in setup.cfg. 


Style
-----
Detectree2 currently utilises the following tools for code checks:

- ``autopep8``: Ensure consistent formatting of Python files 
- ``mypy``: Validate Python type hints 
- ``flake8``: Multiple checks for - linting - syntax errors or anti-patterns - (lack of) executable flags on files - docstring validation - function complexity
- ``isort``: Checks that imports are correctly sorted

- Line length = 120 characters
- Google docstrings
- Function signatures span 

Flake8
------
Flake8 includes linting, syntax errors, and mccabe function complexity analysis. 

Flake8 errors have been purposely ignored in several places using ``noqa: <CODE>`` annotations to allow flake8 CI to pass. This is not a permanent fix and the errors should eventually be addressed. For example: ``noqa: E501`` ensures that line lengths beyond (120 characters) are ignored by the linter and ``noqa: 901`` ignores the Mccabe complexity measure. 

These can also be set globally in setup.cfg, but the fewer the better. It is also possible to set `continue-on-error` in the flake8 workflow or `--exit-zero` flake8 argument. In practice it was found that users tended to ignore flake8 errors as a result of these two options, so the ``noqa`` solution is preferred. 



docstrings
----------
We adopt google docstrings:

Other dependencies include ``flake8-docstrings``, ``pydocstyle``, 

The python code is 

Static typing
-------------
Static typing is written for compatibility with python3.7 and above. This could be updated as the project moves towards more modern python3. More settings in ``setup.cfg``. 

Workflows
---------
Currently there are three files:

- ``pythonapp.yml``
- ``dockertest.yml``
- ``documentation.yml``

``dockertest.yml`` is an attempt to utilise docker to speed up deployment of detectree2. The dockerfiles are in ``ma595/detectree2-docker``, which uses ``ma595/detectree2-data`` to store the data required for the workflow.


TODO: Harmonise all files into one. There is no good reason to separate this functionality.

Automatic Documentation
-----------------------
Documentation is generated automatically using Sphinx and github actions (point to the relevant workflow here (Documentation.yaml)). 

Documentation can be generated locally to test rendering. It is better to develop locally rather than rely on the CI and hosted docs as a check, as it can take quite some time to build using the workflow. 

The Documentation


Tests
-----
Test-driven development

TIPS: Always write tests for newly introduced logic when contributing code,
TODO: Prevent merge unless all tests are passing


Submitting a PR
---------------

Collaborator vs Contributor privileges 

Fork repo etc. 

Squash commits on merge. 

Continuous integration
----------------------

Setting up github actions

TODO:
Add GPU testing to workflow. 


Building Detectree2 using Pip
-----------------------------

It is relatively straightforward to install detectree2 on Colab. Simply pip install and all dependencies will be installed automatically. 

On other systems the process is more involved especially if root access is not available. See workflow pythonapp.yaml workflow for a working CPU deployment. 

First we need to install pytorch, torchvision and torchaudio (compatible versions https://pypi.org/project/torchvision/):

This can be done inside virtualenv (if root access is unavailable): 
python3 -m venv ./venv
. venv/bin/activate
pip install --upgrade pip
pip install wheel
pip install opencv-python
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

Then point to preinstalled GDAL header files:

export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal

then

pip install . (add -e flag to allow editable installs)

TODO: 

pin torch and torchvision versions in setup.py
https://detectron2.readthedocs.io/en/latest/tutorials/install.html
http://www.tekroi.in/detectron2/projects/DensePose/setup.py
https://stackoverflow.com/questions/66738473/installing-pytorch-with-cuda-in-setup-py

Fixing detecton2 version
------------------------
We can fix the version of detectree2 by pointing to the pre-built wheel using pip:

python -m pip install detectron2==0.6 -f \ https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

Or by changing the detectron2 line in setup.py (which will build the latest version from source):

detectron2@https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/detectron2-0.6%2Bcu113-cp38-cp38-linux_x86_64.whl

It may be preferable to do this as errors have been introduced into the detectron2 codebase and may take a day or two to fix. 
We can also point to a specific working commit:

pip install git+https://github.com/facebookresearch/detectron2.git@5aeb252b194b93dc2879b4ac34bc51a31b5aee13
detectron2@git+https://github.com/facebookresearch/detectron2.git@5aeb252b194b93dc2879b4ac34bc51a31b5aee13

GDAL complexities
-----------------
As mentioned above, GDAL presents a number of complexities. We must point to the location of the preinstalled GDAL headers, and the GDAL version must match the pip package version. https://github.com/OSGeo/gdal/issues/2293
For instance, on my cluster:

gdal-config -v 

gives, 3.0.4. So this means we must install the corresponding pip version: GDAL==3.0.4. 

In the event that GDAL does not exist on the system, install it as so (assuming root access):

sudo apt install libgdal-dev gdal-bin


Building Detectree2 using Conda
-------------------------------
Many of the aforementioned complexities can be solved using Conda. This is currently working for python 3.9.13, in branch matt/conda. 

Install miniconda, and source (usually ~/.miniconda/bin/activate if not in .bashrc already). Begin by installing mamba:

conda install mamba -c conda-forge
mamba env create -f envrironment.yaml 
mamba activate detectree2

Alternatively we may use a conda lock file which has transitive dependencies pinned. This improves reproducibility. 

mamba create --name detectree2env --file conda-linux-64.lock

and if we modify our environment, we can update the lock file as so:

conda-lock -k explicit --conda mamba

and then update conda packages based on the regenerated lock file: 

mamba update --file conda-linux-64.lock

The downside of this approach is that it takes much longer to install compared to pip, even with mamba's improved dependency resolution. 

TODO: It would be nice to eventually use poetry, because of how easy it is to package a distribution. But detectron2 is not PEP517 compliant. 

Python development environment
==============================

Setting up visual studio. 


