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

The following section provides a walkthrough of using git/GitHub to 

Possible improvements:
- Prevent commit direct to master.




Current repository setup (TO CHANGE)
------------------------------------


Most of the below can be seen in the .github/workflows/*.yaml files. If anything is unclear about style adopted / 


Style
-----
Detectree2 currently utilises the following tools for code checks:

- ``autopep8``: Ensure consistent formatting of Python files 
- ``mypy``: Validate Python type hints 
- ``flake8``: Multiple checks for - linting - syntax errors or anti-patterns - (lack of) executable flags on files - docstring validation. Function complexity. 


The code checks are triggered automatically on 
this is steered using github actions 
with settings in setup.cfg


different components of flake8 - mccabe, linting. 

google docstrings

Automatic Documentation
-----------------------
Documentation is generated automatically using Sphinx and github actions (point to the relevant workflow here). 

It is good to generate the documentation locally before running the CI as it can take quite some time to build and test. 

The Documentation


Tests
-----
don't merge code unless all tests are passing,
always write tests for newly introduced logic when contributing code,


Submitting a PR
---------------

Collaborator vs Contributor privileges 

Fork repo etc. 

Squash commits on merge. 

Continuous integration
----------------------

Setting up github actions


Deployment issues
-----------------


Requires headers to be installed 
Must point to the location of these headers.
GDAL version must match https://github.com/OSGeo/gdal/issues/2293

Solution:

Use Conda



Python development environment
==============================


