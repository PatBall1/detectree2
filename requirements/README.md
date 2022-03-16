# Requirements

## Dependencies structure
This directory contains the environment setup to reproduce all code in this repository. By listing all of your requirements 
in the repository you can easily track the packages needed to recreate the analysis. 

- `environment.yml`: Is the general environment specification for deployment. Per default, this automatically installs `dev-requirements.txt` into the python environment.
- `dev-requirements.txt`: PIP requirements file for the packages needed for developing code (includes convenient dependencies, linters, formatters)  
- `test-requirements.txt`: PIP requirements file for the packages needed to run continuous integration (includes linting, unit test dependencies)  
- `requirements.txt`: PIP requirements file for the packages needed to run code for deployment (minimal dependencies only)  

## Workflow
A good workflow is: 
1. `pip install` the packages that your analysis needs
2. Run `pip freeze > requirements.txt` to pin the exact package versions used to recreate the analysis
3. If you find you need to install another package, run `pip freeze > requirements.txt` again and commit the changes to version control.