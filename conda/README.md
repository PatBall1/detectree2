# Conda

## Conda / mamba install
It is recommended to install mamba to speed up build process.

`mamba create --name detectenv --file conda-linux-64.lock`


## Update conda install

Re-generate Conda lock file(s) based on environment.yml

`conda-lock -k explicit --conda mamba`

Update Conda packages based on re-generated lock file

`mamba update --file conda-linux-64.lock`

