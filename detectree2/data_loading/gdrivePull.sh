#!/bin/bash
#SBATCH --partition=high-mem 
#SBATCH -o %j.out 
#SBATCH -e %j.err
#SBATCH --time=48:00:00
#SBATCH --mem=128GB


conda activate gediee
#pip install -e /gws/nopw/j04/forecol/jgcb3/gedi/gedi_ee/

DRIVE_FOLDER=Manuscripts/forecasting
DOWNLOAD_FOLDER=/home/users/patball/forecol/jgcb3/forecasting

python -u /gws/nopw/j04/forecol/jgcb3/gedi/gedi_ee/src/data/gee_download.py $DRIVE_FOLDER $DOWNLOAD_FOLDER > dwnld_gedi_${SLURM_JOB_ID}.txt