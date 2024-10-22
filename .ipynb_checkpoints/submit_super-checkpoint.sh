#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --gpus=1
#SBATCH --mem=25G
#SBATCH --mail-user=mwh1998@byu.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# some helpful debugging options
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate "ToMae"

export TOMOGRAM_PATH="/home/mwh1998/fsl_groups/grp_tomo_db1_d3/compute/TomoDB1_d3/Hneptunium_secretin/aba2006-11-01-6/Hyphomonas_10bin_full.rec"
# export TOMOGRAM_PATH="/home/mwh1998/fsl_groups/grp_tomo_db1_d1/nobackup/archive/TomoDB1_d1/FlagellarMotor_P1/Hylemonella gracilis/yc2012-09-23-13/20120923_Hylemonella_10003_full.rec"

python segment_tomograms.py 

