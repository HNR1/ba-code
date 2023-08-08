#!/bin/bash
#PBS -l select=1:ncpus=1:mem=15gb:ngpus=1:accelerator_model=gtx1080ti
#PBS -l walltime=09:59:00
#PBS -A "SDwithToMe"
set -e

module load PyTorch/1.13.0

export PIP_CONFIG_FILE=/software/python/pip.conf

cd /gpfs/project/hebal100/ba-code

python -m pip install --user -r scripts/requirements.txt

python src/gen_imgs.py 'data/run16' 50 768 768 'run4/logger/log_ZUxgl.csv'

