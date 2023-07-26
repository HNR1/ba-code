#!/bin/bash
#PBS -l select=1:ncpus=1:mem=20gb:ngpus=1:accelerator_model=gtx1080ti 
#PBS -l walltime=07:59:00
#PBS -A "SDwithToMe"
set -e
 
module load PyTorch/1.13.0

export PIP_CONFIG_FILE=/software/python/pip.conf

cd /gpfs/project/hebal100/ba-code

python -m pip install --user -r scripts/requirements.txt

python tests/test_data.py 'test/test3' 10 768 768
