#!/bin/bash -l

# Set SCC project
#$ -P ds598
#$ -l h_rt=30:00:00
#$ -m beas
#S -M yliu2702@bu.edu

module load miniconda
module load academic-ml/spring-2024
conda activate yliu_env

export PYTHONPATH="/projectnb/ds598/projects/yliu2702:$PYTHONPATH"

python demo_model/train.py
python demo_model/test.py

### The command below is used to submit the job to the cluster
### qsub -pe omp 4 -P ds598 -l gpus=1 demo_train.sh
### How to que a interactive GPU
### qrsh -P ds598 -l gpus=1