#!/bin/bash -l

#$ -P cs542
#$ -l h_rt=24:00:00
#$ -N baselineModel
#$ -l gpus=1 -l gpu_c=3.5


module load python/3.6.2
module load cuda/9.1
module load cudnn/7.1
module load tensorflow/r1.8


python /projectnb/cs542/wchapman/seizure_prediction_ml/baselineModel.py
