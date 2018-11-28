#!/bin/bash -l

#$ -P cs542
#$ -l h_rt=24:00:00
#$ -N baselineModel

module load python/3.6.2
module load tensorflow/r1.8_cpu


python /projectnb/cs542/wchapman/seizure_prediction_ml/arima/arima_nn.py

