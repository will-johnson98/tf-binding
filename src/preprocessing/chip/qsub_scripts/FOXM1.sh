#!/bin/bash
#$ -N FOXM1
#$ -pe smp 6
#$ -l mem_free=16G
#$ -cwd 
#$ -o /data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/src/preprocessing/chip/qsub_logs/FOXM1.out
#$ -j y
#$ -V


source ~/.bashrc

echo "activating conda environment"
conda activate processing

cd /data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/src/preprocessing/chip

# running basic_tf.sh
./basic_tf.sh FOXM1

