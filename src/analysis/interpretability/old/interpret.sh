#!/bin/bash
#$ -l mem_free=8G
#$ -wd /data1/home/rreid
#$ -pe smp 4


#cd $PBS_O_WORKDIR
/data1/home/rreid/miniconda3/condabin/conda activate processing
python /data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/src/inference/interpretability/interpretability.py