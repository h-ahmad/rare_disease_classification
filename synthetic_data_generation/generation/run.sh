#!/bin/bash

# Load conda setup in Git Bash
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the base environment (used by Spyder)
conda activate common

echo "Using Python at: $(which python)"
python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"

#accelerate config

#GPU="$1"
GPU=1
N_SET_SPLIT=5
#SPLIT_IDX="$2"
SPLIT_IDX=2


BS=10
NIPC=500
SD="sd2.1"
GS=2.0

N_SHOT=135
N_TEMPLATE=1

MODE="datadream"
DD_LR=1e-4
DD_EP=300

DATASET="scabies" # eurosat
IS_DATASETWISE=False
FEWSHOT_SEED="seed0"



# for DATASET in "${DATASETS[@]}"; do
# remove keyword 'set' in linux and remove new line.
set CUDA_VISIBLE_DEVICES=$GPU 
python main.py \
--bs=$BS \
--n_img_per_class=$NIPC \
--sd_version=$SD \
--mode=$MODE \
--guidance_scale=$GS \
--n_shot=$N_SHOT \
--n_template=$N_TEMPLATE \
--dataset=$DATASET \
--n_set_split=$N_SET_SPLIT \
--split_idx=$SPLIT_IDX \
--fewshot_seed=$FEWSHOT_SEED \
--datadream_lr=$DD_LR \
--datadream_epoch=$DD_EP \
--is_dataset_wise_model=$IS_DATASETWISE \

# done

