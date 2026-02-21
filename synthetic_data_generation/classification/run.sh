#!/bin/bash

# Load conda setup in Git Bash
source ~/miniconda3/etc/profile.d/conda.sh
#source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate the base environment (used by Spyder)
conda activate common

echo "Using Python at: $(which python)"
python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"

#accelerate config
GPU=0 # 0, 1, 2, 3

MODEL_TYPE=clip # clip, resnet50
CLIP_DIR=models/clip_dir
CLIP_VERSION=ViT-B/32
IS_LORA_IMAGE=True
IS_LORA_TEXT=True
OUTPUT_DIR=output
DATAROOT=preprocessing
DATASET_SELECTION=scabies # matek
IS_SYNTH_TRAIN=False #True
IS_POOLED_FEWSHOT=False #True
FOLD=0  # 0, 1, 2, 3, 4, 
N_CLASSES=2
LAMBDA_1=0.5 #0.5
IS_HSV=True
IS_HED=True
IS_RAND_AUG=True
IS_MIX_AUG=True
EPOCHS=10 #10, 100, 300
WARMUP_EPOCHS=3 #3, 30
LR=1e-4
WD=1e-8 #1e-4, 1e-8
MIN_LR=1e-8
BATCH_SIZE=64
BATCH_SIZE_EVAL=1  #16 #8  #1
LOG=tensorboard #wandb

echo "==== Script execution started! ====="
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

export CUDA_VISIBLE_DEVICES=$GPU

python main.py \
--model_type=$MODEL_TYPE \
--clip_download_dir=$CLIP_DIR \
--clip_version=$CLIP_VERSION \
--is_lora_image=$IS_LORA_IMAGE \
--is_lora_text=$IS_LORA_TEXT \
--output_dir=$OUTPUT_DIR \
--dataroot=$DATAROOT \
--dataset_selection=$DATASET_SELECTION \
--is_synth_train=$IS_SYNTH_TRAIN \
--is_pooled_fewshot=$IS_POOLED_FEWSHOT \
--fold=$FOLD \
--n_classes=$N_CLASSES \
--lambda_1=$LAMBDA \
--is_hsv=$IS_HSV \
--is_hed=$IS_HED \
--is_rand_aug=$IS_RAND_AUG \
--is_mix_aug=$IS_MIX_AUG \
--epochs=$EPOCHS \
--warmup_epochs=$WARMUP_EPOCHS \
--lr=$LR \
--wd=$WD \
--min_lr=$MIN_LR \
--batch_size=$BATCH_SIZE \
--batch_size_eval=$BATCH_SIZE_EVAL \
--log=tensorboard

echo "==== Script execution finished! ====="