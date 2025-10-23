#!/bin/bash

#SBATCH --job-name=lung_low_risk
#SBATCH --qos=share
#SBATCH --partition=share
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --mail-user=stevenz3@andrew.cmu.edu
#SBATCH --mail-type=END


source ~/miniconda3/etc/profile.d/conda.sh
conda activate med

# export WANDB_DISABLED="true"
# export NCCL_DEBUG=INFO
# export NCCL_P2P_DISABLE=1

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

python full_code.py --model_name pythia-160m --train_data_file /data/stevenz3/EHR/pickle/0.85/train.pkl --eval_data_file /data/stevenz3/EHR/pickle/0.85/valid.pkl --pretrained_model_path /data/stevenz3/EHR/CatchFM-160m/lit_model.pth --output_dir /data/stevenz3/EHR/finetune/ --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --learning_rate 1e-5 --warmup_ratio 0.1 --decay_ratio 0.1 --weight_decay 0.01 --max_gradient_norm 1.0 --gradient_accumulation_steps 1 --num_train_epochs 5 --dataloader_num_workers 8 --logging_steps 1 --num_eval_per_epoch 2 --devices 1 --is_test False --seed 35

# source ~/miniconda3/etc/profile.d/conda.sh &&
# conda activate med &&

# # export WANDB_DISABLED="true"
# # export NCCL_DEBUG=INFO
# # export NCCL_P2P_DISABLE=1

# export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# FLOPS=2e21
# MODEL_NAME=pythia-2.4b
# TRAIN_DATA_FILE=/data/stevenz3/EHR/train.pkl
# VALID_DATA_FILE=/data/stevenz3/EHR/valid.pkl
# TEST_DATA_FILE=/home/liwens/healthcare/Lightning-Pretrain/data/fujen/cancer_lung_low_risk/test_fujenall.pkl


# OUTPUT_DIR=/data/stevenz3/EHR/finetune/
# PRETRAINED_MODEL=/data/stevenz3/EHR/CatchFM-160m/lit_config.json


# TRAIN_BATCH_SIZE=16
# EVAL_BATCH_SIZE=16


# LEARNING_RATE=1e-5
# WARMUP_RATIO=0.1
# DECAY_RATIO=0.1
# WEIGHT_DECAY=0.01
# ACCUM_STEP=1
# EPOCH=5
# NUM_WORKER=8
# GRAD_NORM=1.0


# LOG_STEP=1
# NUM_EVAL=2
# DEVICES=8

# # i=56
# # i=56
# for i in  {35,42,49}
# do
# echo "start training on seed $i"

# export WANDB_NAME="Foundation Model Finetuning ${MODEL_NAME} flop=${FLOPS} cancer_lung_low_risk  on seed $i "

# srun python3 finetune/full_code.py \
#     --model_name $MODEL_NAME \
#     --train_data_file $TRAIN_DATA_FILE \
#     --eval_data_file $VALID_DATA_FILE \
#     --pretrained_model_path $PRETRAINED_MODEL \
#     --output_dir $OUTPUT_DIR \
#     --per_device_train_batch_size $TRAIN_BATCH_SIZE \
#     --per_device_eval_batch_size $EVAL_BATCH_SIZE \
#     --learning_rate $LEARNING_RATE  \
#     --warmup_ratio $WARMUP_RATIO \
#     --decay_ratio $DECAY_RATIO \
#     --weight_decay $WEIGHT_DECAY \
#     --max_gradient_norm $GRAD_NORM \
#     --gradient_accumulation_steps $ACCUM_STEP \
#     --num_train_epochs $EPOCH  \
#     --dataloader_num_workers $NUM_WORKER \
#     --logging_steps $LOG_STEP \
#     --num_eval_per_epoch $NUM_EVAL \
#     --devices $DEVICES \
#     --is_test False \
#     --seed $i &&

# # echo "start testing on seed $i"

# # python3 finetune/full_code.py \
# #     --model_name $MODEL_NAME \
# #     --train_data_file $TRAIN_DATA_FILE \
# #     --eval_data_file $TEST_DATA_FILE \
# #     --output_dir $OUTPUT_DIR \
# #     --per_device_train_batch_size $TRAIN_BATCH_SIZE \
# #     --per_device_eval_batch_size $EVAL_BATCH_SIZE \
# #     --learning_rate $LEARNING_RATE  \
# #     --warmup_ratio $WARMUP_RATIO \
# #     --decay_ratio $DECAY_RATIO \
# #     --weight_decay $WEIGHT_DECAY \
# #     --max_gradient_norm $GRAD_NORM \
# #     --gradient_accumulation_steps $ACCUM_STEP \
# #     --num_train_epochs $EPOCH  \
# #     --dataloader_num_workers $NUM_WORKER \
# #     --logging_steps $LOG_STEP \
# #     --num_eval_per_epoch $NUM_EVAL \
# #     --devices 1 \
# #     --is_test True \
# #     --seed $i 
# # done