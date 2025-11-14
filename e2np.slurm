#!/bin/bash

#SBATCH --job-name=e2np
#SBATCH --output=e2np.out
#SBATCH --error=e2np.err
#SBATCH --partition=preempt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:A100_80GB:1

#SBATCH --mail-user=stevenz3@andrew.cmu.edu
#SBATCH --mail-type=END

# job info
train_file=create_benchmark_dataset
# train_file=ehrshot_to_nhird_patient
# train_file=build_nhird
this_version=10


# start running
echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"
nvidia-smi

source /home/stevenz3/miniconda3/etc/profile.d/conda.sh
conda activate med

cd /data/user_data/stevenz3/Domain-Adaptation-EHRSHOT-Benchmark/

python -m ${train_file}
# python -m ${train_file} --this_version ${this_version}

# python full_code.py --model_name pythia-160m \
#    --train_data_file /data/group_data/cx_healthcare/finetune_data_mixed_risk/cancer_pancreatic/train.pkl \
#    --eval_data_file /data/group_data/cx_healthcare/finetune_data_mixed_risk/cancer_pancreatic/test.pkl \
#    --pretrained_model_path /data/group_data/cx_healthcare/nhird_pretrained/flop_1e18/pythia-160m/step-4800-ckpt-converted/lit_model.pth \
#    --output_dir /data/group_data/cx_healthcare/nhird_finetune/flop_1e18/pythia-160m/cancer_pancreatic/ \
#    --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
#    --learning_rate 1e-5 --warmup_ratio 0.1 --decay_ratio 0.1 \
#    --weight_decay 0.01 --max_gradient_norm 1.0 \
#    --gradient_accumulation_steps 1 --num_train_epochs 5 \
#    --dataloader_num_workers 8 --logging_steps 1 \
#    --num_eval_per_epoch 2 --devices 1 \
#    --is_test True \
#    --seed 42


# python full_code.py --model_name pythia-160m \
#    --train_data_file /data/group_data/cx_healthcare/finetune_data_mixed_risk/cancer_pancreatic/train.pkl \
#    --eval_data_file /data/group_data/cx_healthcare/finetune_data_mixed_risk/cancer_pancreatic/valid.pkl \
#    --pretrained_model_path /data/group_data/cx_healthcare/nhird_pretrained/flop_1e18/pythia-160m/step-4800-ckpt-converted/lit_model.pth \
#    --output_dir /data/user_data/stevenz3/160m_pretained/test/ \
#    --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
#    --learning_rate 1e-5 --warmup_ratio 0.1 --decay_ratio 0.1 \
#    --weight_decay 0.01 --max_gradient_norm 1.0 \
#    --gradient_accumulation_steps 1 --num_train_epochs 5 \
#    --dataloader_num_workers 8 --logging_steps 1 \
#    --num_eval_per_epoch 2 --devices 1 --is_test False --seed 35
