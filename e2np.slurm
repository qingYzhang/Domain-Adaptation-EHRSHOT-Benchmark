#!/bin/bash

#SBATCH --job-name=e2np
#SBATCH --output=e2np.out
#SBATCH --error=e2np.err
#SBATCH --partition=preempt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=250GB
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:L40S:4


# job info
train_file=create_benchmark_dataset
# train_file=ehrshot_to_nhird_patient
this_version=10


# start running
echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"
nvidia-smi

source /home/stevenz3/miniconda3/etc/profile.d/conda.sh
conda activate med

cd /data/user_data/stevenz3/EHRSHOT-Benchmark-nhird-format_qwen_baseline

python -m ${train_file}
# python -m ${train_file} --this_version ${this_version}

# python -m ${train_file} "/data/user_data/stevenz3/EHRSHOT-Benchmark-nhird-format_sfs/ehrshot_patient.json" "/data/user_data/stevenz3/EHRSHOT-Benchmark-nhird-format_sfs/ehrshot_patient_birth.json"