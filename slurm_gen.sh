#!/bin/sh
#SBATCH --job-name=paraphrase_type_gen
#SBATCH --account=etechnik_gpu
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --cpus-per-gpu 32
#SBATCH --gpus 1
#BATCH --requeue

echo $1
echo $2

export TOKENIZERS_PARALLELISM=false

source ~/.bashrc

python3 src/finetune_generation.py --task_name $2 --model_name $1
