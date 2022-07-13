#!/bin/sh
#SBATCH --job-name=paraphrase_type_llama
#SBATCH --account=etechnik_gpu
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --cpus-per-gpu 8
#SBATCH --gpus 8

echo $1
echo $2

ckpt_dir="/beegfs/wahle/github/llama-2/llama-2-70b"
tokenizer_path="/beegfs/wahle/github/llama-2/tokenizer.model"

# ckpt_dir="/beegfs/wahle/llama-1-weights/7B"
# tokenizer_path="/beegfs/wahle/github/llama-1-weights/tokenizer.model"

data_file="out/detection_test.jsonl"

source ~/.bashrc

python3 -m torch.distributed.run --nproc_per_node 8 src/llama_generation.py --ckpt_dir $ckpt_dir --tokenizer_path $tokenizer_path --data_file $data_file