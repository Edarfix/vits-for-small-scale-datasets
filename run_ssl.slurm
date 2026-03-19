#!/bin/bash
#SBATCH --job-name=ssl-vit
#SBATCH --gres=gpu:2
#SBATCH --output=slurm-%j.out

uv run torchrun --nproc_per_node=1 train_ssl.py \
  --arch vit \
  --dataset CIFAR10 --image_size 32 \
  --patch_size 4 \
  --mlp_head_in 192 \
  --local_crops_number 8 \
  --local_crops_scale 0.2 0.5 \
  --global_crops_scale 0.7 1. \
  --out_dim 1024 \
  --batch_size_per_gpu 256 \
  --output_dir "./checkpoint" \
  --num_workers 2