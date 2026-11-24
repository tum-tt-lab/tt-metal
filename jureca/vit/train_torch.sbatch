#!/bin/bash
#SBATCH -A training2661
#SBATCH -p eval-ttbh
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -J vit-torch
#SBATCH -o jureca/vit/logs/%x-%j.log

source jureca/env.sh
python jureca/vit/train_torch.py --model tiny --epochs 1
