#!/bin/bash
#SBATCH -A training2661
#SBATCH -p eval-ttbh
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:10:00
#SBATCH -J vit-temp
#SBATCH -o jureca/vit/temp/logs/%x-%j.log

source jureca/env.sh
python jureca/vit/temp/temp.py
