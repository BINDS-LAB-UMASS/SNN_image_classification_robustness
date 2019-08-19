#!/usr/bin/env bash
#
#SBATCH --partition=1080ti-short
#SBATCH --gres=gpu:1
#SBATCH --time=00-04:00:00
#SBATCH --mem=16000
#SBATCH --account=rkozma
#SBATCH --output=output/occlusion_%j.out
#SBATCH --cpus-per-task=8

occlusion_percentage=${1:-0}

echo $occlusion_percentage

python test_occlusion_robustness.py --occlusion_percentage $occlusion_percentage
exit
