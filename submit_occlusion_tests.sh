#!/usr/bin/env bash

for occlusion_percentage in {0..100}
do
    sbatch test_occlusion_robustness.sh $occlusion_percentage
done