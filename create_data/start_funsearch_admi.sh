#!/bin/bash

# Usage: cd create_data && bash start_funsearch_admi.sh
# Launches 3 independent FunSearch runs in the background.

export CUDA_VISIBLE_DEVICES=0

mkdir -p results/logs

nohup python -m algodisco.methods.funsearch.main_funsearch \
    --config configs/admi_funsearch_run1.yaml \
    > results/logs/funsearch_run1.out 2>&1 &
echo "run1 PID $!"

nohup python -m algodisco.methods.funsearch.main_funsearch \
    --config configs/admi_funsearch_run2.yaml \
    > results/logs/funsearch_run2.out 2>&1 &
echo "run2 PID $!"

nohup python -m algodisco.methods.funsearch.main_funsearch \
    --config configs/admi_funsearch_run3.yaml \
    > results/logs/funsearch_run3.out 2>&1 &
echo "run3 PID $!"
