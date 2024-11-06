#!/bin/bash

####################################################### TASK A: Term Typing #######################################################
# Dataset is SNOMEDCT_US
model_names=("bloom_560m" "bloom_1b1" "bloom_1b7" "bloom_3b" "bloom_7b1" "llama3")
rqs=("rq2" "rq2" "rq2" "rq1" "rq2" "rq3")
runs=("r1" "r2" "r3")

# Loop through each model, run number, and RQ combination
for i in "${!model_names[@]}"; do
    model_name="${model_names[$i]}"
    rq="${rqs[$i]}"
    for run_no in "${runs[@]}"; do
        echo "Running task_a.py with model_name=$model_name, run_no=$run_no, rq=$rq"
        python task_a.py --model_name="$model_name" --run_no="$run_no" --rq="$rq"
    done
done
