!/bin/bash
model_names=("mistral" "flan_t5_small" "flan_t5_base" "flan_t5_large" "flan_t5_xl" "flan_t5_xxl")
rqs=("rq3" "rq2" "rq2" "rq2" "rq1" "rq2")
runs=("r1" "r2" "r3")
datasets=("umls")
task="C"

for dataset in "${datasets[@]}"; do
    for i in "${!model_names[@]}"; do
        model_name="${model_names[$i]}"
        rq="${rqs[$i]}"
        for run_no in "${runs[@]}"; do
            echo "Running task_b.py with model_name=$model_name, dataset=$dataset, run_no=$run_no, rq=$rq"
            python task_bc.py --model_name="$model_name" --task="$task" --dataset="$dataset" --run_no="$run_no" --rq="$rq"
        done
    done
done
