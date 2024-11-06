!/bin/bash
model_names=("gpt4_0613" "gpt4_1106_preview" "gpt4_0125_preview" "gpt4_turbo_2024_04_09" "o1preview")
rqs=("rq1" "rq2" "rq2" "rq2" "rq3")
runs=("r1" "r2" "r3")
datasets=("umls")
task="B"


 for dataset in "${datasets[@]}"; do
     for i in "${!model_names[@]}"; do
         model_name="${model_names[$i]}"
         rq="${rqs[$i]}"
         for run_no in "${runs[@]}"; do
             echo "Running task_b.py with model_name=$model_name, dataset=$dataset, run_no=$run_no, rq=$rq"
             python task_bc.py --model_name="$model_name" --task="$task" --dataset="$dataset" --run_no="$run_no" --rq="$rq"
             echo "Going for sleep for approximately 1 hours"
             sleep 3700
         done
     done
 done
