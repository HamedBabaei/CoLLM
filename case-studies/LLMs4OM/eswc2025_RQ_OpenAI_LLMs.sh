#!/bin/bash
approach="rag"
encoder="rag"
use_all_encoders=False
use_all_models=False
load_from_json=True
device="cuda"
do_evaluation=False
nshots=3
batch_size=64
llm_confidence_th=0.7
ir_score_threshold=0.9

cd ..


run_experiment() {
  local dataset_name=$1
  local approach_encoders_to_consider=$2
  local models=("${!3}")
  local outputs_names=("${!4}")

  for outputs in "${outputs_names[@]}"; do
    for models_to_consider in "${models[@]}"; do
      python -c "
from ontomap import OMPipelines
args = {
    'approach': '$approach',
    'encoder': '$encoder',
    'use-all-encoders': $use_all_encoders,
    'approach-encoders-to-consider': '$approach_encoders_to_consider',
    'use-all-models': $use_all_models,
    'models-to-consider': $models_to_consider,
    'load-from-json': $load_from_json,
    'device': '$device',
    'do-evaluation': $do_evaluation,
    'outputs-dir': '$outputs',
    'batch-size': '$batch_size',
    'nshots': '$nshots',
    'track': '$dataset_name',
    'llm_confidence_th': '$llm_confidence_th',
    'ir_score_threshold': '$ir_score_threshold'
}
print(f'Arguments are: {args}')
runner = OMPipelines(**args)
runner()
"
      sleep 3700
    done
  done
}

# Define models and outputs for each RQ and dataset
rq1_models=("['ChatGPTOpenAIAdaRAG']")
rq2_models=("['ChatGPTOpenAIAdaRAGV2']")

anatomy_rq1_outputs=("outputs-rag-r1-anatomy-run-1" "outputs-rag-r1-anatomy-run-2" "outputs-rag-r1-anatomy-run-3")
anatomy_rq2_outputs=("outputs-rag-r2-anatomy-run-1" "outputs-rag-r2-anatomy-run-2" "outputs-rag-r2-anatomy-run-3")

bioml_rq1_outputs=("outputs-rag-r1-bio-ml-run-1" "outputs-rag-r1-bio-ml-run-2" "outputs-rag-r1-bio-ml-run-3")
bioml_rq2_outputs=("outputs-rag-r2-bio-ml-run-1" "outputs-rag-r2-bio-ml-run-2" "outputs-rag-r2-bio-ml-run-3")

################################################## RUN EXPERIMENTS #################################################
# Anatomy Dataset - RQ1
run_experiment "anatomy" "['label']" rq1_models[@] anatomy_rq1_outputs[@]

# Anatomy Dataset - RQ2
run_experiment "anatomy" "['label']" rq2_models[@] anatomy_rq2_outputs[@]

# Bio-ML Dataset - RQ1
run_experiment "bio-ml" "['label-children']" rq1_models[@] bioml_rq1_outputs[@]

# Bio-ML Dataset - RQ2
run_experiment "bio-ml" "['label-children']" rq2_models[@] bioml_rq2_outputs[@]
