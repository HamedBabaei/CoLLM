#!/bin/bash

# Global variables
approach="rag"
encoder="rag"
use_all_encoders=False
use_all_models=False
load_from_json=True
device="cuda"
do_evaluation=False
nshots=3
batch_size=8
llm_confidence_th=0.7
ir_score_threshold=0.9

cd ..

run_evaluation() {
  local models=$1
  local approach_encoders=$2
  local dataset_name=$3
  local output_prefix=$4

  local outputs_names=("outputs-rag-${output_prefix}-${dataset_name}-run-1" "outputs-rag-${output_prefix}-${dataset_name}-run-2" "outputs-rag-${output_prefix}-${dataset_name}-run-3")

  for outputs in "${outputs_names[@]}"; do
    for models_to_consider in "${models[@]}"; do
      python -c "
from ontomap import OMPipelines
args = {
    'approach': '$approach',
    'encoder': '$encoder',
    'use-all-encoders': $use_all_encoders,
    'approach-encoders-to-consider': $approach_encoders,
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
    done
  done
}


# MSE
run_evaluation "['LLaMA7BBertRAG']" "['label-children']" "mse" "r1"
run_evaluation "['LLaMA7BLLMBertRAGV30', 'LLaMA7BLLMBertRAGV31']" "['label-children']" "mse" "r2"
run_evaluation "['MistralLLMBERTRAGV3']" "['label-children']" "mse" "r3"
# CommonKG
run_evaluation "['LLaMA7BAdaRAG']" "['label']" "commonkg" "r1"
run_evaluation "['LLaMA7BLLMBertRAGV30', 'LLaMA7BLLMBertRAGV31']" "['label']" "commonkg" "r2"
run_evaluation "['MistralLLMBERTRAGV3']" "['label']" "commonkg" "r3"
# Biodiv
run_evaluation "['MistralAdaRAG']" "['label']" "biodiv" "r1"
run_evaluation "['MistralLLMAdaRAGV2', 'MistralLLMAdaRAGV3']" "['label']" "biodiv" "r2"
run_evaluation "['MistralLLMBERTRAGV3']" "['label']" "biodiv" "r3"
# Anatomy
run_evaluation "['MistralLLMBERTRAGV3']" "['label']" "anatomy" "r3"
# Bio-ML
run_evaluation "['MistralLLMBERTRAGV3']" "['label-children']" "bio-ml" "r3"
# Phenotype
run_evaluation "['MistralBERTRAG']" "['label-parent']" "phenotype" "r1"
run_evaluation "['MistralLLMBERTRAGV2', 'MistralLLMBERTRAGV3']" "['label-parent']" "phenotype" "r2"
run_evaluation "['LLaMA7BLLMBertRAGV31']" "['label-parent']" "phenotype" "r3"
