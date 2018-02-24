#!/bin/bash
set -e

# ----------------------------------------
# Evaluate a BiDAF model on the QA dataset
# ----------------------------------------

input_file=$1
bidaf_model=$2
# Set this to name your run
run_name=default

if [ -z $bidaf_model ] ; then
  echo "USAGE: ./scripts/evaluate_bidaf.sh question_file.jsonl bidaf_model.tar.gz"
  exit 1
fi

input_file_prefix=${input_file%.jsonl}
model_name=$(basename ${bidaf_model%.tar.gz})

# File containing retrieved HITS per choice (using the key "support")
input_file_with_hits=${input_file_prefix}_with_hits_${run_name}.jsonl

# File with all the HITS combined per question (using the key "para")
bidaf_input=${input_file_prefix}_with_paras_${run_name}.jsonl

# Collect HITS from ElasticSearch for each question + answer choice
if [ ! -f ${input_file_with_hits} ]; then
	python asq_solvers/processing/add_retrieved_text.py \
		${input_file} \
		${input_file_with_hits}
fi

# Merge HITS for each question
if [ ! -f ${bidaf_input} ]; then
	python asq_solvers/processing/convert_to_para_comprehension.py \
	${input_file_with_hits} \
	${input_file} \
	${bidaf_input}
fi

# Run BiDafModel
bidaf_output=${input_file_prefix}_qapredictions_bidaf_${model_name}_${run_name}.jsonl
if [ ! -f ${bidaf_output} ]; then
	python asq_solvers/run.py predict \
				--output-file ${bidaf_output} --silent \
				${bidaf_model} \
				${bidaf_input}
fi

python asq_solvers/processing/produce_scores.py ${bidaf_output}
