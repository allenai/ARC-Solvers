#!/bin/bash

set -e

# TODO Replace with input argument
input_file=data/ASQ-Additional/ASQ-AdditionalRnd100-Test.jsonl
input_file_with_hits=${input_file%.jsonl}_with_hits.jsonl
input_file_as_entailment=${input_file%.jsonl}_as_entailment.jsonl
entailment_predictions=${input_file%.jsonl}_predictions.jsonl
qa_predictions=${input_file%.jsonl}_qapredictions.jsonl

# Collect HITS from ElasticSearch for each question + answer choice
if [ ! -f $input_file_with_hits ]; then
	python asq_solvers/processing/add_retrieved_text.py \
		$input_file \
		$input_file_with_hits
fi

# Convert the dataset into an entailment dataset i.e. add "premise" and "hypothesis" fields to
# the JSONL file where premise is the retrieved HIT for each answer choice and hypothesis is the
# question + answer choice converted into a statement.
if [ ! -f $input_file_as_entailment ]; then
	python asq_solvers/processing/convert_to_entailment.py \
		$input_file_with_hits \
		$input_file_as_entailment
fi

# Compute entailment predictions for each premise and hypothesis
if [ ! -f $entailment_predictions ]; then
	python asq_solvers/run.py predict_custom \
	--overrides "dataset_reader.type=decompatt" \
	--output-file $entailment_predictions --silent \
	data/models/decompatt/model.tar.gz $input_file_as_entailment
fi

# Compute qa predictions by aggregating the entailment predictions for each question+answer
# choice (using max)
if [ ! -f $qa_predictions ]; then
	python asq_solvers/processing/evaluate_predictions.py \
		$entailment_predictions \
		$input_file \
		$qa_predictions
fi