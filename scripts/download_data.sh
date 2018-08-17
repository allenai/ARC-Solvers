#!/bin/bash

# Fail if any command fails
set -e
set -x

# NOTE: Make sure ElasticSearch v6+ is running on ES_HOST. Update es_search.py if you are not
# running ElasticSearch on your localhost
ES_HOST="localhost"

mkdir -p data/
cd data/

QUESTIONS_URL="https://s3-us-west-2.amazonaws.com/ai2-website/data/ARC-V1-Feb2018.zip"
MODELS_URL="https://s3-us-west-2.amazonaws.com/ai2-website/data/ARC-V1-Models-Aug2018.zip"

# Download the questions
wget $QUESTIONS_URL
unzip $(basename $QUESTIONS_URL)
mv ARC-V1-Feb2018-2 ARC-V1-Feb2018
rm -rf __MACOSX

# Download the model
wget $MODELS_URL
unzip $(basename $MODELS_URL)

cd ..

# Build the index
python scripts/index-corpus.py \
	data/ARC-V1-Feb2018/ARC_Corpus.txt \
	arc_corpus \
	$ES_HOST
