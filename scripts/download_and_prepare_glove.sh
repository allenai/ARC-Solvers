#!/usr/bin/env bash

EMBEDDINGS_DIR=data/glove

wget -O glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip

unzip glove.840B.300d.zip && gzip glove.840B.300d.txt

mkdir -p ${EMBEDDINGS_DIR}
mv glove.840B.300d.txt.gz ${EMBEDDINGS_DIR}/glove.840B.300d.txt.gz