#!/bin/bash

pip install -r requirements.txt

# Copied from AllenNLP
# Temporary fix to the build whilst NLTK sort stuff out.
python -m nltk.downloader -u https://pastebin.com/raw/D3TBY4Mj punkt
python -m spacy.en.download all