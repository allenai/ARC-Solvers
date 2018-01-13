#!/usr/bin/python3
# This script uses the Python Elasticsearch API to index a user-specified text corpus in an 
# ElasticSearch cluster. The corpus is expected to be a text file with a sentence per line.
# Each sentence is indexed as a separate document, and per the mappings defined here, the
# Snowball Stemmer is used to stem all tokens.
# If an index with the requested name does not exists, creates it, if not simply adds
# documents to existing index. 

import argparse, elasticsearch, json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

if __name__=="__main__":

    # Arguments
    parser = argparse.ArgumentParser(description='Add lines from a file to a simple text Elasticsearch index.')
    parser.add_argument('file', help='Path of file to index, e.g. /path/to/my_corpus.txt')
    parser.add_argument('index', help='Name of index to create')
    parser.add_argument('host', help='Elasticsearch host.')
    parser.add_argument('-p','--port', default=9200, help='port, default is 9200')
    args = parser.parse_args()

    # Get Index Name
    index_name = args.index

    # Document Type constant
    TYPE = "sentence"

    # Get an ElasticSearch client
    es = Elasticsearch(hosts=[{"host":args.host, "port":args.port}])

    # Mapping used to index all corpora used in Aristo solvers
    mapping = '''
    {
      "mappings": {
        "sentence": {
          "dynamic": "false",
          "properties": {
            "docId": {
              "index": "not_analyzed",
              "type": "string"
            },
            "text": {
              "analyzer": "snowball",
              "type": "string",
              "fields": {
                "raw": {
                  "index": "not_analyzed",
                  "type": "string"
                }
              }
            },
            "tags": {
              "index": "not_analyzed",
              "type": "string"
            }
          }
        }
      }
    }'''

    
    # Function that constructs a json body to add each line of the file to index
    def make_documents(f):
        for l in f:
            doc = {
                    '_op_type': 'create',
                    '_index': index_name,
                    '_type': TYPE,
                    '_source': {'text': l.strip() }
            }
            yield(doc)


    # Create an index, ignore if it exists already
    try:
        res = es.indices.create(index=index_name, ignore=400, body=mapping)

        # Bulk-insert documents into index
        with open(args.file, "r") as f:            
            res = bulk(es, make_documents(f))
            doc_count = res[0]

        # Test Search.
        print("Index {0} is ready. Added {1} documents.".format(index_name, doc_count))
        query = input("Enter a test search phrase: ")
        result = es.search(index=index_name, doc_type=TYPE, body={"query": {"match": {"text": query.strip()}}})
        if result.get('hits') is not None and result['hits'].get('hits') is not None:
            print(result['hits']['hits'])
        else:
            print({})
   
    except Exception as inst:
        print(inst)
