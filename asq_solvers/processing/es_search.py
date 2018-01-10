from typing import Dict, List

from elasticsearch import Elasticsearch
import re


class EsSearch:
    def __init__(self,
                 es_client: str = "localhost",
                 indices: str = "busc",
                 max_question_length: int = 1000,
                 max_hits_retrieved: int = 500,
                 max_hit_length: int = 300,
                 max_hits_per_choice: int = 100):
        """
        Class to search over the text corpus using ElasticSearch
        :param es_client: Location of the ElasticSearch service
        :param indices: Comma-separated list of indices to search over
        :param max_question_length: Max number of characters used from the question for the
        query (for efficiency)
        :param max_hits_retrieved: Max number of HITS requested from ElasticSearch
        :param max_hit_length: Max number of characters for accepted HITS
        :param max_hits_per_choice: Max number of HITS returned per answer choice
        """
        self._es_client = es_client
        self._indices = indices
        self._max_question_length = max_question_length
        self._max_hits_retrieved = max_hits_retrieved
        self._max_hit_length = max_hit_length
        self._max_hits_per_choice = max_hits_per_choice
        # Regex for negation words used to ignore Lucene results with negation
        self._negation_regexes = [re.compile(r) for r in ["not\\s", "n't\\s", "except\\s"]]

    def get_hits_for_question(self, question: str, choices: List[str]) -> Dict[str, List[str]]:
        """
        :param question: Question text
        :param choices: List of answer choices
        :return: Dictionary of HITS per answer choice
        """
        choice_hits = dict()
        for choice in choices:
            choice_hits[choice] = self.filter_hits(self.get_hits_for_choice(question, choice))
        return choice_hits

    # Constructs an ElasticSearch query from the input question and choice
    def construct_qa_query(self, question, choice):
        return {"from": 0, "size": self._max_hits_retrieved,
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"text": question[-self._max_question_length:]}}
                        ],
                        "filter": [
                            {"match": {"text": choice}}
                        ]
                    }
                }}

    # Retrieve unfiltered HITS for input question and answer choice
    def get_hits_for_choice(self, question, choice):
        es = Elasticsearch([self._es_client], retries=3)
        res = es.search(index=self._indices, body=self.construct_qa_query(question, choice))
        # print("Got {} Hits for choice: {}".format(res['hits']['total'], choice))
        hits = [es_hit["_source"]["text"] for es_hit in res['hits']['hits']]
        return hits

    # Remove HITS that contain negation, are too long, are duplicates, are noisy.
    def filter_hits(self, hits: List[str]) -> List[str]:
        filtered_hits = []
        selected_hit_keys = set()
        for hit_sentence in hits:
            hit_sentence = hit_sentence.strip().replace("\n", " ")
            if len(hit_sentence) > self._max_hit_length:
                continue
            for negation_regex in self._negation_regexes:
                if negation_regex.search(hit_sentence):
                    # ignore hit
                    continue
            if self.get_key(hit_sentence) in selected_hit_keys:
                continue
            if not self.is_clean_sentence(hit_sentence):
                continue
            filtered_hits.append(hit_sentence)
            selected_hit_keys.add(self.get_key(hit_sentence))
        return filtered_hits[:self._max_hits_per_choice]

    # Check if the sentence is not noisy
    def is_clean_sentence(self, s):
        # must only contain expected characters, should be single-sentence and only uses hyphens
        # for hyphenated words
        return re.match("^[a-zA-Z0-9][a-zA-Z0-9;:,\(\)%\-\&\.'\"\s]+\.?$", s) is not None and \
               re.match(".*\D\. \D.*", s) is None and \
               re.match(".*\s\-\s.*", s) is None

    # Create a de-duplication key for a HIT
    def get_key(self, hit):
        return re.sub('[^0-9a-zA-Z\.\-^;&%]+', '', re.sub('http[^ ]+', '', hit)).strip().rstrip(".")
