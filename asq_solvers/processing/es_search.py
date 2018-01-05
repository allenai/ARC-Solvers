from elasticsearch import Elasticsearch
import re


class EsSearch:
    def __init__(self,
                 es_client: str = "localhost",
                 indices: str = "busc",
                 max_question_length:int = 1000,
                 max_hits_retrieved: int = 500,
                 max_hit_length: int = 300,
                 max_hits_per_choice: int = 100):
        self._es_client = es_client
        self._indices = indices
        self._max_question_length = max_question_length
        self._max_hits_retrieved = max_hits_retrieved
        self._max_hit_length = max_hit_length
        self._max_hits_per_choice = max_hits_per_choice
        # Regex for negation words used to ignore Lucene results with negation
        self._negation_regexes = [ re.compile(r) for r in ["not\\s", "n't\\s", "except\\s"]]

    def get_hits_for_question(self, question, choices):
        choice_hits = dict()
        for choice in choices:
            choice_hits[choice] = self.filter_hits(self.get_hits_for_choice(question, choice))
        return choice_hits

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

    def get_hits_for_choice(self, question, choice):
        es = Elasticsearch([self._es_client], retries=3)
        res = es.search(index=self._indices, body=self.construct_qa_query(question, choice))
        # print("Got {} Hits for choice: {}".format(res['hits']['total'], choice))
        hits = [es_hit["_source"]["text"] for es_hit in res['hits']['hits']]
        return hits

    def filter_hits(self, hits):
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

    def get_question_choices(self, raw_question):
        question_match = re.search("(.*)\([aA]\)(.*)\([bB]\)(.*)\([cC]\)(.*)\([dD]\)(.*)",
                                   raw_question)
        if not question_match:
            # only do this check if no match found for (A)
            question_match = re.search("(.*)\(1\)(.*)\(2\)(.*)\(3\)(.*)\(4\)(.*)", raw_question)
            if not question_match:
                raise ValueError("No choices found in question {}".format(raw_question))
        question_text = question_match.group(1).strip()
        choices = [question_match.group(idx).strip() for idx in [2, 3, 4, 5]]
        return question_text, choices

    def is_clean_sentence(self, s):
        # must only contain expected characters, single-sentence and no space-separated hyphens
        return re.match("^[a-zA-Z0-9][a-zA-Z0-9;:,\(\)%\-\&\.'\"\s]+\.?$", s) is not None and \
               re.match(".*\D\. \D.*", s) is None and \
               re.match(".*\s\-\s.*", s) is None

    def get_key(self, question):
        return re.sub('[^0-9a-zA-Z\.\-^;&%]+', '',
                      re.sub('http[^ ]+', '', question)).strip().rstrip(".")
