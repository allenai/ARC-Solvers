import json
import os
import sys
from contextlib import ExitStack
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))))
from asq_solvers.processing.es_search import EsSearch

es_search = EsSearch()


def add_retrieved_text(qa_file, output_file):
    with ExitStack() as stack:
        print("Writing to {} from {}".format(output_file, qa_file))
        output_handle = stack.enter_context(open(output_file, 'w'))
        qa_handle = stack.enter_context(open(qa_file, 'r'))
        for line in qa_handle:
            json_line = json.loads(line)
            for output_dict in add_hits_to_qajson(json_line):
                output_handle.write(json.dumps(output_dict) + "\n")


def add_hits_to_qajson(qa_json):
    question_text = qa_json["question"]["stem"]
    choices = [choice["text"] for choice in qa_json["question"]["choices"]]
    hits_per_choice = es_search.get_hits_for_question(question_text, choices)
    output_dicts_per_question = []
    for choice in qa_json["question"]["choices"]:
        choice_text = choice["text"]
        hits = hits_per_choice[choice_text]
        for hit in hits:
            output_dict_per_hit = create_output_dict(qa_json, choice, hit)
            output_dicts_per_question.append(output_dict_per_hit)
    return output_dicts_per_question


def create_output_dict(qa_json, choice_json, hit_sentence):
    output_dict = {
        "id": qa_json["id"],
        "question": {
            "stem": qa_json["question"]["stem"],
            "choice": choice_json,
            "support": hit_sentence
        },
        "answerKey": qa_json["answerKey"]
    }
    return output_dict


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise ValueError("Provide at least two arguments: "
                         "question-answer json file, output file name")
    add_retrieved_text(sys.argv[1], sys.argv[2])

