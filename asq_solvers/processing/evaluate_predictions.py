"""
Script to compute the QA score from the entailment predictions for each supporting sentence and
answer choice.
USAGE:
 python scripts/evaluate_predictions.py predictions_file qa_file output_file

Minimal expected format of files.
 1. predictions_file:
  {"id": "Mercury_SC_415702",
   "question": {
      "choice": {"text": "dry palms", "label": "A"},
    }
   "score": 0.31790056824684143
  }

 2. qa_file:
   {
    "id":"Mercury_SC_415702",
    "question": {
       "stem":"George wants to warm his hands quickly by rubbing them. Which skin surface will
               produce the most heat?",
       "choices":[
                  {"text":"dry palms","label":"A"},
                  {"text":"wet palms","label":"B"},
                  {"text":"palms covered with oil","label":"C"},
                  {"text":"palms covered with lotion","label":"D"}
                 ]
    },
    "answerKey":"A"
  }
"""

import json
import os
import sys
from contextlib import ExitStack
from typing import List, Dict

import numpy
from cytoolz.itertoolz import itemgetter

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))))


def evaluate_predictions(predictions_file, qa_file, output_file):
    print("Writing qa predictions to {} from entailment predictions at {}".format(
        output_file, predictions_file))
    qid_choice_scores = get_scores_per_qid_and_choice(predictions_file)
    score_predictions(qid_choice_scores, qa_file, output_file)


def get_scores_per_qid_and_choice(predictions_file):
    with open(predictions_file, 'r') as predictions_handle:
        qid_choice_scores = dict()
        for line in predictions_handle:
            json_line = json.loads(line)
            qid = json_line["id"]
            if "score" not in json_line:
                raise Exception("Missing score in line:" + line)
            choice_score = json_line["score"]
            choice_text = json_line["question"]["choice"]["text"]
            if qid in qid_choice_scores:
                choice_scores = qid_choice_scores[qid]
                if choice_text not in choice_scores:
                    choice_scores[choice_text] = []
                choice_scores[choice_text].append(choice_score)
            else:
                qid_choice_scores[qid] = dict()
                qid_choice_scores[qid][choice_text] = [choice_score]
        return qid_choice_scores


def score_predictions(qid_choice_scores: Dict[str, Dict[str, List[float]]],
                      qa_file: str, output_file: str) -> None:
    with open(qa_file, 'r') as qa_handle, open(output_file, 'w') as output_handle:
        total_score = 0
        num_questions = 0
        for line in qa_handle:
            json_line = json.loads(line)
            id = json_line["id"]
            answer_choices = json_line["question"]["choices"]
            for choice in answer_choices:
                choice_text = choice["text"]
                # if we have any entailment prediction for this answer choice, pick the
                if id in qid_choice_scores and choice_text in qid_choice_scores[id]:
                    choice["score"] = score_choice(qid_choice_scores[id][choice_text])
                else:
                    choice["score"] = 0
            # Get the maximum answer choice score
            max_choice_score = max(answer_choices, key=itemgetter("score"))["score"]
            # Collect all answer choices with the same score
            selected_answers = [choice["label"] for choice in answer_choices
                                if choice["score"] == max_choice_score]
            answer_key = json_line["answerKey"]

            if answer_key in selected_answers:
                question_score = 1 / len(selected_answers)
            else:
                question_score = 0
            total_score += question_score
            json_line["selected_answers"] = ",".join(selected_answers)
            json_line["question_score"] = question_score
            num_questions += 1
            output_handle.write(json.dumps(json_line) + "\n")
        print("Metrics:\n\tScore={}\n\tQuestions:{}\n\tExam Score:{:.3f}".format(
            total_score, num_questions, (total_score / num_questions)))


# Returns the score for an answer choice given the scores per supporting sentence
def score_choice(choice_predictions: List[float]) -> float:
    return round(numpy.max(choice_predictions), 4)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise ValueError("Provide at least three arguments: "
                         "predictions_file, original qa file, output file")
    evaluate_predictions(sys.argv[1], sys.argv[2], sys.argv[3])
