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
import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))))

from operator import itemgetter
from typing import List, Dict

from allennlp.common.util import JsonDict


def evaluate_predictions(predictions_file, qa_file, output_file):
    print("Writing qa predictions to {} from entailment predictions at {}".format(
        output_file, predictions_file))
    qid_choice_scores = get_scores_per_qid_and_choice(predictions_file)
    score_predictions(qid_choice_scores, qa_file, output_file)


def get_scores_per_qid_and_choice(predictions_file) -> Dict[str, Dict[str, List[JsonDict]]]:
    """
    Reads the file with entailment predictions to produce predictions per answer choice per qid.
    :return: dictionary from qid -> (dictionary from choice text -> list of entailment predictions)
    """
    with open(predictions_file, 'r') as predictions_handle:
        qid_choice_predictions = dict()
        for line in predictions_handle:
            json_line = json.loads(line)
            qid = json_line["id"]
            if "score" not in json_line:
                raise Exception("Missing score in line:" + line)
            choice_score = json_line["score"]
            choice_support = json_line["question"]["support"]
            choice_text = json_line["question"]["choice"]["text"]
            choice_prediction = {
                "score": choice_score,
                "support": choice_support,
            }
            if qid in qid_choice_predictions:
                choice_scores = qid_choice_predictions[qid]
                if choice_text not in choice_scores:
                    choice_scores[choice_text] = []
                choice_scores[choice_text].append(choice_prediction)
            else:
                qid_choice_predictions[qid] = dict()
                qid_choice_predictions[qid][choice_text] = [choice_prediction]
        return qid_choice_predictions


def score_predictions(qid_choice_predictions: Dict[str, Dict[str, List[JsonDict]]],
                      qa_file: str, output_file: str) -> None:
    """
    Uses the entailment predictions per answer choice per qid to compute the QA score
    :param qid_choice_predictions: qid -> (choice text -> predictions)
    :param qa_file: Original QA JSONL file
    :param output_file: Output file with selected choices ("selected_answers" key) and score (
    "question_score" key) per multiple-choice question.
    """
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
                if id in qid_choice_predictions and choice_text in qid_choice_predictions[id]:
                    update_choice_with_scores(qid_choice_predictions[id][choice_text], choice)
                else:
                    update_choice_with_scores([], choice)
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
        print("Metrics:\n\tScore={}\n\tQuestions:{}\n\tExam Score:{:.5f}".format(
            total_score, num_questions, (total_score / num_questions)))


def update_choice_with_scores(choice_predictions: List[JsonDict],
                              input_choice: JsonDict) -> None:
    """
    Uses the entailment predictions to compute the solvers score for the answer choice. This
    function will update input answer choice json with two new keys "score" and "support"
    corresponding to the solver score and best supporting sentence for this choice respectively.
    :param choice_predictions: list of predictions for this choice
    :param input_choice: input json for this answer choice that will be updated in-place
    """
    if len(choice_predictions):
        sorted_predictions = sorted(choice_predictions,
                                    key=itemgetter("score"), reverse=True)
        score = score_choice_predictions([pred["score"] for pred in sorted_predictions])
        support = sorted_predictions[0]["support"]
        input_choice["score"] = score
        input_choice["support"] = support
    else:
        input_choice["score"] = 0
        input_choice["support"] = ""


# Returns the score for an answer choice given the scores per supporting sentence
def score_choice_predictions(choice_predictions: List[float]) -> float:
    # Round to four decimal points
    return round(max(choice_predictions), 4)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise ValueError("Provide at least three arguments: "
                         "predictions_file, original qa file, output file")
    evaluate_predictions(sys.argv[1], sys.argv[2], sys.argv[3])
