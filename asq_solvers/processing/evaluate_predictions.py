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

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))))


def evaluate_predictions(predictions_file, qa_file, output_file):
    print("Writing qa predictions to {} from entailment predictions at {}".format(
        output_file, predictions_file))
    id_choice_score_dict = get_scores_per_id_and_choice(predictions_file)
    score_predictions(id_choice_score_dict, qa_file, output_file)


def get_scores_per_id_and_choice(predictions_file):
    with ExitStack() as stack:
        predictions_handle = stack.enter_context(open(predictions_file, 'r'))
        id_choice_score_dict = dict()
        for line in predictions_handle:
            json_line = json.loads(line)
            id = json_line["id"]
            if "score" not in json_line:
                print("Missing score in line:" + line)
            score = json_line["score"]
            choice_text = json_line["question"]["choice"]["text"]
            if id in id_choice_score_dict:
                predictions = id_choice_score_dict[id]
                if choice_text in predictions:
                    if predictions[choice_text] > score:
                        predictions[choice_text] = score
                else:
                    predictions[choice_text] = score
            else:
                id_choice_score_dict[id] = dict()
                id_choice_score_dict[id][choice_text] = score
        return id_choice_score_dict


def score_predictions(id_choice_score_dict, qa_file, output_file):
    with ExitStack() as stack:
        total_score = 0
        num_questions = 0
        qa_handle = stack.enter_context(open(qa_file, 'r'))
        output_handle = stack.enter_context(open(output_file, 'w'))
        for line in qa_handle:
            json_line = json.loads(line)
            id = json_line["id"]
            best_score = 0
            selected_answers = []
            for choice in json_line["question"]["choices"]:
                choice_text = choice["text"]
                if id in id_choice_score_dict and choice_text in id_choice_score_dict[id]:
                    choice["score"] = id_choice_score_dict[id][choice_text]
                else:
                    choice["score"] = 0
                if choice["score"] > best_score:
                    selected_answers = [choice["label"]]
                    best_score = choice["score"]
                elif choice["score"] == best_score:
                    selected_answers.append(choice["label"])
            answer_key = json_line["answerKey"]
            question_score = 0
            if answer_key in selected_answers:
                question_score = 1 / len(selected_answers)
            total_score += question_score
            json_line["selected_answers"] = ",".join(selected_answers)
            json_line["question_score"] = question_score
            num_questions += 1
            output_handle.write(json.dumps(json_line) + "\n")
        print("Metrics:\n\tScore={}\n\tQuestions:{}\n\tExam Score:{:.3f}".format(
            total_score, num_questions, (total_score / num_questions)))


def create_output_dict(input_json, choice_json, hit_sentence):
    output_dict = {
        "id": input_json["id"],
        "question": {
            "stem": input_json["question"]["stem"],
            "choice": choice_json,
            "support": hit_sentence
        },
        "answerKey": input_json["answerKey"]
    }
    return output_dict


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise ValueError("Provide at least three arguments: "
                         "predictions_file, original qa file, output file")
    evaluate_predictions(sys.argv[1], sys.argv[2], sys.argv[3])
