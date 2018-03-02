"""
Script to compute the QA score from the scores per choice
USAGE:
 python scripts/calculate_scores.py predictions_file

Minimal expected format of predictions_file:
   {
    "question": {
       "stem":"George wants to warm his hands quickly by rubbing them. Which skin surface will
               produce the most heat?",
       "choices":[
                  {"text":"dry palms","label":"A", "score": 0.6},
                  {"text":"wet palms","label":"B", "score": 0.4},
                  {"text":"palms covered with oil","label":"C", "score": 0.2},
                  {"text":"palms covered with lotion","label":"D", "score": 0.3}
                 ]
    },
    "answerKey":"A"
  }
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))))

from operator import itemgetter


def calculate_scores(qa_predictions: str) -> None:
    """
    Uses the scores per answer choice to compute the QA score
    :param qa_predictions: QA predictions with scores per choice
    """
    with open(qa_predictions, 'r') as qa_handle:
        total_score = 0
        num_questions = 0
        partially_correct = 0
        correct = 0
        incorrect = 0
        for line in qa_handle:
            json_line = json.loads(line)
            answer_choices = json_line["question"]["choices"]
            max_choice_score = max(answer_choices, key=itemgetter("score"))["score"]
            # Collect all answer choices with the same score
            selected_answers = [choice["label"] for choice in answer_choices
                                if choice["score"] == max_choice_score]
            answer_key = json_line["answerKey"]

            if answer_key in selected_answers:
                question_score = 1 / len(selected_answers)
                if question_score < 1:
                    partially_correct += 1
                else:
                    correct += 1
            else:
                question_score = 0
                incorrect += 1
            total_score += question_score
            num_questions += 1

        print("""Metrics:
       Total Points={:.2f}
       Questions:{}
       Exam Score:{:.2f}
          Correct:      {}
          Incorrect:    {}
          Partial:      {}
                """.format(total_score, num_questions, (total_score / num_questions)*100,
                           correct, incorrect, partially_correct))


if __name__ == "__main__":
    if len(sys.argv) < 1:
        raise ValueError("Provide at least one argument: "
                         "predictions_file")
    calculate_scores(sys.argv[1])
