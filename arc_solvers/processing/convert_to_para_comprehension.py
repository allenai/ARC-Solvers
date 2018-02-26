"""
Script to convert the retrieved hits into a paragraph comprehension dataset. Questions with no
hits are mapped to a blank paragraph.
USAGE:
 python scripts/convert_to_para_comprehension.py hits_file qa_file output_file

JSONL format of files
 1. hits_file:
 {
   "id": "Mercury_SC_415702",
   "question": {
      "stem": "George wants to warm his hands quickly by rubbing them. Which skin surface will
               produce the most heat?"
      "choice": {"text": "dry palms", "label": "A"},
      "support": {
         "text": "Use hand sanitizers according to directions, which usually involves rubbing for
                  at least ten seconds, then allowing hands to air dry."
         ...
        }
    },
     "answerKey":"A"
  }

 2. output_file:
   {
   "id": "Mercury_SC_415702",
   "question": {
      "stem": "George wants to warm his hands quickly by rubbing them. Which skin surface will
               produce the most heat?"
       "choices":[
                  {"text":"dry palms","label":"A"},
                  {"text":"wet palms","label":"B"},
                  {"text":"palms covered with oil","label":"C"},
                  {"text":"palms covered with lotion","label":"D"}
                 ]
    },
    "para": "Use hand sanitizers according to directions, which usually involves rubbing for
             at least ten seconds, then allowing hands to air dry. ..."
    },
    "answerKey":"A"
  }
"""

import json
import sys


def convert_to_para_comprehension(hits_file: str, qa_file: str, output_file: str):
    qid_choices = dict()
    qid_stem = dict()
    qid_answer = dict()
    qid_sentences = dict()
    with open(qa_file, 'r') as qa_handle:
        for line in qa_handle:
            json_line = json.loads(line)
            qid = json_line["id"]
            choices = json_line["question"]["choices"]
            qid_choices[qid] = choices
            qid_sentences[qid] = []
            qid_stem[qid] = json_line["question"]["stem"]
            qid_answer[qid] = json_line["answerKey"]

    with open(hits_file, 'r') as hits_handle:
        print("Writing to {} from {}".format(output_file, hits_file))
        for line in hits_handle:
            json_line = json.loads(line)
            qid = json_line["id"]
            sentence = json_line["question"]["support"]["text"]
            if not sentence.endswith("."):
                sentence = sentence + "."
            qid_sentences[qid].append(sentence)

    with open(output_file, 'w') as output_handle:
        for qid, sentences in qid_sentences.items():
            if len(sentences):
                output_dict = {
                    "id": qid,
                    "question": {
                        "stem": qid_stem[qid],
                        "choices": qid_choices[qid]
                    },
                    "para": " ".join(sentences),
                    "answerKey": qid_answer[qid]
                }
                output_handle.write(json.dumps(output_dict))
                output_handle.write("\n")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise ValueError("Provide at least three arguments: "
                         "json file with hits, qa file, output file name")
    convert_to_para_comprehension(sys.argv[1], sys.argv[2], sys.argv[3])
