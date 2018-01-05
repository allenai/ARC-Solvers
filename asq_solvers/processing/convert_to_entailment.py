import json
import re
import sys
from contextlib import ExitStack


def convert_to_entailment(qa_file, output_file):
    with ExitStack() as stack:
        print("Writing to {} from {}".format(output_file, qa_file))
        output_handle = stack.enter_context(open(output_file, 'w'))
        qa_handle = stack.enter_context(open(qa_file, 'r'))
        for line in qa_handle:
            json_line = json.loads(line)
            output_dict = convert_qajson_to_entailment(json_line)
            output_handle.write(json.dumps(output_dict))
            output_handle.write("\n")


def convert_qajson_to_entailment(qa_json):
    question_text = qa_json["question"]["stem"]
    choice = qa_json["question"]["choice"]["text"]
    support = qa_json["question"]["support"]
    hypothesis = create_hypothesis(get_fitb_from_question(question_text), choice)
    output_dict = create_output_dict(qa_json, support, hypothesis)
    return output_dict


def get_fitb_from_question(question_text: str) -> str:
    fitb = replace_wh_word_with_blank(question_text)
    if not re.match(".*_+.*", fitb):
        print("Can't create hypothesis from: '{}'. Appending ___ !".format(question_text))
        fitb = re.sub("[\.\? ]*$", "", question_text.strip()) + " ___"
    return fitb


def create_hypothesis(fitb: str, choice: str) -> str:
    if ". ___" in fitb or fitb.startswith("___"):
        choice = choice[0].upper() + choice[1:]
    else:
        choice = choice.lower()

    if not fitb.endswith("___"):
        choice = choice.rstrip(".")
    hypothesis = re.sub("___+", choice, fitb)
    return hypothesis.replace("..", ".")


def replace_wh_word_with_blank(question_str: str):
    match_arr = []
    wh_words = ["which", "what", "where", "when", "how", "who", "why"]
    for wh in wh_words:
        m = re.search(wh + "\?[^\.]*[\. ]*$", question_str.lower())
        if m:
            match_arr = [(wh, m.start())]
            break
        else:
            m = re.search(wh + "[ ,][^\.]*[\. ]*$", question_str.lower())
            if m:
                match_arr.append((wh, m.start()))

    if len(match_arr) > 0:
        match_arr.sort(key=lambda x: x[1])
        replaceWord = match_arr[0][0]
        replaceIdx = match_arr[0][1]
        question_str = re.sub("\?$", ".", question_str.strip())
        fitb_question = question_str[:replaceIdx] + "___" + question_str[replaceIdx + len(replaceWord):]
        return fitb_question.replace("___ of the following", "___")
    elif re.match(".*[^\.\?] *$", question_str):
        return question_str + " ___"
    else:
        return re.sub(" this[ \?]", " ___ ", question_str)


def create_output_dict(input_json, premise, hypothesis):
    input_json["premise"] = premise
    input_json["hypothesis"] = hypothesis
    return input_json


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise ValueError("Provide at least two arguments: "
                         "json file with hits, output file name")
    convert_to_entailment(sys.argv[1], sys.argv[2])

