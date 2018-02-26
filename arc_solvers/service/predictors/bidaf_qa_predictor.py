import logging
from operator import itemgetter
from typing import List

from allennlp.common.util import JsonDict, sanitize
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.models.model import Model
from allennlp.service.predictors.predictor import Predictor
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Predictor.register("bidaf_qa")
class BidafQaPredictor(Predictor):
    """
    Converts the QA JSON into an instance that is expected by BiDAF model.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._stemmer = PorterStemmer()
        self._stop_words = set(stopwords.words('english'))

    @overrides
    def _json_to_instance(self,  # type: ignore
                          json_dict: JsonDict) -> Instance:
        # pylint: disable=arguments-differ
        """
        Expects JSON that looks like ``{"question": { "stem": "..."}, "para": "..."}``.
        """
        question_text = json_dict["question"]["stem"]
        passage_text = json_dict["para"]
        return self._dataset_reader.text_to_instance(question_text, passage_text)

    @overrides
    def predict_json(self, inputs: JsonDict, cuda_device: int = -1):
        instance = self._json_to_instance(inputs)
        outputs = self._model.forward_on_instance(instance, cuda_device)
        json_output = inputs
        span_str = outputs["best_span_str"]
        # If the file has an answer key, calculate the score
        if "answerKey" in json_output:
            answer_choices = json_output["question"]["choices"]
            # Score each answer choice based on its overlap with the predicted span.
            for choice in answer_choices:
                choice_text = choice["text"]
                choice_score = self._overlap_score(choice_text, span_str)
                choice["score"] = choice_score

            # Get the maximum answer choice score
            max_choice_score = max(answer_choices, key=itemgetter("score"))["score"]
            # Collect all answer choices with the same score
            selected_answers = [choice["label"] for choice in answer_choices
                                if choice["score"] == max_choice_score]
            answer_key = json_output["answerKey"]
            if answer_key in selected_answers:
                question_score = 1 / len(selected_answers)
            else:
                question_score = 0
            json_output["selected_answers"] = ",".join(selected_answers)
            json_output["question_score"] = question_score
        json_output["best_span_str"] = span_str
        return sanitize(json_output)

    def _overlap_score(self, answer: str, predicted_span: str) -> float:
        """
        Scores the predicted span against the correct answer by calculating the proportion of the
        stopword-filtered stemmed words in the correct answer covered by the predicted span
        :param answer: correct answer
        :param predicted_span: predicted span
        :return:
        """
        answer_tokens = self._get_tokens(answer)
        # degenerate case: if the answer only has stopwords, we can not score it.
        if not len(answer_tokens):
            return 0.0
        span_tokens = self._get_tokens(predicted_span)
        overlap = [tok for tok in answer_tokens if tok in span_tokens]
        score = len(overlap) / len(answer_tokens)
        return score

    def _get_tokens(self, phrase: str) -> List[str]:
        # Get the stopword-filtered lowercase stemmed tokens from input phrase
        return [self._stemmer.stem(word) for word in word_tokenize(phrase)
                if word.lower() not in self._stop_words]
