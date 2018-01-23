import logging

from allennlp.common.util import JsonDict, sanitize
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.models.model import Model
from allennlp.service.predictors.predictor import Predictor
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Predictor.register("dgem")
class DgemPredictor(Predictor):
    """
    Converts the QA JSON into an instance that is expected by the Decomposable Attention Model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        labels = self._model.vocab.get_index_to_token_vocabulary("labels").values()
        #
        if "entails" in labels:
            self._entailment_idx = self._model.vocab.get_token_index("entails", "labels")
        elif "entailment" in labels:
            self._entailment_idx = self._model.vocab.get_token_index("entailment", "labels")
        else:
            raise Exception("No label for entailment found in the label space: {}".format(
                ",".join(labels)))

    @overrides
    def _json_to_instance(self,  # type: ignore
                          json_dict: JsonDict) -> Instance:
        # pylint: disable=arguments-differ
        premise_text = json_dict["premise"]
        hypothesis_text = json_dict["hypothesis"]
        hypothesis_structure = json_dict["hypothesisStructure"]
        return self._dataset_reader.text_to_instance(premise_text, hypothesis_text,
                                                     hypothesis_structure)

    @overrides
    def predict_json(self, inputs: JsonDict, cuda_device: int = -1):
        instance = self._json_to_instance(inputs)
        outputs = self._model.forward_on_instance(instance, cuda_device)
        json_output = inputs
        json_output["score"] = outputs["label_probs"][self._entailment_idx]
        return sanitize(json_output)