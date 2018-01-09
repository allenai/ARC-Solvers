import logging

from allennlp.common.util import JsonDict
from allennlp.data.instance import Instance
from allennlp.service.predictors.predictor import Predictor
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Predictor.register("dgem")
class DgemPredictor(Predictor):
    """
    Converts the QA JSON into an instance that is expected by the Decomposable Attention Model.
    """
    @overrides
    def _json_to_instance(self,  # type: ignore
                          json_dict: JsonDict) -> Instance:
        # pylint: disable=arguments-differ
        premise_text = json_dict["premise"]
        hypothesis_text = json_dict["hypothesis"]
        hypothesis_structure = json_dict["hypothesisStructure"]
        return self._dataset_reader.text_to_instance(premise_text, hypothesis_text,
                                                     hypothesis_structure)
