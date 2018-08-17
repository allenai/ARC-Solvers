from typing import Dict, List, Any
import json
import logging

from allennlp.data import Dataset
from overrides import overrides

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, ListField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("arc-multi-choice-json")
class ArcMultiChoiceJsonReader(DatasetReader):
    """
    Reading instances in ARC jsonl format.  This data is
    formatted as jsonl, one json-formatted instance per line.  An example of the json in the data is:

        {"id":"MCAS_2000_4_6",
        "question":{"stem":"Which technology was developed most recently?",
            "choices":[
                {"text":"cellular telephone","label":"A"},
                {"text":"television","label":"B"},
                {"text":"refrigerator","label":"C"},
                {"text":"airplane","label":"D"}
            ]},
        "answerKey":"A"
        }

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__()
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        instances = []
        with open(file_path, 'r') as data_file:
            logger.info("Reading instances in ARC jsonl format from dataset at: %s", file_path)
            for line in data_file:
                item_json = json.loads(line.strip())

                item_id = item_json["id"]
                question_text = item_json["question"]["stem"]

                choice_label_to_id = {}
                choice_text_list = []

                for choice_id, choice_item in enumerate(item_json["question"]["choices"]):
                    choice_label = choice_item["label"]
                    choice_label_to_id[choice_label] = choice_id

                    choice_text = choice_item["text"]

                    choice_text_list.append(choice_text)

                answer_id = choice_label_to_id[item_json["answerKey"]]

                instances.append(self.text_to_instance(item_id, question_text, choice_text_list, answer_id))

        return Dataset(instances)


    @overrides
    def text_to_instance(self,  # type: ignore
                         item_id: Any,
                         question_text: str,
                         choice_text_list: List[str],
                         answer_id: int) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        question_tokens = self._tokenizer.tokenize(question_text)
        choices_tokens_list = [self._tokenizer.tokenize(x) for x in choice_text_list]
        fields['question'] = TextField(question_tokens, self._token_indexers)
        fields['choices_list'] = ListField([TextField(x, self._token_indexers) for x in choices_tokens_list])
        fields['label'] = LabelField(answer_id, skip_indexing=True)

        metadata = {
           "id": item_id,
           "question_text": question_text,
           "choice_text_list": choice_text_list,
           "question_tokens": [x.text for x in question_tokens],
           "choice_tokens_list": [[x.text for x in ct] for ct in choices_tokens_list],
        }

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'ArcMultiChoiceJsonReader':
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))

        return ArcMultiChoiceJsonReader(tokenizer=tokenizer,
                          token_indexers=token_indexers)