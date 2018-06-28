from allennlp.modules.matrix_attention import LegacyMatrixAttention
from typing import Dict, Optional, AnyStr, List, Any

import torch
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, MatrixAttention, SimilarityFunction
from allennlp.modules import TextFieldEmbedder
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics import CategoricalAccuracy

from arc_solvers.common.common_utils import update_params
from arc_solvers.nn.util import embedd_encode_and_aggregate_list_text_field, embedd_encode_and_aggregate_text_field


@Model.register("qa_multi_choice_max_att")
class QAMultiChoiceMaxAttention(Model):
    """
    This ``Model`` implements a simple context encoder baseline that models the interaction between question and choice:
    The basic outline of this model is to get an embedded representation for the
    question and choice, model an interaction between them and use a linear projection to the class dimension + softmax to get a final predictions:

    question_encoded = context_enc(question_words)  # context encoder can be any AllenNLP supported or None
    choice_encoded = context_enc(choice_words)

    question_aggregate = aggregate_method(question_encoded)
    choice_aggregate = aggregate_method(choice_encoded)

    inter = concat([question_aggregate, choice_aggregate, abs(choice_aggregate - question_aggregate), question_aggregate * choice_aggregate)

    The model is a popular baseline but it is most simialar to Conneau, A. et al. (2017) ‘Supervised Learning of Universal Sentence Representations from Natural Language Inference Data’,
    without the feed-forward layer!

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``question`` and ``choice`` ``TextFields`` we get as input to the
        model.
    aggregate_feedforward : ``FeedForward``
        These feedforward networks are applied to the concatenated result of the
        encoder networks, and its output is used as the entailment class logits.
    question_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        After embedding the question, we can optionally apply an encoder.  If this is ``None``, we
        will do nothing.
    choice_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        After embedding the choice, we can optionally apply an encoder.  If this is ``None``,
        we will use the ``question_encoder`` for the encoding (doing nothing if ``question_encoder``
        is also ``None``).
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    share_encoders : ``bool``, optional (default=``false``)
        Shares the weights of the question and choice encoders.
    aggregate_question : ``str``, optional (default=``max``, allowed options [max, avg, sum, last])
        The aggregation method for the encoded question.
    aggregate_choice : ``str``, optional (default=``max``, allowed options [max, avg, sum, last])
        The aggregation method for the encoded choice.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 question_encoder: Optional[Seq2SeqEncoder] = None,
                 choice_encoder: Optional[Seq2SeqEncoder] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 aggregate_question: Optional[str] = "max",
                 aggregate_choice: Optional[str] = "max",
                 embeddings_dropout_value: Optional[float] = 0.0,
                 params=Params) -> None:
        super(QAMultiChoiceMaxAttention, self).__init__(vocab)

        self._use_cuda = (torch.cuda.is_available() and torch.cuda.current_device() >= 0)

        self._params = params

        self._text_field_embedder = text_field_embedder
        if embeddings_dropout_value > 0.0:
            self._embeddings_dropout = torch.nn.Dropout(p=embeddings_dropout_value)
        else:
            self._embeddings_dropout = lambda x: x

        self._question_encoder = question_encoder

        # choices encoding
        self._choice_encoder = choice_encoder

        self._question_aggregate = aggregate_question
        self._choice_aggregate = aggregate_choice

        self._num_labels = vocab.get_vocab_size(namespace="labels")

        question_output_dim = self._text_field_embedder.get_output_dim()
        if self._question_encoder is not None:
            question_output_dim = self._question_encoder.get_output_dim()

        choice_output_dim = self._text_field_embedder.get_output_dim()
        if self._choice_encoder is not None:
            choice_output_dim = self._choice_encoder.get_output_dim()

        if question_output_dim != choice_output_dim:
            raise ConfigurationError("Output dimension of the question_encoder (dim: {}), "
                                     "plus choice_encoder (dim: {})"
                                     "must match! "
                                     .format(question_output_dim,
                                             choice_output_dim))

        # question to choice attention
        att_question_to_choice_params = params.get("att_question_to_choice")
        if "tensor_1_dim" in att_question_to_choice_params:
            # automatically infer the input size
            att_question_to_choice_params = update_params(att_question_to_choice_params,
                                                          {"tensor_1_dim": question_output_dim,
                                                           "tensor_2_dim": choice_output_dim})
        self._matrix_attention_question_to_choice = LegacyMatrixAttention(
            SimilarityFunction.from_params(att_question_to_choice_params))

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                choices_list: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None,
                ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        question : Dict[str, torch.LongTensor]
            From a ``TextField``
        choices_list : Dict[str, torch.LongTensor]
            From a ``List[TextField]``
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``

        Returns
        -------
        An output dictionary consisting of:

        label_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log
            probabilities of the entailment label.
        label_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing probabilities of the
            entailment label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """

        encoded_choices_aggregated = embedd_encode_and_aggregate_list_text_field(choices_list,
                                                                                 self._text_field_embedder,
                                                                                 self._embeddings_dropout,
                                                                                 self._choice_encoder,
                                                                                 self._choice_aggregate)  # # bs, choices, hs

        encoded_question_aggregated, _ = embedd_encode_and_aggregate_text_field(question, self._text_field_embedder,
                                                                                self._embeddings_dropout,
                                                                                self._question_encoder,
                                                                                self._question_aggregate,
                                                                                get_last_states=False)  # bs, hs

        q_to_choices_att = self._matrix_attention_question_to_choice(encoded_question_aggregated.unsqueeze(1),
                                                                     encoded_choices_aggregated).squeeze()

        label_logits = q_to_choices_att
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        output_dict = {"label_logits": label_logits, "label_probs": label_probs}

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._accuracy(label_logits, label.squeeze(-1))
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self._accuracy.get_metric(reset),
        }

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'QAMultiChoiceMaxAttention':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)

        embeddings_dropout_value = params.pop("embeddings_dropout", 0.0)

        # question encoder
        question_encoder_params = params.pop("question_encoder", None)
        question_enc_aggregate = params.pop("question_encoder_aggregate", "max")
        share_encoders = params.pop("share_encoders", False)

        if question_encoder_params is not None:
            question_encoder = Seq2SeqEncoder.from_params(question_encoder_params)
        else:
            question_encoder = None

        if share_encoders:
            choice_encoder = question_encoder
            choice_enc_aggregate = question_enc_aggregate
        else:
            # choice encoder
            choice_encoder_params = params.pop("choice_encoder", None)
            choice_enc_aggregate = params.pop("choice_encoder_aggregate", "max")

            if choice_encoder_params is not None:
                choice_encoder = Seq2SeqEncoder.from_params(choice_encoder_params)
            else:
                choice_encoder = None

        init_params = params.pop('initializer', None)
        initializer = (InitializerApplicator.from_params(init_params)
                       if init_params is not None
                       else InitializerApplicator())

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   question_encoder=question_encoder,
                   choice_encoder=choice_encoder,
                   initializer=initializer,
                   aggregate_choice=choice_enc_aggregate,
                   aggregate_question=question_enc_aggregate,
                   embeddings_dropout_value=embeddings_dropout_value,
                   params=params)