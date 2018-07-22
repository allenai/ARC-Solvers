import torch
from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import replace_masked_values
from allennlp.nn.util import get_text_field_mask
import allennlp
from typing import Union, Dict

import torch
from allennlp.modules import MatrixAttention, Seq2SeqEncoder

def masked_mean(tensor, dim, mask):
    """
    ``Performs a mean on just the non-masked portions of the ``tensor`` in the
    ``dim`` dimension of the tensor.

    =====================================================================
    From Decomposable Graph Entailment Model code replicated from SciTail repo
    https://github.com/allenai/scitail
    =====================================================================
    """
    if mask is None:
        return torch.mean(tensor, dim)
    if tensor.dim() != mask.dim():
        raise ConfigurationError("tensor.dim() (%d) != mask.dim() (%d)" % (tensor.dim(), mask.dim()))
    masked_tensor = replace_masked_values(tensor, mask, 0.0)
    # total value
    total_tensor = torch.sum(masked_tensor, dim)
    # count
    count_tensor = torch.sum((mask != 0), dim)
    # set zero count to 1 to avoid nans
    zero_count_mask = (count_tensor == 0)
    count_plus_zeros = (count_tensor + zero_count_mask).float()
    # average
    mean_tensor = total_tensor / count_plus_zeros
    return mean_tensor



def seq2vec_seq_aggregate(seq_tensor, mask, aggregate, bidirectional, dim=1):
    """
        Takes the aggregation of sequence tensor

        :param seq_tensor: Batched sequence requires [batch, seq, hs]
        :param mask: binary mask with shape batch, seq_len, 1
        :param aggregate: max, avg, sum
        :param dim: The dimension to take the max. for batch, seq, hs it is 1
        :return:
    """

    seq_tensor_masked = seq_tensor * mask.unsqueeze(-1)
    aggr_func = None
    if aggregate == "last":
        raise NotImplemented("This is currently not supported with AllenNLP 0.2.")
        seq = allennlp.nn.util.get_final_encoder_states(seq_tensor, mask, bidirectional)
    elif aggregate == "max":
        aggr_func = torch.max
        seq, _ = aggr_func(seq_tensor_masked, dim=dim)
    elif aggregate == "min":
        aggr_func = torch.min
        seq, _ = aggr_func(seq_tensor_masked, dim=dim)
    elif aggregate == "sum":
        aggr_func = torch.sum
        seq = aggr_func(seq_tensor_masked, dim=dim)
    elif aggregate == "avg":
        aggr_func = torch.sum
        seq = aggr_func(seq_tensor_masked, dim=dim)
        seq_lens = torch.sum(mask, dim=dim)  # this returns batch_size, 1
        seq = seq / seq_lens.view([-1, 1])

    return seq


def embed_encode_and_aggregate_text_field(question: Dict[str, torch.LongTensor],
                                          text_field_embedder,
                                          embeddings_dropout,
                                          encoder,
                                          aggregation_type):
    """
    Given a batched token ids (2D) runs embeddings lookup with dropout, context encoding and aggregation
    :param question:
    :param text_field_embedder: The embedder to be used for embedding lookup
    :param embeddings_dropout: Dropout
    :param encoder: Context encoder
    :param aggregation_type: The type of aggregation - max, sum, avg, last
    :return:
    """
    embedded_question = text_field_embedder(question)
    question_mask = get_text_field_mask(question).float()
    embedded_question = embeddings_dropout(embedded_question)

    encoded_question = encoder(embedded_question, question_mask)

    # aggregate sequences to a single item
    encoded_question_aggregated = seq2vec_seq_aggregate(encoded_question, question_mask, aggregation_type,
                                                        None, 1)  # bs X d

    return encoded_question_aggregated


def embed_encode_and_aggregate_list_text_field(texts_list: Dict[str, torch.LongTensor],
                                               text_field_embedder,
                                               embeddings_dropout,
                                               encoder: Seq2SeqEncoder,
                                               aggregation_type,
                                               init_hidden_states=None):
    """
    Given a batched list of token ids (3D) runs embeddings lookup with dropout, context encoding and aggregation on
    :param question:
    :param text_field_embedder: The embedder to be used for embedding lookup
    :param embeddings_dropout: Dropout
    :param encoder: Context encoder
    :param aggregation_type: The type of aggregation - max, sum, avg, last
    :param get_last_states: If it should return the last states.
    :param texts_list:
    :param text_field_embedder:
    :param embeddings_dropout:
    :param encoder:
    :param aggregation_type:
    :param init_hidden_states:
    :return:
    """
    embedded_texts = text_field_embedder(texts_list)
    embedded_texts = embeddings_dropout(embedded_texts)

    batch_size, choices_cnt, choice_tokens_cnt, d = tuple(embedded_texts.shape)

    embedded_texts_flattened = embedded_texts.view([batch_size * choices_cnt, choice_tokens_cnt, -1])
    # masks

    texts_mask_dim_3 = get_text_field_mask(texts_list).float()
    texts_mask_flatened = texts_mask_dim_3.view([-1, choice_tokens_cnt])

    # context encoding
    multiple_texts_init_states = None
    if init_hidden_states is not None:
        if init_hidden_states.shape[0] == batch_size and init_hidden_states.shape[1] != choices_cnt:
            if init_hidden_states.shape[1] != encoder.get_output_dim():
                raise ValueError("The shape of init_hidden_states is {0} but is expected to be {1} or {2}".format(str(init_hidden_states.shape),
                                                                            str([batch_size, encoder.get_output_dim()]),
                                                                            str([batch_size, choices_cnt, encoder.get_output_dim()])))
            # in this case we passed only 2D tensor which is the default output from question encoder
            multiple_texts_init_states = init_hidden_states.unsqueeze(1).expand([batch_size, choices_cnt, encoder.get_output_dim()]).contiguous()

            # reshape this to match the flattedned tokens
            multiple_texts_init_states = multiple_texts_init_states.view([batch_size * choices_cnt, encoder.get_output_dim()])
        else:
            multiple_texts_init_states = init_hidden_states.view([batch_size * choices_cnt, encoder.get_output_dim()])

    encoded_texts_flattened = encoder(embedded_texts_flattened, texts_mask_flatened, hidden_state=multiple_texts_init_states)

    aggregated_choice_flattened = seq2vec_seq_aggregate(encoded_texts_flattened, texts_mask_flatened,
                                                        aggregation_type,
                                                        encoder,
                                                        1)  # bs*ch X d

    aggregated_choice_flattened_reshaped = aggregated_choice_flattened.view([batch_size, choices_cnt, -1])
    return aggregated_choice_flattened_reshaped

