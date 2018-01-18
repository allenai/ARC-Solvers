"""
=====================================================================
Decomposable Graph Entailment Model code replicated from SciTail repo
https://github.com/allenai/scitail
=====================================================================
"""

from typing import Dict, List, Any, Tuple

import numpy
import torch
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.matrix_attention import MatrixAttention
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.similarity_functions.similarity_function import SimilarityFunction
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask, last_dim_softmax, weighted_sum, \
    replace_masked_values
from allennlp.training.metrics import CategoricalAccuracy
from numpy.core.arrayprint import array2string, set_printoptions
from torch import FloatTensor

from asq_solvers.modules.single_time_distributed import SingleTimeDistributed
from asq_solvers.nn.util import masked_mean


@Model.register("tree_attention")
class TreeAttention(Model):
    """
    This ``Model`` implements the decomposable graph entailment model using graph structure from
    the hypothesis and aligning premise words onto this structure.

    The basic outline of this model is to get attention over the premise for each node in the
    graph and use these attentions to compute the probability of each node being true and each
    edge being true.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``premise`` and ``hypothesis`` and nodes
    premise_encoder : ``Seq2SeqEncoder``
        After embedding the premise, we apply an encoder to get the context-based representation

    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 phrase_probability: FeedForward,
                 edge_probability: FeedForward,
                 premise_encoder: Seq2SeqEncoder,
                 edge_embedding: Embedding,
                 use_encoding_for_node: bool,
                 ignore_edges: bool,
                 attention_similarity: SimilarityFunction,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(TreeAttention, self).__init__(vocab)

        self._text_field_embedder = text_field_embedder
        self._premise_encoder = premise_encoder
        self._nodes_attention = SingleTimeDistributed(MatrixAttention(attention_similarity), 0)
        self._num_labels = vocab.get_vocab_size(namespace="labels")
        self._phrase_probability = TimeDistributed(phrase_probability)
        self._ignore_edges = ignore_edges
        if not self._ignore_edges:
            self._num_edges = vocab.get_vocab_size(namespace="edges")
            self._edge_probability = TimeDistributed(edge_probability)
            self._edge_embedding = edge_embedding
        self._use_encoding_for_node = use_encoding_for_node
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(self,
                premise: Dict[str, torch.LongTensor],
                hypothesis: Dict[str, torch.LongTensor],
                nodes: Dict[str, torch.LongTensor],
                edge_sources: torch.LongTensor,
                edge_targets: torch.LongTensor,
                edge_labels: torch.LongTensor,
                metadata: List[Dict[str, Any]] = None,
                label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        premise: Dict[str, torch.LongTensor]
            From a ``TextField`` on the premise
        hypothesis: Dict[str, torch.LongTensor]
            From a ``TextField`` on the hypothesis
        nodes: Dict[str, torch.LongTensor]
            From a ``ListField`` of ``TextField`` for the list of node phrases in the hypothesis
        edge_sources: torch.LongTensor
            From a ``ListField`` of ``IndexField`` for the list of edges in the hypothesis. The
            indices correspond to the index of source node in the list of nodes
        edge_targets: torch.LongTensor
            From a ``ListField`` of ``IndexField`` for the list of edges in the hypothesis. The
            indices correspond to the index of target node in the list of nodes
        edge_labels: torch.LongTensor
            From a ``ListField`` of ``LabelField`` for the list of edge labels in the hypothesis
        metadata: List[Dict[str, Any]]
            Metadata information
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
        # batch x premise words x emb. dim
        embedded_premise = self._text_field_embedder(premise)
        premise_mask = get_text_field_mask(premise).float()
        # mask over the nodes. dim: batch x node x node words
        nodes_mask = get_text_field_mask(nodes)

        if self._use_encoding_for_node or not self._ignore_edges:
            encoded_premise = self._premise_encoder(embedded_premise, premise_mask)

        # embeddings for each node. dim: batch x nodes x node words x emb. dim
        embedded_nodes = self._text_field_embedder(nodes)

        set_printoptions(threshold=numpy.inf, linewidth=numpy.inf)

        # node model
        if self._use_encoding_for_node:
            premise_representation = encoded_premise
        else:
            premise_representation = embedded_premise

        mean_node_premise_attention, mean_phrase_distribution = self._get_node_probabilities(
            embedded_nodes, premise_representation,
            nodes_mask, premise_mask, metadata)
        if not self._ignore_edges:
            # edge model
            mean_edge_distribution = self._get_edge_probabilities(encoded_premise,
                                                                  mean_node_premise_attention,
                                                                  edge_sources, edge_targets,
                                                                  edge_labels, metadata)

            label_logits = mean_phrase_distribution + mean_edge_distribution
        else:
            label_logits = mean_phrase_distribution
        label_probs = torch.nn.functional.softmax(label_logits)
        output_dict = {"label_logits": label_logits, "label_probs": label_probs}

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._accuracy(label_logits, label.squeeze(-1))
            output_dict["loss"] = loss
        return output_dict

    def _get_node_probabilities(self, embedded_nodes, embedded_premise, nodes_mask, premise_mask,
                                metadata) -> Tuple[FloatTensor, FloatTensor]:
        """
        Compute the average entailment distribution based on the nodes in the hypothesis.
        Returns a tuple of (attention of each node over the premise, average entailment
        distribution) with dimensions batch x nodes x premise words and batch x num classes
        respectively.
        """
        # attention for each node. dim: batch x nodes x node words x premise words
        node_premise_attention = self._nodes_attention(embedded_nodes, embedded_premise)

        normalized_node_premise_attention = last_dim_softmax(node_premise_attention, premise_mask)

        expanded_nodes_mask_premise = nodes_mask.unsqueeze(-1).expand_as(
            normalized_node_premise_attention).float()
        # aggregate representation. dim: batch x nodes x premise words
        mean_node_premise_attention = masked_mean(normalized_node_premise_attention, 2,
                                                  expanded_nodes_mask_premise)

        # convert batch x nodes and batch x premise to batch x nodes x premise mask
        nodes_only_mask = (torch.sum(nodes_mask, -1) > 0).float()
        node_premise_mask = nodes_only_mask.unsqueeze(-1).expand_as(mean_node_premise_attention) \
                            * premise_mask.unsqueeze(1).expand_as(mean_node_premise_attention)
        masked_mean_node_premise_attention = replace_masked_values(mean_node_premise_attention,
                                                                   node_premise_mask, 0)
        # aggreate node representation over premise. dim: batch x nodes x emb. dim
        aggregate_node_premise_representation = weighted_sum(embedded_premise,
                                                             masked_mean_node_premise_attention)
        expanded_nodes_mask_embedding = nodes_mask.unsqueeze(-1).expand_as(
            embedded_nodes).float()
        # dim: batch x nodes x emb. dim
        aggregate_node_representation = masked_mean(embedded_nodes, 2,
                                                    expanded_nodes_mask_embedding)

        sub_representation = aggregate_node_premise_representation - aggregate_node_representation
        dot_representation = aggregate_node_premise_representation * aggregate_node_representation
        # dim: batch x nodes x emb. dim * 4
        combined_node_representation = torch.cat([aggregate_node_premise_representation,
                                                  aggregate_node_representation,
                                                  sub_representation,
                                                  dot_representation], 2)
        # dim: batch x nodes x num_classes
        phrase_prob_distribution = self._phrase_probability(combined_node_representation)

        # ignore nodes with no text and expand to num of output classes
        # dim: batch x node x node words -> batch x node  -> batch x node x num_classes
        nodes_class_mask = nodes_only_mask.unsqueeze(-1).expand_as(
            phrase_prob_distribution).float()

        mean_phrase_distribution = masked_mean(phrase_prob_distribution, 1, nodes_class_mask)
        return mean_node_premise_attention, mean_phrase_distribution

    def _get_edge_probabilities(self, encoded_premise, mean_node_premise_attention, edge_sources,
                                edge_targets, edge_labels, metadata) -> FloatTensor:
        # dim: batch x nodes x emb. dim
        aggregate_node_premise_lstm_representation = weighted_sum(encoded_premise,
                                                                  mean_node_premise_attention)
        # dim: batch x edges x 1
        edge_mask = (edge_sources != -1).float()
        edge_source_lstm_repr = self._select_embeddings_using_index(
            aggregate_node_premise_lstm_representation,
            replace_masked_values(edge_sources.float(), edge_mask, 0))
        edge_target_lstm_repr = self._select_embeddings_using_index(
            aggregate_node_premise_lstm_representation,
            replace_masked_values(edge_targets.float(), edge_mask, 0))
        # edge label embeddings. dim: batch x edges x edge dim
        masked_edge_labels = replace_masked_values(edge_labels.float(), edge_mask, 0).squeeze(
            2).long()
        edge_label_embeddings = self._edge_embedding(masked_edge_labels)
        # dim: batch x edges x (2* emb dim + edge dim)

        combined_edge_representation = torch.cat([edge_source_lstm_repr, edge_label_embeddings,
                                                  edge_target_lstm_repr], 2)
        edge_prob_distribution = self._edge_probability(combined_edge_representation)
        edges_only_mask = edge_mask.expand_as(edge_prob_distribution).float()
        mean_edge_distribution = masked_mean(edge_prob_distribution, 1, edges_only_mask)
        return mean_edge_distribution

    @staticmethod
    def _get_unpadded_matrix_for_example(input_matrix, idx, mask) -> str:
        input_matrix_for_example = input_matrix.data.cpu().numpy()[idx]
        mask_for_example = mask.data.cpu().numpy()[idx]
        if mask_for_example.shape != input_matrix_for_example.shape:
            raise ValueError("Different shapes for mask and input: {} vs {}".format(
                mask_for_example.shape, input_matrix_for_example.shape))
        if mask_for_example.ndim != 2:
            raise ValueError("Cannot handle more than two dimensions. Found {}".format(
                mask_for_example.shape))
        # Find the max rows and columns to print
        zero_rows = numpy.argwhere(mask_for_example[:, 0] == 0)
        max_rows = numpy.min(zero_rows) if zero_rows.size != 0 else mask_for_example.shape[0]
        zero_cols = numpy.argwhere(mask_for_example[0, :] == 0)
        max_cols = numpy.min(zero_cols) if zero_cols.size != 0 else mask_for_example.shape[1]
        return array2string(input_matrix_for_example[:max_rows, :max_cols],
                            precision=4, suppress_small=True)

    @staticmethod
    def _select_embeddings_using_index(embedding_matrix, index_tensor) -> FloatTensor:
        """
        Uses the indices in index_tensor to select vectors from embedding_matrix
        :param embedding_matrix: Embeddings with dim: batch x N x emb. dim
        :param index_tensor:  Indices with dim: batch x M x 1
        :return: selected embeddings with dim: batch x M x emb. dim
        """
        if index_tensor.size()[-1] != 1:
            raise ValueError("Expecting last index to be 1. Found {}".format(index_tensor.size()))
        expanded_index_size = [x for x in index_tensor.size()[:-1]] + [embedding_matrix.size()[-1]]
        # dim: batch x M x emb. dim
        expanded_index_tensor = index_tensor.expand(expanded_index_size).long()
        return torch.gather(embedding_matrix, 1, expanded_index_tensor)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self._accuracy.get_metric(reset),
        }

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'TreeAttention':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)

        premise_encoder_params = params.pop("premise_encoder", None)
        premise_encoder = Seq2SeqEncoder.from_params(premise_encoder_params)

        attention_similarity = SimilarityFunction.from_params(params.pop('attention_similarity'))
        phrase_probability = FeedForward.from_params(params.pop('phrase_probability'))
        edge_probability = FeedForward.from_params(params.pop('edge_probability'))

        edge_embedding = Embedding.from_params(vocab, params.pop('edge_embedding'))
        use_encoding_for_node = params.pop('use_encoding_for_node')
        ignore_edges = params.pop('ignore_edges', False)

        init_params = params.pop('initializer', None)
        initializer = (InitializerApplicator.from_params(init_params)
                       if init_params is not None
                       else InitializerApplicator())

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   phrase_probability=phrase_probability,
                   edge_probability=edge_probability,
                   premise_encoder=premise_encoder,
                   edge_embedding=edge_embedding,
                   use_encoding_for_node=use_encoding_for_node,
                   attention_similarity=attention_similarity,
                   ignore_edges=ignore_edges,
                   initializer=initializer)
