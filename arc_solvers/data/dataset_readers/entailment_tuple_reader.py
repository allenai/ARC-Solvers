"""
=====================================================================
Decomposable Graph Entailment Model code replicated from SciTail repo
https://github.com/allenai/scitail
=====================================================================
"""

import logging
from builtins import ValueError
from typing import Dict, List, Set, Tuple

import tqdm
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField
from allennlp.data.fields.index_field import IndexField
from allennlp.data.fields.list_field import ListField
from allennlp.data.fields.metadata_field import MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from overrides import overrides

logger = logging.getLogger(__name__)


@DatasetReader.register("entailment_tuple")
class EntailmentTupleReader(DatasetReader):
    """
    Reads a file with entailment data with additional tuple structure for the hypothesis. The
    input file is in the format "premise\thypothesis\tlabel\ttuple structure" where the tuple
    structure is represented using "$$$" to split tuples and "<>" to split fields.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        Used to tokenize the premise, hypothesis and nodes in the hypothesis structure
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens":
    SingleIdTokenIndexer()}``)
        Used to index the tokens extracted by the tokenizer
    """

    def __init__(self,
                 max_tokens: int, max_tuples: int,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._max_tokens = max_tokens
        self._max_tuples = max_tuples
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        instances = []
        with open(file_path, 'r') as entailment_file:
            logger.info("Reading entailment instances from TSV dataset at: %s", file_path)
            for line in tqdm.tqdm(entailment_file):
                fields = line.split("\t")
                if len(fields) != 4:
                    raise ValueError("Expected four fields: "
                                     "premise   hypothesis  label   hypothesis_structure. "
                                     "Found {} fields in {}".format(len(fields), line))
                premise, hypothesis, label, hypothesis_structure = fields
                instances.append(self.text_to_instance(premise, hypothesis, hypothesis_structure,
                                                       label))
        if not instances:
            raise ConfigurationError("No instances were read from the given filepath {}. "
                                     "Is the path correct?".format(file_path))
        return Dataset(instances)

    @overrides
    def text_to_instance(self,
                         premise: str,
                         hypothesis: str,
                         hypothesis_structure: str,
                         label: str = None) -> Instance:
        fields: Dict[str, Field] = {}
        premise_tokens = self._tokenizer.tokenize(premise)[-self._max_tokens:]
        hypothesis_tokens = self._tokenizer.tokenize(hypothesis)[-self._max_tokens:]

        fields['premise'] = TextField(premise_tokens, self._token_indexers)
        fields['hypothesis'] = TextField(hypothesis_tokens, self._token_indexers)
        metadata = {
            'premise': premise,
            'hypothesis': hypothesis,
            'premise_tokens': [token.text for token in premise_tokens],
            'hypothesis_tokens': [token.text for token in hypothesis_tokens]
        }
        fields['metadata'] = MetadataField(metadata)
        self._add_structure_to_fields(hypothesis_structure, fields)
        if label:
            fields['label'] = LabelField(label)
        return Instance(fields)

    def _add_structure_to_fields(self, structure, fields) -> None:
        """
        Add structure (nodes and edges) to the instance fields. Specifically, convert
        "plants<>produce<>oxygen" into ("produce", subj, "plants"), ("produce", obj, "oxygen"),
        ("plants", subj-obj, "oxygen"). Each quoted string forms a node represented using a
        TextField. Each source and target node in an edge is represented using IndexField into
        the list of nodes and the edge label is represented using a LabelField with "edges"
        namespace.
        """
        # take the last tuples
        tuples = structure.split("$$$")[-self._max_tuples:]
        node_list, edge_list = self._extract_nodes_and_edges_from_tuples(tuples)
        if not len(node_list):
            print("No nodes in {} for premise:{} and hypothesis: {}".format(
                structure, fields['metadata'].metadata["premise"],
                fields['metadata'].metadata["hypothesis"]))
        nodes_field = ListField(node_list)
        edge_source_list = []
        edge_target_list = []
        edge_label_list = []
        for edge in edge_list:
            source_field = IndexField(edge[0], nodes_field)
            target_field = IndexField(edge[2], nodes_field)
            label_field = LabelField(edge[1], "edges")
            edge_source_list.append(source_field)
            edge_target_list.append(target_field)
            edge_label_list.append(label_field)
        fields['nodes'] = nodes_field
        # Currently AllenNLP doesn't allow for ListFields containing ListFields,
        # so creating separate ListFields for source, target and labels for the edges
        fields['edge_sources'] = ListField(edge_source_list)
        fields['edge_targets'] = ListField(edge_target_list)
        fields['edge_labels'] = ListField(edge_label_list)

    def _extract_nodes_and_edges_from_tuples(self, tuples: List[str]) -> Tuple[List[TextField],
                                                                               List[Tuple]]:
        """
        Extract the nodes and edges from the list of tuples. Returns a list of nodes and list of
        edges where the nodes are represented as list of ``TextField`` and edges as list of
        (source index, edge label, target index). The source and target indices refer to the
        index of the node in the list of nodes.
        """
        # list of string representation of the nodes used to find the index of the source/target
        # node for each edge
        node_strings = []
        node_text_fields = []
        edge_tuples = []
        for openie_tuple in tuples:
            tuple_fields = openie_tuple.split("<>")
            nodes, edges = self._extract_nodes_and_edges_from_fields(tuple_fields)
            # first, collect the nodes in the graph
            for node in nodes:
                if node not in node_strings:
                    node_tokens = self._tokenizer.tokenize(node)
                    if not node_tokens:
                        raise ValueError("Empty phrase from {}".format(node))
                    node_strings.append(node)
                    node_text_fields.append(TextField(node_tokens, self._token_indexers))
            # convert the edge representation using strings into the edge representation with
            # indices into the list of nodes compute above
            for edge in edges:
                source_idx = node_strings.index(edge[0])
                if source_idx is None:
                    raise ValueError("'{}' not found in node list: [{}]".format(
                        edge[0], ",".join(node_strings)))
                target_idx = node_strings.index(edge[2])
                if target_idx is None:
                    raise ValueError("'{}' not found in node list: [{}]".format(
                        edge[2], ",".join(node_strings)))
                edge_label = edge[1]
                edge_tuple = (source_idx, edge_label, target_idx)
                edge_tuples.append(edge_tuple)
        return node_text_fields, edge_tuples

    def _extract_nodes_and_edges_from_fields(self, fields) -> (Set[str], List[List[str]]):
        """
        Extract the nodes and edges from the fields of a tuple. Nodes are represented using their
        string and edges as [source node, edge label, target node].
        """
        nodes = set()
        edges = []
        if len(fields) < 2:
            print("Less than two fields in ({})".format(",".join(fields)))
            return nodes, edges
        subj = self._get_tokenized_rep(fields[0])
        pred = self._get_tokenized_rep(fields[1])
        if subj:
            nodes.add(subj)
        if pred:
            nodes.add(pred)
        # create a subj edge between the predicate and subject
        if subj and pred:
            edges.append([pred, "subj", subj])
        if len(fields) > 2:
            obj1 = self._get_tokenized_rep(fields[2])
            if obj1:
                nodes.add(obj1)
                # create a subj-obj edge between the subject and object
                if subj:
                    edges.append([subj, "subj-obj", obj1])
        for obj in fields[2:]:
            last_ent = pred
            # identify the object type and split longer objects, if needed
            for phrase, ptype in self._split_object_phrase(obj):
                clean_phr = self._get_tokenized_rep(phrase)
                if not clean_phr:
                    logger.warning("Unexpected empty phrase from {}".format(obj))
                nodes.add(clean_phr)
                edges.append([last_ent, ptype, clean_phr])
                last_ent = clean_phr
        return nodes, edges

    def _get_tokenized_rep(self, field):
        """
        Get a clean representation of the field based on the tokens. This ensures that
        strings with the same tokens have the same string representation.
        """
        return " ".join([x.text for x in self._tokenizer.tokenize(field.strip())])

    def _split_object_phrase(self, field: str) -> List[Tuple[str, str]]:
        """
        Break longer object phrases into shorter phrases based on the prepositions. E.g. break
        "the process of changing liquid water into water vapor" into {(the process, obj),
        (changing liquid water, of), (water vapor, into)}
        """
        clean_obj, base_type = self._get_base_object_and_type(field)
        tokens = [x.text for x in self._tokenizer.tokenize(clean_obj)]
        split_objects = []
        object_types = []
        current_obj = ""
        current_type = base_type
        for token in tokens:
            if token in self.PREPOSITION_LIST and current_obj != "":
                split_objects.append(current_obj)
                object_types.append(current_type)
                current_obj = ""
                current_type = token
            else:
                current_obj = current_obj + " " + token if current_obj != "" else token
        if current_obj != "":
            split_objects.append(current_obj)
            object_types.append(current_type)
        return list(zip(split_objects, object_types))

    def _get_base_object_and_type(self, field: str) -> Tuple[str, str]:
        """Identify the object type for the object in the OpenIE tuple"""
        if field.startswith("L:"):
            return field[2:], "L"
        if field.startswith("T:"):
            return field[2:], "T"
        for prep in self.PREPOSITION_LIST:
            if field.startswith(prep + " "):
                return field[len(prep) + 1:], prep
        # if no match found, use the generic obj type
        return field, "obj"

    PREPOSITION_LIST = ["with", "at", "from", "into", "during", "including", "until", "against",
                        "among", "throughout", "despite", "towards", "upon", "concerning", "of",
                        "to", "in", "for", "on", "by", "about", "like", "through", "over",
                        "before", "between", "after", "since", "without", "under", "within",
                        "along", "following", "across", "behind", "beyond", "plus", "except",
                        "but", "up", "out", "around", "down", "off", "above", "near"]

    @classmethod
    def from_params(cls, params: Params) -> 'EntailmentTupleReader':
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        max_tuples = params.pop('max_tuples', 30)
        max_tokens = params.pop('max_tokens', 200)
        params.assert_empty(cls.__name__)
        return EntailmentTupleReader(max_tokens=max_tokens,
                                     max_tuples=max_tuples,
                                     tokenizer=tokenizer,
                                     token_indexers=token_indexers)
