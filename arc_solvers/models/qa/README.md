# Models for Question Answering
This document contains a description of the neural models for question answering.

## Multi Choice QA

### BiLSTM Max-out with Question to Choices Max Attention
The [`QAMultiChoiceMaxAttention`](multi_choice/qa_multi_choice_max_att.py)
model computes the attention interaction between the context-encoded
representations of the question and the choices.

At a high level, the model works as follows:
1. Obtain a BiLSTM context representation of the token sequences of the
`question` and each `choice`.
2. Get an aggregated (single vector) representations for `question` and
`choice` using element-wise `max` operation.
3. Compute the score for `choice` as `linear_layer([u, v, u - v, u * v])`,
 where `u` and `v` are the `choice` and `question` representations
 respectively from Step 2.
4. Select as answer the `choice` with the highest score.

Pseudo code of the model is presented below:

```python
# encode question and each choice
question_encoded = context_enc(question_tokens)  # context_enc can be any AllenNLP supported context encoder or None. Bi-directional LSTM is used
choice_encoded = context_enc(choice_tokens)  # seq_length X hidden_size

# get a single vector representations for question and choice
question_aggregate = aggregate_method(question_encoded)  # aggregate_method can be max, min, avg. ``max`` is used.
choice_aggregate = aggregate_method(choice_encoded)  # seq_length X hidden_size

# score for each choice -  the output is a scalar value
att_q_to_ch = linear_layer([question_aggregate,
                                   choice_aggregate,
                                   choice_aggregate - question_aggregate,
                                   question_aggregate * choice_aggregate])  # size 1

# The scores `att_q_to_ch` for the four choices are normalized using ``softmax``
# and the choice with the highest score is selected as the answer.
answer_id = argmax(softmax([att_q_to_ch0, att_q_to_ch1, att_q_to_ch2, att_q_to_ch3]))

```

The model is inspired by the BiLSTM Max-Out model from [Conneau, A. et al. (2017)](https://www.semanticscholar.org/paper/Supervised-Learning-of-Universal-Sentence-from-Data-Conneau-Kiela/80b076d5c01b8cb2611156e4bcb987f850efcb9c).

#### Training and evaluation of the model

To train the model, download the data and word embeddings
(see [Setup data/models](../../../README.md#setup-datamodels)).

Evaluate the trained model:
```bash
python arc_solvers/run.py evaluate \
    --archive_file data/ARC-V1-Models-Aug2018/max_att/model.tar.gz \
    --evaluation_data_file data/ARC-V1-Feb2018/ARC-Challenge/ARC-Challenge-Test.jsonl
```

or

Train a new model:
```bash
python arc_solvers/run.py train \
    -s trained_models/qa_multi_question_to_choices/serialization/ \
    arc_solvers/training_config/qa/multi_choice/reader_qa_multi_choice_max_att_ARC_Chellenge_full.json
```
