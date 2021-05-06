# encoding: utf-8
# python-version: 3.8

import copy
import math
import tensorflow as tf
import utils
import bert_config
import transformer_model

class BertModel:
  def __init__(
    self,
    config
  ):
    super(BertModel, self).__init__()
    self.config = copy.deepcopy(config)

    # embedding
    self.token_embedding_layer = tf.keras.layers.Embedding(
      name       = "BERT/embedding/token_ids",
      input_dim  = self.config.vocab_size,
      output_dim = self.config.hidden_size,
      embeddings_initializer = tf.keras.initializers.TruncatedNormal(
        mean = 0.0,
        stddev = self.config.initializer_range,
        seed = None
      ),
      embeddings_regularizer = None,
      embeddings_constraint  = None,
      mask_zero             = False,
      input_length          = None
    )

    self.token_type_embedding_layer = tf.keras.layers.Embedding(
      name       = "BERT/embedding/token_type_ids",
      input_dim  = self.config.type_vocab_size,
      output_dim = self.config.hidden_size,
      embeddings_initializer = tf.keras.initializers.TruncatedNormal(
        mean = 0.0,
        stddev = self.config.initializer_range,
        seed = None
      ),
      embeddings_regularizer = None,
      embeddings_constraint  = None,
      mask_zero             = False,
      input_length          = None
    )

    self.position_embedding_layer = tf.keras.layers.Embedding(
      name       = "BERT/embedding/position_ids",
      input_dim  = self.config.max_position_embeddings,
      output_dim = self.config.hidden_size,
      embeddings_initializer = tf.keras.initializers.TruncatedNormal(
        mean = 0.0,
        stddev = self.config.initializer_range,
        seed = None
      ),
      embeddings_regularizer = None,
      embeddings_constraint  = None,
      mask_zero             = False,
      input_length          = None
    )

    # encoder - transformer_model
    self.encoder_transformer = transformer_model.TransformerModel(
      num_hidden_layers              = self.config.num_hidden_layers,
      hidden_size                    = self.config.hidden_size,
      num_attention_heads            = self.config.num_attention_heads,
      initializer_range              = self.config.initializer_range,
      attention_probs_dropout_prob   = self.config.attention_probs_dropout_prob,
      hidden_dropout_prob            = self.config.hidden_dropout_prob,
      intermediate_size              = self.config.intermediate_size,
      intermediate_act_fn            = self.config.hidden_act,
      scope = "BERT/Encoder"
    )

    self.pooled_output_layer = tf.keras.layers.Dense(
      name  = "BERT/pooled_output",
      units = self.config.hidden_size,
      activation = tf.tanh,
      kernel_initializer = tf.keras.initializers.TruncatedNormal(
        mean = 0.0,
        stddev = self.config.initializer_range,
        seed = None
      )
    )

    self.masked_lm_dense = tf.keras.layers.Dense(
      name  = "BERT/masked_lm_dense",
      units = self.config.hidden_size,
      activation = utils.get_activation(self.config.hidden_act),
      kernel_initializer = tf.keras.initializers.TruncatedNormal(
        mean = 0.0,
        stddev = self.config.initializer_range,
        seed = None
      )
    )

    self.masked_lm_bias = tf.Variable(
      name = "BERT/masked_lm_bias",
      initial_value = tf.zeros_initializer()(
        shape = [self.config.vocab_size],
        dtype = tf.float32
      )
    )

    self.next_sentence_dense = tf.keras.layers.Dense(
      name  = "BERT/next_sentence_dense_dense",
      units = 2,
      kernel_initializer = tf.keras.initializers.TruncatedNormal(
        mean = 0.0,
        stddev = self.config.initializer_range,
        seed = None
      )
    )


  def __call__(
    self,
    input_ids,
    input_mask  = None,
    segment_ids = None,
    training = True
  ):
    if input_mask is None:
      input_mask = tf.ones(shape = [batch_size, seq_length], dtype = tf.int32)

    if segment_ids is None:
      segment_ids = tf.zeros(shape = [batch_size, seq_length], dtype = tf.int32)

    input_shape = utils.get_shape_list(input_ids, expected_rank = 2)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    # embedding
    x = self.token_embedding_layer(input_ids)

    if self.config.use_token_type_embeddings:
      y = self.token_type_embedding_layer(segment_ids)
      x = tf.math.add(x, y)

    if self.config.use_position_embeddings:
      if seq_length > self.config.max_position_embeddings:
        raise ValueError(
          "`seq_length` (%d) must no be greater than `max_position_embeddings` (%d)." %
          (seq_length, max_position_embeddings)
        )
      position_ids = tf.convert_to_tensor(
        [[x for x in range(seq_length)] for _ in range(batch_size)],
        dtype = tf.int32
      )
      y = self.position_embedding_layer(position_ids)
      x = tf.math.add(x, y)

    x = utils.layer_norm(x)
    if training:
      embedding_output = tf.nn.dropout(x = x, rate = self.config.hidden_dropout_prob)
    else:
      embedding_output = x

    # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
    # mask of shape [batch_size, seq_length, seq_length] which is used
    # for the attention scores.
    to_mask = tf.cast(
      tf.reshape(input_mask, [batch_size, 1, seq_length]),
      tf.float32
    )
    broadcast_ones = tf.ones(
      shape = [batch_size, seq_length, 1],
      dtype = tf.float32
    )
    attention_mask = tf.multiply(broadcast_ones, to_mask)

    all_encoder_output = self.encoder_transformer(
      input_tensor = embedding_output,
      attention_mask = attention_mask,
      do_return_all_layers = True,
      training = training
    )

    sequence_output = all_encoder_output[-1]
    # The "pooler" converts the encoded sequence tensor of shape
    # [batch_size, seq_length, hidden_size] to a tensor of shape
    # [batch_size, hidden_size]. This is necessary for segment-level
    # (or segment-pair-level) classification tasks where we need a fixed
    # dimensional representation of the segment.

    # We "pool" the model by simply taking the hidden state corresponding
    # to the first token. We assume that this has been pre-trained
    first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis = 1)
    pooled_output = self.pooled_output_layer(first_token_tensor)

    return (
      pooled_output,
      sequence_output,
      all_encoder_output,
      embedding_output,
    )

  def get_masked_lm_output(
    self,
    sequence_output,
    masked_lm_positions,
    masked_lm_ids,
    masked_lm_weights
  ):
    """Get loss and log probs for the masked LM."""
    x = utils.gather_indexes(sequence_output, masked_lm_positions)

    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    x = self.masked_lm_dense(x)
    x = utils.layer_norm(x)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    token_embedding_matrix = self.token_embedding_layer.variables[0]

    x = tf.matmul(x, token_embedding_matrix, transpose_b = True)
    logits = tf.nn.bias_add(x, self.masked_lm_bias)
    log_probs = tf.nn.log_softmax(logits, axis = -1)

    label_ids = tf.reshape(masked_lm_ids, [-1])
    label_weights = tf.reshape(masked_lm_weights, [-1])

    one_hot_labels = tf.one_hot(
      label_ids, depth = self.config.vocab_size, dtype = tf.float32
    )

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis = [-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

    return (loss, per_example_loss, log_probs)


  def get_next_sentence_output(
    self,
    pooled_output,
    next_sentence_labels
  ):
    """Get loss and log probs for the next sentence prediction."""

    # Simple binary classification. Note that 0 is "next sentence" and 1 is
    # "random sentence". This weight matrix is not used after pre-training.
    logits = self.next_sentence_dense(pooled_output)
    log_probs = tf.nn.log_softmax(logits, axis = -1)

    labels = tf.reshape(next_sentence_labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth = 2, dtype = tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis = -1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, per_example_loss, log_probs)


  def trainable_variables(self):
    variables = []
    variables.extend(self.token_embedding_layer.trainable_weights)
    variables.extend(self.token_type_embedding_layer.trainable_weights)
    variables.extend(self.position_embedding_layer.trainable_weights)
    variables.extend(self.encoder_transformer.trainable_variables())
    variables.extend(self.pooled_output_layer.trainable_weights)
    variables.extend(self.masked_lm_dense.trainable_weights)
    variables.append(self.masked_lm_bias)
    variables.extend(self.next_sentence_dense.trainable_weights)
    return variables

