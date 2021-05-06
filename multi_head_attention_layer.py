# encoding: utf-8
# python-version: 3.8

import tensorflow as tf
import utils

class MultiHeadAttentionLayer:
  def __init__(
    self,
    hidden_size,
    num_attention_heads,
    initializer_range,
    attention_probs_dropout_prob = 0.0,
    query_act = None,
    key_act   = None,
    value_act = None,
    scope = "Global"
  ):
    if hidden_size % num_attention_heads != 0:
      raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (hidden_size, num_attention_heads)
      )

    self.num_attention_heads = num_attention_heads
    self.size_per_head = int(hidden_size / num_attention_heads)

    self.query_layer = tf.keras.layers.Dense(
      name  = "%s/MultiHeadAttentionLayer/Query" % (scope),
      units = self.num_attention_heads * self.size_per_head,
      activation = utils.get_activation(query_act),
      kernel_initializer = tf.keras.initializers.TruncatedNormal(
        mean = 0.0,
        stddev = initializer_range,
        seed = None
      )
    )

    self.key_layer = tf.keras.layers.Dense(
      name  = "%s/MultiHeadAttentionLayer/Key" % (scope),
      units = self.num_attention_heads * self.size_per_head,
      activation = utils.get_activation(key_act),
      kernel_initializer = tf.keras.initializers.TruncatedNormal(
        mean = 0.0,
        stddev = initializer_range,
        seed = None
      )
    )

    self.value_layer = tf.keras.layers.Dense(
      name  = "%s/MultiHeadAttentionLayer/Value" % (scope),
      units = self.num_attention_heads * self.size_per_head,
      activation = utils.get_activation(value_act),
      kernel_initializer = tf.keras.initializers.TruncatedNormal(
        mean = 0.0,
        stddev = initializer_range,
        seed = None
      )
    )

    self.attention_probs_softmax_layer = tf.keras.layers.Softmax(
      name = "%s/MultiHeadAttentionLayer/AttentionProbsSoftmax" % (scope)
    )

    self.attention_probs_dropout_layer = tf.keras.layers.Dropout(
      name = "%s/MultiHeadAttentionLayer/AttentionProbsDropout" % (scope),
      rate = attention_probs_dropout_prob
    )

  def __call__(
    self,
    from_tensor,
    to_tensor, 
    batch_size = None,
    from_seq_length = None,
    to_seq_length = None,
    attention_mask = None,
    do_return_2d_tensor = False,
    training = True
  ):
    from_shape = utils.get_shape_list(from_tensor, expected_rank = [2, 3])
    to_shape   = utils.get_shape_list(to_tensor, expected_rank = [2, 3])

    if len(from_shape) != len(to_shape):
      raise ValueError(
        "The rank of `from_tensor` must match the rank of `to_tensor`."
      )

    if len(from_shape) == 3:
      batch_size      = from_shape[0]
      from_seq_length = from_shape[1]
      to_seq_length   = to_shape[1]
    else:
      if (batch_size is None) or (from_seq_length is None) or (to_seq_length is None):
        raise ValueError(
          "When passing in rank 2 tensors to attention_layer, the values "
          "for `batch_size`, `from_seq_length`, and `to_seq_length` "
          "must all be specified."
        )

    from_tensor_2d = utils.reshape_to_matrix(from_tensor)
    to_tensor_2d   = utils.reshape_to_matrix(to_tensor)

    # `query_out` = [batch_size*from_seq_length, num_attention_heads*size_per_head]
    query_out = self.query_layer(from_tensor_2d)
    query_out = tf.reshape(
      query_out, [batch_size, from_seq_length, self.num_attention_heads, self.size_per_head]
    )
    query_out = tf.transpose(query_out, [0, 2, 1, 3])

    # `key_out` = [batch_size*to_seq_length, num_attention_heads*size_per_head]
    key_out   = self.key_layer(to_tensor_2d)
    key_out = tf.reshape(
      key_out, [batch_size, to_seq_length, self.num_attention_heads, self.size_per_head]
    )
    key_out = tf.transpose(key_out, [0, 2, 1, 3])

    # `value_out` = [batch_size*to_seq_length, num_attention_heads*size_per_head]
    value_out = self.value_layer(to_tensor_2d)
    value_out = tf.reshape(
      value_out,
      [batch_size, to_seq_length, self.num_attention_heads, self.size_per_head]
    )
    value_out = tf.transpose(value_out, [0, 2, 1, 3])

    # `attention_scroes` = [batch_size, num_attention_heads, from_seq_length, to_seq_length]
    attention_scores = tf.matmul(query_out, key_out, transpose_b = True)
    attention_scores = tf.multiply(attention_scores, 1.0 / tf.math.sqrt(float(self.size_per_head)))

    if attention_mask is not None:
      # `attention_mask` = [batch_size, 1, from_sequence_length, to_sequence_length]
      attention_mask = tf.expand_dims(attention_mask, axis = [1])
      attention_adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
      attention_scores = tf.add(attention_scores, attention_adder)

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [batch_size, num_attention_heads, from_seq_length, to_seq_length]
    attention_probs = self.attention_probs_softmax_layer(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    if training:
      attention_probs = self.attention_probs_dropout_layer(attention_probs)
      

    # `context_out` = [batch_size, num_attention_heads, from_seq_length, size_per_head]
    context_out = tf.matmul(attention_probs, value_out)
    # `context_out` = [batch_size, from_seq_length, num_attention_heads, size_per_head]
    context_out = tf.transpose(context_out, [0, 2, 1, 3])

    if do_return_2d_tensor:
      # `context_out` = [batch_size*num_attention_heads, from_seq_length*size_per_head]
      context_out = tf.reshape(
        context_out,
        [batch_size * from_seq_length, self.num_attention_heads * self.size_per_head]
      )
    else:
      # `context_out` = [batch_size, num_attention_heads, from_seq_length*size_per_head]
      context_out = tf.reshape(
        context_out,
        [batch_size, from_seq_length, self.num_attention_heads * self.size_per_head]
      )
    return context_out

  def trainable_variables(self):
    variables = []
    variables.extend(self.query_layer.trainable_weights)
    variables.extend(self.key_layer.trainable_weights)
    variables.extend(self.value_layer.trainable_weights)
    return variables

