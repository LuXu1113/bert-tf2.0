# encoding: utf-8
# python-version: 3.8

import tensorflow as tf
import multi_head_attention_layer as mhal
import utils

class TransformerModel:
  def __init__(
    self,
    num_hidden_layers,
    hidden_size,
    num_attention_heads,
    initializer_range,
    attention_probs_dropout_prob,
    hidden_dropout_prob,
    intermediate_size,
    intermediate_act_fn,
    scope = "Global"
  ):
    self.layers = []
    for layer_idx in range(num_hidden_layers):
      self.layers.append({})

      self.layers[layer_idx]["multi_head_attention_layer"] = mhal.MultiHeadAttentionLayer(
        hidden_size                  = hidden_size,
        num_attention_heads          = num_attention_heads,
        initializer_range            = initializer_range,
        attention_probs_dropout_prob = attention_probs_dropout_prob,
        scope                        = "%s/TransformerModel/Layer-%d" % (scope, layer_idx)
      )
      self.layers[layer_idx]["linear_projection"] = tf.keras.layers.Dense(
        name  = "%s/TransformerModel/Layer-%d/LinearProjection" % (scope, layer_idx),
        units = hidden_size,
        kernel_initializer = tf.keras.initializers.TruncatedNormal(
          mean = 0.0,
          stddev = initializer_range,
          seed = None
        )
      )
      self.layers[layer_idx]["linear_projection_dropout"] = tf.keras.layers.Dropout(
        name = "%s/TransformerModel/Layer-%d/LinearProjectionDropout" % (scope, layer_idx),
        rate = hidden_dropout_prob
      )
      self.layers[layer_idx]["linear_projection_norm"] = tf.keras.layers.LayerNormalization(
        name = "%s/TransformerModel/Layer-%d/LinearProjectionNorm" % (scope, layer_idx),
        axis = -1,
        epsilon = 1e-12,
        dtype = tf.float32
      )
      self.layers[layer_idx]["intermediate"] = tf.keras.layers.Dense(
        name  = "%s/TransformerModel/Layer-%d/Intermediate" % (scope, layer_idx),
        units = intermediate_size,
        activation = utils.get_activation(intermediate_act_fn),
        kernel_initializer = tf.keras.initializers.TruncatedNormal(
          mean = 0.0,
          stddev = initializer_range,
          seed = None
        )
      )
      self.layers[layer_idx]["output"] = tf.keras.layers.Dense(
        name  = "%s/TransformerModel/Layer-%d/Output" % (scope, layer_idx),
        units = hidden_size,
        kernel_initializer = tf.keras.initializers.TruncatedNormal(
          mean = 0.0,
          stddev = initializer_range,
          seed = None
        )
      )
      self.layers[layer_idx]["output_dropout"] = tf.keras.layers.Dropout(
        name = "%s/TransformerModel/Layer-%d/OutputDropout" % (scope, layer_idx),
        rate = hidden_dropout_prob
      )
      self.layers[layer_idx]["output_norm"] = tf.keras.layers.LayerNormalization(
        name = "%s/TransformerModel/Layer-%d/OutputNorm" % (scope, layer_idx),
        axis = -1,
        epsilon = 1e-12,
        dtype = tf.float32
      )


  def __call__(self,
    input_tensor,
    attention_mask = None,
    do_return_all_layers = False,
    training = True,
  ):
    input_shape = utils.get_shape_list(input_tensor, expected_rank = 3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    prev_output = utils.reshape_to_matrix(input_tensor)

    all_layer_outputs = []
    for layer_idx in range(len(self.layers)):
      layer_input = prev_output

      attention_heads = []
      attention_head = self.layers[layer_idx]["multi_head_attention_layer"](
        from_tensor     = layer_input,
        to_tensor       = layer_input,
        batch_size      = batch_size,
        from_seq_length = seq_length,
        to_seq_length   = seq_length,
        attention_mask  = attention_mask,
        do_return_2d_tensor = True,
        training = training
      )
      attention_heads.append(attention_head)

      attention_output = None
      if len(attention_heads) == 1:
        attention_output = attention_heads[0]
      else:
        # In the case where we have other sequences, we just concatenate
        # them to the self-attention head before the projection.
        attention_output = tf.concat(attention_heads, axis = -1)
      
      # Run a linear projection of `hidden_size` then add a residual
      # with `layer_input`.
      attention_output = self.layers[layer_idx]["linear_projection"](
        attention_output
      )
      if training:
        attention_output = self.layers[layer_idx]["linear_projection_dropout"](
          attention_output,  
        )
      attention_output = self.layers[layer_idx]["linear_projection_norm"](
        tf.add(attention_output, layer_input)
      )

      # The activation is only applied to the "intermediate" hidden layer.
      intermediate_output = self.layers[layer_idx]["intermediate"](
        attention_output
      )

      # Down-project back to `hidden_size` then add the residual.
      layer_output = self.layers[layer_idx]["output"](
        intermediate_output
      )
      if training:
        layer_output = self.layers[layer_idx]["output_dropout"](
          layer_output
        )
      layer_output = self.layers[layer_idx]["output_norm"](
        tf.add(layer_output, attention_output)
      )

      prev_output = layer_output
      all_layer_outputs.append(layer_output)

    if do_return_all_layers:
      final_outputs = []
      for layer_output in all_layer_outputs:
        final_output = utils.reshape_from_matrix(layer_output, input_shape)
        final_outputs.append(final_output)
      return final_outputs
    else:
      final_outputs = utils.reshape_from_matrix(prev_output, input_shape)
      return final_outputs

  def trainable_variables(self):
    variables = []
    for layer_idx in range(len(self.layers)):
      variables.extend(
        self.layers[layer_idx]["multi_head_attention_layer"].trainable_variables()
      )
      variables.extend(
        self.layers[layer_idx]["linear_projection"].trainable_weights
      )
      variables.extend(
        self.layers[layer_idx]["intermediate"].trainable_weights
      )
      variables.extend(
        self.layers[layer_idx]["output"].trainable_weights
      )
    return variables
