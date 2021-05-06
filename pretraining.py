# encoding: utf-8
# python-version: 3.8

import absl
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
import utils
import bert_config
import bert_model
import optimizer

# Required parameters
absl.flags.DEFINE_string(
  "bert_config_file", None,
  "The config json file corresponding to the pre-trained BERT model. "
  "This specifies the model architecture."
)
absl.flags.DEFINE_string(
  "input_file", None,
  "Input TF example files (can be a glob or comma separated)."
)
absl.flags.DEFINE_string(
  "output_dir", None,
  "The output directory where the model checkpoints will be written."
)

# Other parameters
absl.flags.DEFINE_bool(
  "do_train", False,
  "Whether to run training."
)
absl.flags.DEFINE_bool(
  "do_eval", False,
  "Whether to run eval on the dev set."
)
absl.flags.DEFINE_integer(
  "train_batch_size", 32,
  "Total batch size for training"
)
absl.flags.DEFINE_integer(
  "eval_batch_size", 32,
  "Total batch size for eval"
)
absl.flags.DEFINE_integer(
  "max_seq_length", 64,
  "The maximum total input sequence length after WordPiece tokenization. "
  "Sequences longer than this will be truncated, and sequences shorter "
  "than this will be padded. Must match data generation."
)
absl.flags.DEFINE_integer(
  "num_train_steps", 100000,
  "Number of training steps."
)
absl.flags.DEFINE_integer(
  "num_warmup_steps", 10000,
  "Number of warmup steps."
)
absl.flags.DEFINE_integer(
  "max_predictions_per_seq", 2,
  "Maximum number of masked LM predictions per sequence. "
  "Must match data generation."
)
absl.flags.DEFINE_float(
  "learning_rate", 5e-5,
  "The initial learning rate for Adam."
)
absl.flags.DEFINE_integer(
  "num_parallel_calls", 2,
  ""
)


@tf.function
def get_input(
  input_files,
  is_training,
):
  if is_training:
    # shuffle example files
    data_set = tf.data.Dataset.from_tensor_slices(input_files)
    data_set = data_set.repeat()
    print(len(input_files))
    data_set = data_set.shuffle(buffer_size = len(input_files))

    # interleave examples between files
    cycle_length = max(absl.flags.FLAGS.num_parallel_calls, len(input_files))
    data_set = data_set.interleave(
      tf.data.TFRecordDataset,
      cycle_length = cycle_length,
      block_length = 1,
      num_parallel_calls = absl.flags.FLAGS.num_parallel_calls,
      deterministic = not is_training,
    )

    # shuffle examples
    data_set = data_set.shuffle(buffer_size = 100)

  else:
    # read example files
    data_set = tf.data.TFRecordDataSet(input_files)
    
    # Since we evaluate for a fixed number of steps we don't want to encounter
    # out-of-range exceptions.
    data_set = data_set.repeat()

  # decode examples
  name_to_features = {
    "input_ids":
      tf.io.FixedLenFeature([absl.flags.FLAGS.max_seq_length], tf.int64),
    "input_mask":
      tf.io.FixedLenFeature([absl.flags.FLAGS.max_seq_length], tf.int64),
    "segment_ids":
      tf.io.FixedLenFeature([absl.flags.FLAGS.max_seq_length], tf.int64),
    "masked_lm_positions":
      tf.io.FixedLenFeature([absl.flags.FLAGS.max_predictions_per_seq], tf.int64),
    "masked_lm_ids":
      tf.io.FixedLenFeature([absl.flags.FLAGS.max_predictions_per_seq], tf.int64),
    "masked_lm_weights":
      tf.io.FixedLenFeature([absl.flags.FLAGS.max_predictions_per_seq], tf.float32),
    "next_sentence_labels":
      tf.io.FixedLenFeature([1], tf.int64),
  }
  data_set = data_set.map(
    lambda record: tf.io.parse_single_example(record, name_to_features),
    num_parallel_calls = absl.flags.FLAGS.num_parallel_calls,
    deterministic      = True
  )

  # batch 
  if is_training:
    data_set = data_set.batch(
      absl.flags.FLAGS.train_batch_size, drop_remainder = True
    )
  else:
    data_set = data_set.batch(
      absl.flags.FLAGS.eval_batch_size, drop_remainder = False
    )

  return data_set


def train_fn(bert_config, input_files, output_dir):
  train_dataset = get_input(input_files, is_training = True)
  model         = bert_model.BertModel(bert_config)

  opt = optimizer.create_optimizer(
    init_lr = absl.flags.FLAGS.learning_rate,
    num_train_steps = absl.flags.FLAGS.num_train_steps,
    num_warmup_steps = absl.flags.FLAGS.num_warmup_steps
  )

  global_step = 0
  while global_step < absl.flags.FLAGS.num_train_steps:
    for step, batch in enumerate(train_dataset):
      with tf.GradientTape() as tape:
        ( pooled_output,
          sequence_output,
          all_encoder_output,
          embedding_output,
        ) = model(
          batch["input_ids"],
          batch["input_mask"],
          batch["segment_ids"],
          training = True
        )

        ( masked_lm_loss,
          masked_lm_example_loss,
          masked_lm_log_probs
        ) = model.get_masked_lm_output(
          sequence_output,
          batch["masked_lm_positions"],
          batch["masked_lm_ids"],
          batch["masked_lm_weights"]
        )

        ( next_sentence_loss,
          next_sentence_example_loss,
          next_sentence_log_probs
        ) = model.get_next_sentence_output(
          pooled_output,
          batch["next_sentence_labels"]
        )

        loss_value = masked_lm_loss + next_sentence_loss

      trainable_variables = model.trainable_variables()

      grads = tape.gradient(loss_value, trainable_variables)
      opt.apply_gradients(zip(grads, trainable_variables))
      
      real_step = global_step + step
      if real_step % 200 == 0:
        absl.logging.info(
          "Training loss (for one batch) at step %d: %.6f" % (real_step, float(loss_value))
        )
        absl.logging.info(
          "Seen so far: %s samples" % ((real_step + 1) * absl.flags.FLAGS.train_batch_size)
        )

    global_step += step

def main(argv):
  # bert_config
  config = bert_config.BertConfig.from_json_file(
    absl.flags.FLAGS.bert_config_file
  )

  # input files
  input_files = []
  for input_pattern in absl.flags.FLAGS.input_file.split(","):
    input_files.extend(tf.io.gfile.glob(input_pattern))

  absl.logging.info("*** Input Files ***")
  for input_file in input_files:
    absl.logging.info("  %s" % input_file)

  # create output directory
  tf.io.gfile.makedirs(absl.flags.FLAGS.output_dir)

  if absl.flags.FLAGS.do_train:
    train_fn(config, input_files, absl.flags.FLAGS.output_dir)

#  if absl.flags.FLAGS.do_eval:

if __name__ == "__main__":
  absl.logging.set_verbosity(absl.logging.INFO)

  absl.flags.mark_flag_as_required("bert_config_file")
  absl.flags.mark_flag_as_required("input_file")
  absl.flags.mark_flag_as_required("output_dir")

  absl.app.run(main)

