from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from metrics import f1 as f1_metric
from nets.s3dg import *
import numpy as np
from os import cpu_count, path, putenv
from s3dg_vars import s3dg_vars
from preprocessing.s3dg_preprocessing import preprocess_video
from metric_weights_96 import metric_weights, weight_bounds

slim = tf.contrib.slim

# adapted from tf.slim train image classifier
def get_variables_to_train(trainable_scopes):
  """Returns a list of variables to train.

  Returns:
    A list of variables to train by the optimizer.
  """
  if trainable_scopes is None:
    return tf.trainable_variables()
  # The provided trainable scoped may be a list of strings or a single string
  # with comma-separated scopes
  scopes = [scope.strip() for scope in trainable_scopes.split(',')] \
    if isinstance(trainable_scopes, str) else trainable_scopes
  variables_to_train = []
  for scope in scopes:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    variables_to_train.extend(variables)
  return variables_to_train

# The slim-based implementation of s3dg expects its
# prediction_fn to accept a scope argument
def scoped_sigmoid(logits, scope=None):
  with tf.name_scope(scope):
    return tf.sigmoid(logits)

# The estimator API expects a single model_fn to support training, evaluation
# or prediction, depending on the mode passed to the model_fn by the estimator.
def s3dg_fn(features, labels, mode, params):
  # Compute logits.
  with slim.arg_scope(s3dg_arg_scope(weight_decay=params['weight_decay'])):
    logits, endpoints = s3dg(
      features,
      num_classes=params['num_classes'],
      dropout_keep_prob=1. - params['dropout_rate'],
      is_training=mode == tf.estimator.ModeKeys.TRAIN,
      prediction_fn=scoped_sigmoid,
      min_depth=params['min_depth'],
      depth_multiplier=params['depth_multiplier'])

  # Compute predictions using round instead of argmax since our prediction
  # function is sigmoid (for multi-label classification) and not softmax
  # (for multi-class classification).
  predicted_classes = tf.round(endpoints['Predictions'])

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
      'class_ids': predicted_classes,
      'probabilities': endpoints['Predictions'],
      'logits': logits,
    }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  # Compute primary loss.
  sigmoid_loss = tf.losses.sigmoid_cross_entropy(labels, logits)
  tf.summary.scalar('Losses/sigmoid_loss', sigmoid_loss)

  # L1 loss is not included by default, but helps with our particular task
  for var in tf.trainable_variables():
    if var.op.name.find(r'weights') > 0 \
        and var not in tf.get_collection(tf.GraphKeys.WEIGHTS):
      tf.add_to_collection(tf.GraphKeys.WEIGHTS, var)

  l1_loss = tf.contrib.layers.apply_regularization(
    regularizer=tf.contrib.layers.l1_regularizer(scale=params['weight_decay']),
    weights_list=tf.get_collection(tf.GraphKeys.WEIGHTS))
  tf.summary.scalar('Losses/l1_loss', l1_loss)

  # L2 loss is already computed when utilizing the slim argument scope,
  # including the weight decay arument. Just display the existing value
  l2_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
  tf.summary.scalar('Losses/l2_loss', l2_loss)

  regularization_loss = tf.add(l1_loss, l2_loss)
  tf.summary.scalar('Losses/regularization_loss', regularization_loss)

  total_loss = tf.add(sigmoid_loss, regularization_loss)
  tf.summary.scalar('Losses/total_loss', total_loss)

  # Compute evaluation metrics.
  auc = tf.metrics.auc(labels=labels, predictions=predicted_classes,
                       name='auc_op', weights=params['metric_weights'])

  precision = tf.metrics.precision(
    labels=labels, predictions=predicted_classes, name='precision_op',
    weights=params['metric_weights'])

  recall = tf.metrics.recall(labels=labels, predictions=predicted_classes,
                             name='recall_op', weights=params['metric_weights'])

  f1 = f1_metric(labels=labels, predictions=predicted_classes, name='f1_op',
                 weights=params['metric_weights'])

  if mode == tf.estimator.ModeKeys.EVAL:
    metrics = {
      'Metrics/eval/auc': auc,
      'Metrics/eval/f1': f1,
      'Metrics/eval/precision': precision,
      'Metrics/eval/recall': recall
    }

    return tf.estimator.EstimatorSpec(
      mode, loss=total_loss, eval_metric_ops=metrics)

  # Create training op.
  assert mode == tf.estimator.ModeKeys.TRAIN

  if params['add_image_summaries']:
    for batch_num in range(params['batch_size']):
      tf.summary.image('processed_video_frame', tf.expand_dims(
        features[batch_num, int(params['clip_length'] / 2)], 0))

  # Add summaries for end_points.
  for endpoint in endpoints:
    x = endpoints[endpoint]
    tf.summary.histogram('activations/' + endpoint, x)
    tf.summary.scalar('sparsity/' + endpoint, tf.nn.zero_fraction(x))

  # Add summaries if we are training only and not evaluating
  # If evaluating, the estimator spec will add summaries automatically
  tf.summary.scalar('Metrics/train/auc', auc[1])
  tf.summary.scalar('Metrics/train/precision', precision[1])
  tf.summary.scalar('Metrics/train/recall', recall[1])
  tf.summary.scalar('Metrics/train/f1', f1[1])

  # Add histograms for variables.
  for variable in tf.global_variables():
    tf.summary.histogram(variable.op.name, variable)

  # prepare optimizer.
  if params['optimizer'] == 'momentum':
    #SGD + Momentum is the optimizer used to pre-train s3dg
    optimizer = tf.train.MomentumOptimizer(
      learning_rate=params['learning_rate'], momentum=params['momentum'])
  else:
    # pure SDG is a safe optimizer to use when troubleshooting problems
    # restoring Momentum variables from checkpoints using the Estimator API
    optimizer = tf.train.GradientDescentOptimizer(
      learning_rate=params['learning_rate'])

  variables_to_train = get_variables_to_train(params['variables_to_train'])

  train_op = tf.contrib.training.create_train_op(
    total_loss=total_loss, optimizer=optimizer,
    variables_to_train=variables_to_train)

  return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)

# Parse the input tf.Example proto using the dictionary above.
feature_spec = {
  'feature': tf.FixedLenFeature([], tf.string),
  'label': tf.FixedLenFeature([], tf.string)
}

# Create a dictionary describing the features.
def parse_serialized_example(example_proto):
  tf.logging.info('example_proto: {}'.format(example_proto))
  # Parse the input tf.Example proto using the dictionary above.
  example = tf.parse_single_example(example_proto, feature_spec)
  tf.logging.info('example: {}'.format(example))
  return example['feature'], example['label']

def main(argv):
  args = parser.parse_args(argv[1:])

  # prepare to ingest the data set
  def preprocess_for_eval(feature, label):
    return preprocess_video(
      feature, label, args.num_classes, args.clip_length, args.frame_height,
      args.frame_width, args.channels)

  def preprocess_for_train(feature, label):
    return preprocess_video(
      feature, label, args.num_classes, args.clip_length, args.frame_height,
      args.frame_width, args.channels, is_training=True)

  def get_train_dataset():
    dataset = tf.data.Dataset.list_files(
      path.join(args.train_subset_dir_path, '*.tfrecord'))
    dataset = tf.data.TFRecordDataset(dataset,
                                      buffer_size=args.tfrecord_size * 2 ** 20,
                                      num_parallel_reads=cpu_count())
    dataset = dataset.map(parse_serialized_example,
                          num_parallel_calls=cpu_count())
    dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(
      buffer_size=args.batch_size))
    dataset = dataset.apply(tf.data.experimental.map_and_batch(
      map_func=preprocess_for_train, batch_size=args.batch_size,
      num_parallel_calls=cpu_count()))
    dataset = dataset.prefetch(args.prefetch_size)
    return dataset

  def get_eval_dataset():
    dataset = tf.data.Dataset.list_files(
      path.join(args.eval_subset_dir_path, '*.tfrecord'))
    dataset = tf.data.TFRecordDataset(dataset,
                                      buffer_size=args.tfrecord_size * 2 ** 20,
                                      num_parallel_reads=cpu_count())
    dataset = dataset.map(parse_serialized_example,
                          num_parallel_calls=cpu_count())
    dataset = dataset.apply(tf.data.experimental.map_and_batch(
      map_func=preprocess_for_eval, batch_size=args.batch_size,
      num_parallel_calls=cpu_count()))
    dataset = dataset.prefetch(args.prefetch_size)
    return dataset

  def get_predict_dataset():
    dataset = tf.data.Dataset.list_files(
      path.join(args.predict_subset_dir_path, '*.tfrecord'))
    dataset = tf.data.TFRecordDataset(dataset,
                                      buffer_size=args.tfrecord_size * 2 ** 20,
                                      num_parallel_reads=cpu_count())
    dataset = dataset.map(parse_serialized_example,
                          num_parallel_calls=cpu_count())
    dataset = dataset.apply(tf.data.experimental.map_and_batch(
      map_func=preprocess_for_eval, batch_size=args.batch_size,
      num_parallel_calls=cpu_count()))
    dataset = dataset.prefetch(args.prefetch_size)
    return dataset

  # prepare to use zero or more GPUs
  if args.num_gpus == 1:
    gpu_options = tf.GPUOptions(
      allow_growth=True, per_process_gpu_memory_fraction=.9)
    session_config = tf.ConfigProto(
      allow_soft_placement=True, gpu_options=gpu_options)
    distribute_strategy = None
    putenv('CUDA_VISIBLE_DEVICES', '{}'.format(args.gpu_num))
  elif args.num_gpus > 1:
    gpu_options = tf.GPUOptions(
      allow_growth=True, per_process_gpu_memory_fraction=.9)
    session_config = tf.ConfigProto(
      allow_soft_placement=True, gpu_options=gpu_options)

    # we prefer ParameterServerStrategy, but use MirroredStrategy when warm
    # starting because ParameterServerStrategy is broken in that use case
    if args.warm_start:
      # virtual gpu names are independent of device name
      devices = ['/gpu:{}'.format(i) for i in range(args.num_gpus)]
      # MirroredStrategy dependency NCCL not implemented on Windows
      distribute_strategy = tf.distribute.MirroredStrategy(devices=devices)
    else:
      distribute_strategy = tf.contrib.distribute.ParameterServerStrategy(
        num_gpus_per_worker=args.num_gpus)
    # TODO: parameterize list of CUDA_VISIBLE_DEVICE numbers
    device_names = ''
    for i in range(args.num_gpus - 1):
      device_names += '{},'.format(i)
    device_names += '{}'.format(args.num_gpus - 1)
    putenv('CUDA_VISIBLE_DEVICES', device_names)
  else:  # just use the CPU
    session_config = None
    distribute_strategy = None
    putenv('CUDA_VISIBLE_DEVICES', '')

  # since training halts for validation to be performed, assign all available
  # resources to evaluation (e.g. use the same training distribution strategy)
  estimator_config = tf.estimator.RunConfig(
    model_dir=args.model_dir,
    save_summary_steps=args.monitor_steps,
    save_checkpoints_steps=args.monitor_steps,
    session_config=session_config,
    keep_checkpoint_max=10000,
    keep_checkpoint_every_n_hours=10000,
    log_step_count_steps=args.monitor_steps,
    train_distribute=distribute_strategy,
    eval_distribute=distribute_strategy)

  if args.warm_start:
    # prepare to restore weights from an existing checkpoint
    try:
      variables_to_warm_start = s3dg_vars[args.variables_to_warm_start]
    except KeyError:
      variables_to_warm_start = '.*'

    initialization_ckpt = args.checkpoint_path if args.checkpoint_path \
      else args.model_dir

    warm_start_settings = tf.estimator.WarmStartSettings(
      ckpt_to_initialize_from=initialization_ckpt,
      vars_to_warm_start=variables_to_warm_start)
  else:
    warm_start_settings = None

  try:
    variables_to_train = s3dg_vars[args.variables_to_train]
  except KeyError:
    variables_to_train = None

  try:
    weights = metric_weights[args.metric_weights]
  except KeyError:
    weights = None

  # create the model
  classifier = tf.estimator.Estimator(
    model_fn=s3dg_fn,
    params={
      'num_classes': args.num_classes,
      'learning_rate': args.learning_rate,
      'optimizer': args.optimizer,
      'momentum': args.momentum,
      'dropout_rate': args.dropout_rate,
      'variables_to_train': variables_to_train,
      'weight_decay': args.weight_decay,
      'min_depth': args.min_depth,
      'depth_multiplier': args.depth_multiplier,
      'add_image_summaries': args.add_image_summaries,
      'clip_length': args.clip_length,
      'batch_size': args.batch_size,
      'metric_weights': weights
    },
    config=estimator_config,
    warm_start_from=warm_start_settings)

  if args.mode == 'train_and_eval':
    # train and evaluate the model.
    tf.estimator.train_and_evaluate(
      estimator=classifier,
      train_spec=tf.estimator.TrainSpec(
        input_fn=get_train_dataset,
        max_steps=args.train_steps),
      eval_spec=tf.estimator.EvalSpec(
        input_fn=get_eval_dataset,
        steps=None,
        start_delay_secs=0,
        throttle_secs=0))
  elif args.mode == 'train':
    # train the model.
    classifier.train(input_fn=get_train_dataset, steps=args.train_steps)
  elif args.mode == 'eval':
    # evaluate the model.
    eval_result = classifier.evaluate(input_fn=get_eval_dataset)
    tf.logging.info(
      '\nEvaluation set metrics:\n\tauc: {Metrics/eval/auc:0.3f}\n\tprecision: '
      '{Metrics/eval/precision:0.3f}\n\trecall: {Metrics/eval/recall:0.3f}'
      '\n\tf1: {Metrics/eval/f1:0.3f}\n'.format(**eval_result))
  elif args.mode == 'predict':
    # Generate predictions from the model
    predictions = classifier.predict(input_fn=get_predict_dataset)

    labels = []

    with tf.Session().as_default() as sess:
      dataset = get_predict_dataset()
      iterator = dataset.make_one_shot_iterator()
      get_next = iterator.get_next()

      while True:
        try:
          # sess.run outputs numpy arrays, so we use numpy downstream
          feature, label = sess.run(get_next)
          labels.append(label)
        except tf.errors.OutOfRangeError:
          break

    for count, pred_dict, truth in zip(range(len(labels)), predictions, labels):
      truth = np.squeeze(truth)
      probabilities = pred_dict['probabilities']
      abs_error = np.abs(np.subtract(truth, probabilities))
      classifications = np.round(probabilities)

      if args.metric_weights:
        lower_bound, upper_bound = weight_bounds[args.metric_weights]
        abs_error = abs_error[lower_bound:upper_bound]
        classifications = classifications[lower_bound:upper_bound]
        truth = truth[lower_bound:upper_bound]

      num_not_equal = np.sum(np.not_equal(classifications, truth))

      tf.logging.info('{}: num_not_equal: {}, abs_error: {}'.format(
        count, num_not_equal, np.round(abs_error, 3)))

  elif args.mode == 'export':
    def serving_input_receiver_fn():
      """An input receiver that expects a serialized tf.Example."""
      serialized_tf_example = tf.placeholder(dtype=tf.string, shape=[])

      parsed_features = tf.parse_single_example(
        serialized_tf_example, feature_spec)

      parsed_features['feature'], parsed_features['label'] = preprocess_for_eval(
        parsed_features['feature'], parsed_features['label'])

      return tf.estimator.export.TensorServingInputReceiver(
        features=tf.expand_dims(parsed_features['feature'], 0),
        receiver_tensors=serialized_tf_example)

    classifier.export_saved_model(
      args.export_path, serving_input_receiver_fn=serving_input_receiver_fn,
      checkpoint_path=args.checkpoint_path)
  else:
    raise ValueError(
      '--mode parameter requires specification using an argument from the set'
        ' {\'train\', \'eval\', \'predict\'}')

parser = argparse.ArgumentParser()

parser.add_argument('--mode', required=True,
                    help='train_and_eval, train, eval, predict or export')
parser.add_argument('--batch_size', default=6, type=int)
parser.add_argument('--prefetch_size', default=tf.data.experimental.AUTOTUNE,
                    type=int)
parser.add_argument('--monitor_steps', default=100, type=int)
parser.add_argument('--train_steps', default=None, type=int)
parser.add_argument('--num_classes', default=204, type=int)
parser.add_argument('--min_depth', default=16, type=int)
parser.add_argument('--depth_multiplier', default=1., type=float)
parser.add_argument('--learning_rate', default=1e-1, type=float)
parser.add_argument('--optimizer', default='momentum', help='sgd or momentum')
parser.add_argument('--momentum', default=.9, type=float)
parser.add_argument('--weight_decay', default=1e-7, type=float)
parser.add_argument('--dropout_rate', default=.2, type=float)
parser.add_argument('--gpu_num', default=0, type=int)
parser.add_argument('--num_gpus', default=1, type=int)
parser.add_argument('--clip_length', default=64, type=int)
parser.add_argument('--frame_height', default=s3dg.default_image_size, type=int)
parser.add_argument('--frame_width', default=s3dg.default_image_size, type=int)
parser.add_argument('--channels', default=3, type=int)
parser.add_argument('--warm_start', action='store_true')
parser.add_argument('--variables_to_warm_start', default=None)
parser.add_argument('--variables_to_train', default=None)
parser.add_argument('--add_image_summaries', type=bool, default=True)
parser.add_argument('--train_subset_dir_path', default=None)
parser.add_argument('--eval_subset_dir_path', default=None)
parser.add_argument('--predict_subset_dir_path', default=None)
parser.add_argument('--tfrecord_size', default=40, type=int,
                    help='approximate size of the given data set\'s tfrecords '
                         'in bytes')
parser.add_argument('--checkpoint_path',default=None)
parser.add_argument('--export_path',default=None)
parser.add_argument('--model_dir', required=True)
parser.add_argument('--metric_weights',default=None,
                    help="One of {gate, veh, trn, ped, cyc}")


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
