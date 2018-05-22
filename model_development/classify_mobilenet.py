import os
import signal
import sys
import tensorflow as tf
import time

from nets import inception
from preprocessing import inception_preprocessing

tf.app.flags.DEFINE_string(
    'model_path', None, 'An absolute path to a protobuf model file.')

tf.app.flags.DEFINE_string(
    'labels_path', None, 'An absolute path to a protobuf model file.')

tf.app.flags.DEFINE_string(
    'io_tensor_names_path', None, 'An absolute path to a protobuf model file.')

tf.app.flags.DEFINE_string(
    'image_dir', None, 'An absolute path to a protobuf model file.')

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of images analyzed concurrently.')

tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 0.9,
    'The ratio of total memory across all available GPUs to use with this process. '
    'Defaults to a suggested max of 0.9.')

tf.app.flags.DEFINE_integer(
    'gpu_device_num', 0,
    'The device number of a single GPU to use for evaluation on a multi-GPU system. '
    'Defaults to zero.')

tf.app.flags.DEFINE_boolean(
    'cpu_only', False,
    'Explicitly assign all evaluation ops to the CPU on a GPU-enabled system. '
    'Defaults to False.')

FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

path = os.path


def read_meta_file(file_path):
    meta_lines = [line.rstrip().split(':') for line in tf.gfile.GFile(file_path).readlines()]
    return {line[0]: line[1] for line in meta_lines}


def read_tensor_names(io_tensor_names_path):
    meta_map = read_meta_file(io_tensor_names_path)
    return {key: value + ':0' for key, value in meta_map.items()}


def read_labels(labels_path):
    meta_map = read_meta_file(labels_path)
    return {int(key): value for key, value in meta_map.items()}


def read_graph(model_path):
    """Creates a graph from saved GraphDef file and returns a saver."""
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def decode_and_preprocess_for_eval(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image_size = inception.inception_v3.default_image_size
  return inception_preprocessing.preprocess_for_eval(
    image, image_size, image_size, central_fraction=None)


def interrupt_handler(signal_number, _):
    tf.logging.info(
        'Received interrupt signal (%d). Unsetting CUDA_VISIBLE_DEVICES environment variable.', signal_number)
    os.unsetenv('CUDA_VISIBLE_DEVICES')
    sys.exit(0)

if not FLAGS.model_path:
    raise ValueError('You must supply the model path with --model_path')

if not FLAGS.image_dir:
    raise ValueError('You must supply the image directory with --image_dir')

if FLAGS.io_tensor_names_path:
    io_tensor_names_path = FLAGS.io_tensor_names_path
else:
    io_tensor_names_path = path.join(path.dirname(FLAGS.model_path), 'io_tensor_names.txt')

if not tf.gfile.Exists(io_tensor_names_path):
    raise ValueError('checkpoint_path must contain a text file named checkpoint')

if FLAGS.cpu_only:
    device_name = '/cpu:0'

    session_config = None

    tf.logging.info('Setting CUDA_VISIBLE_DEVICES environment variable to None.')
    os.putenv('CUDA_VISIBLE_DEVICES', '')
else:
    device_name = '/gpu:' + str(FLAGS.gpu_device_num)

    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=float(FLAGS.gpu_memory_fraction))
    session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    tf.logging.info('Setting CUDA_VISIBLE_DEVICES environment variable to %d.', FLAGS.gpu_device_num)
    os.putenv('CUDA_VISIBLE_DEVICES', str(FLAGS.gpu_device_num))

signal.signal(signal.SIGINT, interrupt_handler)

image_names = sorted(os.listdir(FLAGS.image_dir))
image_paths = [path.join(FLAGS.image_dir, image_name) for image_name in image_names]

with tf.Graph().as_default(), tf.device(device_name):
    with tf.Session(config=session_config) as sess:
        read_graph(FLAGS.model_path)
        tensor_names_map = read_tensor_names(io_tensor_names_path)
        softmax = sess.graph.get_tensor_by_name(tensor_names_map['output_tensor_name'])

        # Loading the injected placeholder
        input_placeholder = sess.graph.get_tensor_by_name(tensor_names_map['input_tensor_name'])

        print('Starting evaluation at ' + time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime()))
        start_time = time.time()
        prediction_array = np.ndarray(dtype=np.float32, shape=(num_frames, num_labels))

        image_list_placeholder = tf.placeholder(dtype=tf.string)

        images = tf.data.Dataset.from_tensor_slices(image_list_placeholder)
        images = images.map(decode_and_preprocess_for_eval, batch_size)
        image_batches = images.batch(batch_size)

        batch_iterator = image_batches.make_initializable_iterator()
        next_batch = batch_iterator.get_next()

        tf_session.run(batch_iterator.initializer, feed_dict={image_list_placeholder: image_list})

        for batch_num in range(num_batches):
            try:
                image_batch = next_batch.eval(session=tf_session)

                lower_index = batch_num * batch_size
                upper_index = min((batch_num + 1) * batch_size, num_frames)

                prediction_array[lower_index:upper_index] = tf_session.run(
                    placeholder_map['softmax_tensor'],
                    {placeholder_map['image_batch_placeholder']: image_batch,
                     placeholder_map['image_size_placeholder']: image_size})

                draw_progress_bar(percentage(upper_index, num_frames) / 100, 40)
            except tf.errors.OutOfRangeError:
                break
        for image_path in image_paths:
            # Open specified url and load image as a string
            image_string = tf.gfile.FastGFile(image_path, 'rb').read()
            probabilities = sess.run(softmax, {input_placeholder: image_string})
            # print(np.array_str(probabilities, precision=2, suppress_small=True) + ': ' + path.basename(image_path))

        end_time = time.time()
        print('Finished evaluation at ' + time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime()))
        print('Evaluation elapsed ' + str(end_time - start_time) + ' seconds')

os.unsetenv('CUDA_VISIBLE_DEVICES')