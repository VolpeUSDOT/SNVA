import os
import signal
import sys
import tensorflow as tf
import time

path = os.path

tf.app.flags.DEFINE_string(
    'model_path', '/tmp/tfmodel/', 'An absolute path to a protobuf model file.')

tf.app.flags.DEFINE_string(
    'image_dir', '/tmp/tfmodel/', 'An absolute path to a protobuf model file.')

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


def create_graph(model_path):
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def interrupt_handler(signal_number, _):
    tf.logging.info(
        'Received interrupt signal (%d). Unsetting CUDA_VISIBLE_DEVICES environment variable.', signal_number)
    os.unsetenv('CUDA_VISIBLE_DEVICES')
    sys.exit(0)

if not FLAGS.model_path:
    raise ValueError('You must supply the model path with --model_path')

if not FLAGS.image_dir:
    raise ValueError('You must supply the image directory with --image_dir')

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
        create_graph(FLAGS.model_path)

        softmax = sess.graph.get_tensor_by_name("MobilenetV1/Predictions/Reshape_1:0")

        # Loading the injected placeholder
        input_placeholder = sess.graph.get_tensor_by_name("input_image:0")

        print('Starting evaluation at ' + time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime()))
        start_time = time.time()

        for image_path in image_paths:
            # Open specified url and load image as a string
            image_string = tf.gfile.FastGFile(image_path, 'rb').read()
            probabilities = sess.run(softmax, {input_placeholder: image_string})
            # print(np.array_str(probabilities, precision=2, suppress_small=True) + ': ' + path.basename(image_path))

        end_time = time.time()
        print('Finished evaluation at ' + time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime()))
        print('Evaluation elapsed ' + str(end_time - start_time) + ' seconds')

os.unsetenv('CUDA_VISIBLE_DEVICES')