from nets import nets_factory
from preprocessing import preprocessing_factory
import tensorflow as tf

tf.app.flags.DEFINE_string(
  'image_file', None, 'The path to the image to be analyzed.')

tf.app.flags.DEFINE_string(
  'model_file', None, 'The path to the model protobuf file.')

tf.app.flags.DEFINE_string(
  'model_name', None, 'The name of the model for use in image preprocessing.')

tf.app.flags.DEFINE_string(
  'input_node_name', 'images',
  'Name of the tensor through which images are input to the network')

tf.app.flags.DEFINE_string(
  'output_node_name', None,
  'Name of the tensor through which images are input to the network')

FLAGS = tf.app.flags.FLAGS


def create_graph(model_file):
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(model_file, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def main():
  with tf.Graph().as_default():
    with tf.Session() as session:
      create_graph(FLAGS.model_file)

      image_size = nets_factory.get_network_default_image_size(FLAGS.model_name)

      preprocessing_fn = preprocessing_factory.get_preprocessing(FLAGS.model_name)

      def map_fn(image):
        return preprocessing_fn(image, image_size, image_size, central_fraction=None)

      image_string = tf.gfile.FastGFile(FLAGS.image_file, 'rb').read()

      image_tensor = tf.image.decode_jpeg(image_string)
      image_tensor = tf.image.convert_image_dtype(image_tensor, dtype=tf.float32)

      image_dataset = tf.data.Dataset.from_tensors(image_tensor)
      image_dataset = image_dataset.map(map_fn)
      image_dataset = image_dataset.batch(1)

      image_dataset_iterator = image_dataset.make_one_shot_iterator()
      image = session.run(image_dataset_iterator.get_next())

      softmax = session.graph.get_tensor_by_name(FLAGS.output_node_name + ":0")

      image_placeholder = session.graph.get_tensor_by_name(FLAGS.input_node_name + ":0")

      probabilities = session.run(softmax, {image_placeholder: image})

      print(probabilities)


if __name__ == '__main__':
  main()
