import tensorflow as tf
import sys


def create_graph(model_file):
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(model_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

model_file = sys.argv[1]

path = sys.argv[2]

# Open specified url and load image as a string
image_string = tf.gfile.FastGFile(path, 'rb').read()

with tf.Graph().as_default():
    with tf.Session() as new_sess:
        create_graph(model_file)

        softmax = new_sess.graph.get_tensor_by_name("DensenetBC/Predictions/Reshape_1:0")

        # Loading the injected placeholder
        input_placeholder = new_sess.graph.get_tensor_by_name("input_image:0")

        probabilities = new_sess.run(softmax, {input_placeholder: image_string})
        print(probabilities)