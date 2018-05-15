import os
import sys

import tensorflow as tf

from model_development.nets import inception_v4
from preprocessing import inception_preprocessing
from model_development.nets import inception_resnet_v2
from model_development.nets import inception_v3

modelType = sys.argv[1]
datasetName = sys.argv[2]
modelPath = sys.argv[3]
labelsPath = sys.argv[4]

# define model types
models = []
models.append({'type': "inception_v3",
               'path': modelPath,
               'labels': labelsPath})

models.append({'type': "inception_v4",
               'path': modelPath,
               'labels': labelsPath})

models.append({'type': "inception_resnet_v2",
               'path': modelPath,
               'labels': labelsPath})

fileTarget = open('../test-results/' + datasetName + '-test-results.txt', 'w')

relevant_path = '../testing/' + datasetName
included_extensions = ['jpg', 'jpeg']
file_names = [fn for fn in os.listdir(relevant_path)
              if any(fn.endswith(ext) for ext in included_extensions)]

# load the TF-SLIM framework
slim = tf.contrib.slim

# default batch size is 3 and inception only really supports images that are 300 pixel in width
batch_size = 3
image_size = 299


def load_labels(path):
    # load the labels and remove stuff we don't want to display
    file_data = [line.split() for line
                 in tf.gfile.GFile(path)]
    return [item[0].split(":") for item in file_data]


def load_images(filenames):
    # load images into a tensor
    print('Loading Images...')
    processed_images = []
    for file in filenames:
        print(relevant_path + '/' + file)
        testImage_string = tf.gfile.FastGFile(relevant_path + '/' + file, 'rb').read()
        testImage = tf.image.decode_jpeg(testImage_string, channels=3)
        processed_image = inception_preprocessing.preprocess_image(testImage, image_size, image_size, is_training=False)
        processed_images.append(processed_image)

    return processed_images


def process_inceptionV3(image_tensor, model_path, filenames, labels):
    # accept image tensor of any size and return scores

    results = []

    logits, _ = inception_v3.inception_v3(image_tensor, num_classes=2, is_training=False)
    probabilities = tf.nn.softmax(logits)
    checkpoint_path = tf.train.latest_checkpoint(model_path)
    init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_model_variables('InceptionV3'))

    with tf.Session() as sess:
        init_fn(sess)

        np_image, probabilities = sess.run([image_tensor, probabilities])
        count = 0
        for item_prob in probabilities:
            sorted_inds = [i[0] for i in sorted(enumerate(-item_prob), key=lambda x: x[1])]

            # setup labels for each of the items
            names = []
            unsorted_inds = []
            labelcount = 0
            for label in labels:
                names.append(label[1])
                unsorted_inds.append(int(label[0]))
                labelcount = labelcount + 1

            score = []
            for i in range(labelcount):
                index = sorted_inds[i]
                score.append(names[index] + '(' + str(round(item_prob[index] * 100, 2)) + '%)' + '\t')

            results_line = []
            for i in range(labelcount):
                index = unsorted_inds[i]
                results_line.append({names[index]: item_prob[index], 'filename': filenames[count]})

            newline = '%25s     %25s    %25s' % (filenames[count], score[0], score[1])
            fileTarget.write(newline)
            fileTarget.write('\n')
            count = count + 1
            results.append(results_line)

    return results


def runInceptionV3(filenames):
    with tf.Graph().as_default():
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            processed_images = load_images(filenames)
            results = process_inceptionV3(processed_images, models[0]["path"], filenames,
                                          load_labels(models[0]["labels"]))
            for result_item in results:
                print(result_item)


def process_inceptionV4(image_tensor, model_path, filenames, labels):
    # accept image tensor of any size and return scores

    results = []

    logits, _ = inception_v4.inception_v4(image_tensor, num_classes=2, is_training=False)
    probabilities = tf.nn.softmax(logits)
    checkpoint_path = tf.train.latest_checkpoint(model_path)
    init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_model_variables('InceptionV4'))

    with tf.Session() as sess:
        init_fn(sess)

        np_image, probabilities = sess.run([image_tensor, probabilities])
        count = 0
        for item_prob in probabilities:
            sorted_inds = [i[0] for i in sorted(enumerate(-item_prob), key=lambda x: x[1])]

            # setup labels for each of the items
            names = []
            unsorted_inds = []
            labelcount = 0
            for label in labels:
                names.append(label[1])
                unsorted_inds.append(int(label[0]))
                labelcount = labelcount + 1

            score = []
            for i in range(labelcount):
                index = sorted_inds[i]
                score.append(names[index] + '(' + str(round(item_prob[index] * 100, 2)) + '%)' + '\t')

            results_line = []
            for i in range(labelcount):
                index = unsorted_inds[i]
                results_line.append({names[index]: item_prob[index], 'filename': filenames[count]})

            newline = '%25s     %25s    %25s' % (filenames[count], score[0], score[1])
            fileTarget.write(newline)
            fileTarget.write('\n')
            count = count + 1
            results.append(results_line)

    return results


def runInceptionV4(filenames):
    with tf.Graph().as_default():
        with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
            processed_images = load_images(filenames)
            results = process_inceptionV4(processed_images, models[1]["path"], filenames,
                                          load_labels(models[1]["labels"]))
            for result_item in results:
                print(result_item)


def process_inceptionResnetV2(image_tensor, model_path, filenames, labels):
    # accept image tensor of any size and return scores

    results = []
    logits, _ = inception_resnet_v2.inception_resnet_v2(image_tensor, num_classes=2, is_training=False)
    probabilities = tf.nn.softmax(logits)
    checkpoint_path = tf.train.latest_checkpoint(model_path)
    init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_model_variables('InceptionResnetV2'))

    with tf.Session() as sess:
        init_fn(sess)

        np_image, probabilities = sess.run([image_tensor, probabilities])
        count = 0
        for item_prob in probabilities:
            sorted_inds = [i[0] for i in sorted(enumerate(-item_prob), key=lambda x: x[1])]

            # setup labels for each of the items
            names = []
            unsorted_inds = []
            labelcount = 0
            for label in labels:
                names.append(label[1])
                unsorted_inds.append(int(label[0]))
                labelcount = labelcount + 1

            score = []
            for i in range(labelcount):
                index = sorted_inds[i]
                score.append(names[index] + '(' + str(round(item_prob[index] * 100, 2)) + '%)' + '\t')

            results_line = []
            for i in range(labelcount):
                index = unsorted_inds[i]
                results_line.append({names[index]: item_prob[index], 'filename': filenames[count]})

            newline = '%25s     %25s    %25s' % (filenames[count], score[0], score[1])
            fileTarget.write(newline)
            fileTarget.write('\n')
            count = count + 1
            results.append(results_line)

    return results


def runInceptionResnetV2(filenames):
    with tf.Graph().as_default():
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            processed_images = load_images(filenames)
            results = process_inceptionResnetV2(processed_images, models[2]["path"], filenames,
                                                load_labels(models[2]["labels"]))
            for result_item in results:
                print(result_item)


fileTarget.write('####################################################################################')
fileTarget.write("\n")
fileTarget.write('                            ' + modelType + ' Validation Tests                               ')
fileTarget.write("\n")
fileTarget.write('####################################################################################')
fileTarget.write("\n")

if (modelType == 'inception_resnet_v2'):
    runInceptionResnetV2(file_names)
if (modelType == 'inception_v4'):
    runInceptionV4(file_names)
if (modelType == 'inception_v3'):
    runInceptionV3(file_names)
fileTarget.write('\n')
