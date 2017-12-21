
import argparse
import numpy as np
import os
from os import path
import random
import shutil
from skimage import io
import time
import tensorflow as tf
from tensorflow.contrib import keras

keras_img = keras.preprocessing.image

DATA_DIR_PATH = path.join(os.getenv('SNVA_SHARE_PATH'), 'Data/fhwa/fhwa_nds')
DATASET_DIR_PATH = path.join(os.getenv('SNVA_SHARE_PATH'), 'Datasets/fhwa/fhwa_nds')


class SceneDetectionDatasetCreator(object):
    def __init__(self,
                 class_dir_name_list,
                 # nolabel_dir_name,
                 create_standard_subsets,
                 create_eval_subset,
                 data_dir_path=DATA_DIR_PATH,
                 dataset_dir_path=DATASET_DIR_PATH,
                 random_seed=None):
        self._class_dir_names = class_dir_name_list
        # self._nolabel_dir_name = nolabel_dir_name
        self._data_dir_path = data_dir_path
        self._dataset_dest_path = dataset_dir_path
        self._n_training_frames = 0
        self._n_dev_frames = 0
        self._n_test_frames = 0
        self._n_eval_frames = 0
        self._class_dir_paths = {}
        self._random_seed = random_seed
        self._create_eval_subset = create_eval_subset
        self._create_standard_subsets = create_standard_subsets

    def _create_dataset_paths(self):
        # create training, dev, and test sub-directories of _dataset_dest_path
        # each will contain one sub-folder per class
        subset_dir_names = []

        if self._create_standard_subsets:
            subset_dir_names.extend(['training', 'dev', 'test'])

        if self._create_eval_subset:
            subset_dir_names.append('eval')

        if not path.exists(self._dataset_dest_path):
            os.mkdir(self._dataset_dest_path)

        for subset_dir_name in subset_dir_names:
            subset_dir_path = path.join(self._dataset_dest_path, subset_dir_name)

            if not path.exists(subset_dir_path):
                os.mkdir(subset_dir_path)

            for class_dir_name in self._class_dir_names:
                class_dir_path = path.join(subset_dir_path, class_dir_name)

                if not path.exists(class_dir_path):
                    os.mkdir(class_dir_path)

                self._class_dir_paths[subset_dir_name + '_' + class_dir_name] = class_dir_path

    def create_subsets(self, training_percent=0.7, dev_percent=0.2):
        '''Creates one destination folder for each class-subset pair (e.g. training_class_0_dir or
        dev_class_1_dir). For each subfolder (containing the frames of
        a single video) of the datasource_dir (containing many subfolders for many videos), randomly samples training_percent %, dev_percent % and test_percent % of
        subfolder contents and then moves all training, dev, and test sample frames into training, dev,
        and test folders, respectively. This method fits into the pipeline between frame extraction and tfrecord creation,
        with dataset standardization to eventually be placed between this function and tfrecord creation when implemented'''

        if (training_percent + dev_percent) > (0.9 + 1e-7):
                raise ValueError('At least 10% of data should be reserved for the test set')

        if len(self._class_dir_paths) == 0:
            self._create_dataset_paths()

        if self._create_eval_subset:
            self._n_eval_frames = 0

        if self._create_standard_subsets:
            self._n_training_frames = 0
            self._n_dev_frames = 0
            self._n_test_frames = 0

        random.seed(self._random_seed)

        # for each folder of frames in the data dir
        for video_frame_dir in os.listdir(self._data_dir_path):
            video_frame_dir_path = path.join(self._data_dir_path, video_frame_dir)

            if path.isdir(video_frame_dir_path):
                for class_dir_name in self._class_dir_names:
                    class_sub_dir_path = path.join(video_frame_dir_path, class_dir_name)
                    frame_list = os.listdir(class_sub_dir_path)
                    frame_list.sort()  # in case we want to reproduce results using a given random seed
                    n_frames = len(frame_list)

                    if self._create_standard_subsets:
                        n_training_frames = int(training_percent * n_frames)
                        n_dev_frames = int(dev_percent * n_frames)
                        n_test_frames = n_frames - n_training_frames - n_dev_frames

                    # accumulate the total number of training and development frames for use in standardization
                    if self._create_eval_subset:
                        self._n_eval_frames += n_frames

                    if self._create_standard_subsets:
                        self._n_training_frames += n_training_frames
                        self._n_dev_frames += n_dev_frames
                        self._n_test_frames += n_test_frames

                    frame_indices = [_ for _ in range(n_frames)]

                    if self._create_eval_subset:
                        # create the eval subset before shuffling frames so that
                        # eval probabilities can be viewed in sequential order.
                        for index in frame_indices:
                            dest_video_frame_path = path.join(self._class_dir_paths['eval_' + class_dir_name],
                                                              frame_list[index])
                            if not path.exists(dest_video_frame_path):
                                source_video_frame_path = path.join(class_sub_dir_path, frame_list[index])
                                shutil.copy(source_video_frame_path, dest_video_frame_path)

                    if self._create_standard_subsets:
                        random.shuffle(frame_indices)

                        training_frame_indices = frame_indices[:n_training_frames]

                        for index in training_frame_indices:
                            dest_video_frame_path = path.join(self._class_dir_paths['training_' + class_dir_name],
                                                              frame_list[index])
                            if not path.exists(dest_video_frame_path):
                                source_video_frame_path = path.join(class_sub_dir_path, frame_list[index])
                                shutil.copy(source_video_frame_path, dest_video_frame_path)

                        dev_frame_indices = frame_indices[n_training_frames:n_training_frames + n_dev_frames]

                        for index in dev_frame_indices:
                            dest_video_frame_path = path.join(self._class_dir_paths['dev_' + class_dir_name],
                                                              frame_list[index])
                            if not path.exists(dest_video_frame_path):
                                source_video_frame_path = path.join(class_sub_dir_path, frame_list[index])
                                shutil.copy(source_video_frame_path, dest_video_frame_path)

                        test_frame_indices = frame_indices[n_training_frames + n_dev_frames:]

                        for index in test_frame_indices:
                            dest_video_frame_path = path.join(self._class_dir_paths['test_' + class_dir_name],
                                                              frame_list[index])
                            if not path.exists(dest_video_frame_path):
                                source_video_frame_path = path.join(class_sub_dir_path, frame_list[index])
                                shutil.copy(source_video_frame_path, dest_video_frame_path)

    def standardize_samples(self):
        if len(self._class_dir_paths) == 0:
            self._create_dataset_paths()

        if self._n_training_frames == 0:
            n_training_frames = 0

            for class_dir_path in [value for key, value in self._class_dir_paths.items() if 'training' in key]:
                n_training_frames += len(os.listdir(class_dir_path))
            
            if n_training_frames == 0:
                raise ValueError('The number of training samples is zero, which implies that create_subsets() has not '
                                 'been executed. The dataset must be populated with images in order to apply '
                                 'standardization.')

            self._n_training_frames = n_training_frames

        if self._n_dev_frames == 0:
            n_dev_frames = 0

            for class_dir_path in [value for key, value in self._class_dir_paths.items() if 'dev' in key]:
                n_dev_frames += len(os.listdir(class_dir_path))

            if n_dev_frames == 0:
                raise ValueError('The number of validation samples is zero, which implies that create_subsets() has not '
                                 'been executed. The dataset must be populated with images in order to apply '
                                 'standardization.')

            self._n_dev_frames = n_dev_frames

        if self._n_test_frames == 0:
            n_test_frames = 0

            for class_dir_path in [value for key, value in self._class_dir_paths.items() if 'test' in key]:
                n_test_frames += len(os.listdir(class_dir_path))

            if n_test_frames == 0:
                raise ValueError('The number of testing samples is zero, which implies that create_subsets() has not '
                                 'been executed. The dataset must be populated with images in order to apply '
                                 'standardization.')

            self._n_test_frames = n_test_frames

        start_time = time.time()
        image_path_dictionary = {}

        index = 0
        # TODO: parameterize or derive image dimensions
        #frame_array = np.empty(shape=(self._n_training_frames + self._n_dev_frames, 480, 640, 3), dtype=np.int8)
        frame_array = np.empty(shape=(self._n_test_frames, 480, 640, 3), dtype=np.uint8)

        print('creating image array with shape = ' + str(frame_array.shape))

        # for image_dir in [value for key, value in self._class_dir_paths.items() if 'training' in key or 'dev' in key]:
        for image_dir in [value for key, value in self._class_dir_paths.items() if 'test' in key]:
            image_paths = [os.path.join(image_dir, image) for image in os.listdir(image_dir)]

            for image_path in image_paths:
                image = keras_img.load_img(image_path)
                frame_array[index] = keras_img.img_to_array(image)
                image_path_dictionary[image_path] = index
                index += 1

            print(str(len(image_path_dictionary)) + " images read from " + image_dir)

        data_generator = keras_img.ImageDataGenerator(featurewise_center=True,
                                                      featurewise_std_normalization=True,
                                                      zca_whitening=False)

        data_generator.fit(frame_array, seed=self._random_seed)

        frame_array = data_generator.standardize(np.asarray(frame_array, dtype=np.float32))
        # save_to_dir = path.join(self._dataset_dest_path, 'stdz_tmp')
        # if not path.exists(save_to_dir):
        #     os.mkdir(save_to_dir)
        # np_iter = data_generator.flow(frame_array, seed=self._random_seed, batch_size=1024, shuffle=False, save_to_dir=save_to_dir)
        # while np_iter.next() is not None:
        #     print(str(data_generator.mean))
        #     print(str(data_generator.std))
        # for image_dir in [value for key, value in self._class_dir_paths.items() if 'training' in key or 'dev' in key]:
        for image_path, index in image_path_dictionary.items():
            # io.imsave(image_path, frame_array[index])
            image = keras_img.array_to_img(frame_array[index])
            image.save(image_path)

        print('preprocessing completed in ' + str(time.time() - start_time) + ' seconds')

parser = argparse.ArgumentParser()

parser.add_argument('--positive_class_name', '-p', required=True)
parser.add_argument('--negative_class_name', '-n', required=True)
# parser.add_argument('--nolabel_dir_name', '-nl', required=True)
parser.add_argument('--data_dir_path', '-d', required=True)
parser.add_argument('--dataset_dir_path', '-ds', required=True)
parser.add_argument('--random_seed', '-r', type=int, default=1)
parser.add_argument('--create_standard_subsets', '-css', dest='create_standard_subsets', action='store_true',
                    help='Create training, test and dev subsets')
parser.add_argument('--create_eval_subset', '-ces', dest='create_eval_subset', action='store_true',
                    help='Create a subset for evaluation that contains all samples. '
                         'Use to evaluate models trained on a completely disjoint data set.')
parser.add_argument('--no_standard_subsets', '-nds', dest='create_standard_subsets', action='store_false',
                    help='Don\'t create training, test and dev subsets')
parser.add_argument('--no_eval_subset', '-nes', dest='create_eval_subset', action='store_false',
                    help='Don\'t create a subset for evaluation that contains all samples. '
                         'Use to evaluate models trained on a completely disjoint dataset.')

parser.set_defaults(create_standard_subsets=True, create_eval_subset=False)
# parser.add_argument('--frame_width', '-x')
# parser.add_argument('--frame_height', '-y')

args = parser.parse_args()

creator = SceneDetectionDatasetCreator([args.positive_class_name,
                                       args.negative_class_name],
                                       # args.nolabel_dir_name,
                                       args.create_standard_subsets,
                                       args.create_eval_subset,
                                       args.data_dir_path,
                                       args.dataset_dir_path,
                                       args.random_seed)

creator.create_subsets()
# creator.standardize_samples()