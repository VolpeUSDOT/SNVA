import tensorflow as tf
import sys
import os
import subprocess
import datetime
import platform
import argparse
from glob import iglob
from shutil import copy
from io import BytesIO
from PIL import Image
import timeit
import uuid
import numpy as np

# DONE: Modify detection to only process data after inference has completed.
# DONE: Update smoothing function to work as intended (currently broken)
# DONE: Modify model unpersist function to use loaded model name vs. static assignment.
# TODO: Add support for loading multiple models and performing predictions with all loaded models.
# TODO: Need to cycle through all detected labels and correctly output to report.
# TODO: Add support for basic HTML report which includes processed data & visualizations.

path = os.path
# set logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# set start time
start = timeit.default_timer()

parser = argparse.ArgumentParser(description='Process some video files using Tensorflow!')
parser.add_argument('--temppath', '-tp', dest='temppath', action='store', default='./vidtemp/',
                    help='Path to the directory where temporary files are stored.')
parser.add_argument('--trainingpath', '-rtp', dest='trainingpath', action='store', default='./retraining/',
                    help='Path to the directory where frames for retraining are stored.')
parser.add_argument('--reportpath', '-rp', dest='reportpath', action='store', default='results/',
                    help='Path to the directory where results are stored.')
parser.add_argument('--modelpath', '-mp', dest='modelpath', action='store', default='models/',
                    help='Path to the tensorflow protobuf model file.')
parser.add_argument('--labelpath', '-lp', dest='labelpath', action='store', default='models/default-labels.txt',
                    help='Path to the tensorflow model labels file.')
parser.add_argument('--smoothing', '-sm', dest='smoothing', action='store', default='0',
                    help='Apply a type of "smoothing" factor to detection results.')
parser.add_argument('--fps', '-fps', dest='fps', action='store', default='1',
                    help='Frames Per Second used to sample input video. '
                         'The higher this number the slower analysis will go. Default is 1 FPS')
parser.add_argument('--deinterlace', '-d', dest='deinterlace', action='store_true',
                    help='Apply de-interlacing to video frames during extraction.')
parser.add_argument('--outputclips', '-o', dest='outputclips', action='store_true',
                    help='Output results as video clips containing searched for labelname.')
parser.add_argument('--training', '-tr', dest='training', action='store_true',
                    help='Saves predicted frames for future model retraining.')
parser.add_argument('--outputpadding', '-op', dest='outputpadding', action='store', default='45',
                    help='Number of seconds added to the start and end of created video clips.')
parser.add_argument('--keeptemp', '-k', dest='keeptemp', action='store_true',
                    help='Keep ALL temporary extracted video frames.')
parser.add_argument('--videopath', '-v', dest='videopath', action='store', required=True, help='Path to video file(s).')

args = parser.parse_args()
currentSrcVideo = ''

if platform.system() == 'Windows':
    # path to ffmpeg bin
    FFMPEG_PATH = 'ffmpeg.exe'
else:
    # path to ffmpeg bin
    default_ffmpeg_path = '/usr/local/bin/ffmpeg'
    FFMPEG_PATH = default_ffmpeg_path if path.exists(default_ffmpeg_path) else '/usr/bin/ffmpeg'

# setup video temp directory for video frames
video_tempDir = args.temppath
if not os.path.isdir(video_tempDir):
    os.mkdir(video_tempDir)


def drawProgressBar(percent, barLen=20):
    sys.stdout.write("\r")
    progress = ""
    for i in range(barLen):
        if i < int(barLen * percent):
            progress += "="
        else:
            progress += " "
    sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
    sys.stdout.flush()


def percentage(part, whole):
    return 100 * (float(part) / float(whole))


def copy_files(src_glob, dst_folder):
    for fname in iglob(src_glob):
        newfilename = os.path.basename(fname)
        copy(fname, os.path.join(dst_folder, newfilename))


def remove_video_frames():
    for the_file in os.listdir(video_tempDir):
        file_path = os.path.join(video_tempDir, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def save_training_frames(framenumber, label):
    # copies frames/images to the passed directory for the purposes of retraining the model
    srcpath = os.path.join(args.temppath, '')
    dstpath = os.path.join(args.trainingpath + '/' + label, '')
    if not os.path.exists(dstpath):
        os.makedirs(dstpath)
    copy_files(srcpath + '*' + str(framenumber) + '.jpg', dstpath)


def decode_video(videopath):
    if args.deinterlace == True:
        deinterlace = 'yadif'
    else:
        deinterlace = ''
    video_filename, video_file_extension = path.splitext(path.basename(videopath))
    print(' ')
    print('Decoding video file ' + video_filename)
    video_temp = os.path.join(video_tempDir, str(video_filename) + '_%04d.jpg')
    command = [
        FFMPEG_PATH, '-i', videopath,
        '-vf', 'fps=' + args.fps, '-q:v', '1', '-vsync', 'vfr', video_temp, '-hide_banner', '-loglevel', '0',
        '-vf', deinterlace, '-vf', 'scale=640x480'
    ]
    subprocess.call(command)

    # Read in the image_data, but sort image paths first because os.listdir results are ordered arbitrarily
    file_paths = [os.path.join(video_tempDir, _) for _ in os.listdir(video_tempDir)]
    file_paths.sort()
    return [tf.gfile.FastGFile(_, 'rb').read() for _ in file_paths if os.path.isfile(_)]


def convert2jpeg(raw_image):
    temp_image = BytesIO()
    img = Image.fromarray(raw_image, 'RGB')
    img.save(temp_image, 'jpeg', quality=95)
    img.close()
    temp_image.seek(0)
    return temp_image.getvalue()


def decode_video_pipe(videopath):
    images = []
    if args.deinterlace == True:
        deinterlace = 'yadif'
    else:
        deinterlace = ''
    video_filename, video_file_extension = path.splitext(path.basename(videopath))
    print(' ')
    print('Reading video frames into memory from ' + video_filename)
    command = [
        FFMPEG_PATH, '-i', videopath,
        '-vf', 'fps=' + args.fps, '-r', args.fps, '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-vsync', 'vfr',
        '-hide_banner', '-loglevel', '0', '-vf', deinterlace, '-f', 'image2pipe', '-vf', 'scale=640x480', '-'
    ]
    image_pipe = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=4*1024*1024)

    while True:
        raw_image = image_pipe.stdout.read(640*480*3)
        if not raw_image:
            break
        image = np.fromstring(raw_image, dtype='uint8')
        image = image.reshape((480, 640, 3))

        images.append(convert2jpeg(image))
        image_pipe.stdout.flush()

    return images


def create_clip(videopath, event, totalframes, videoclipend):
    # creates a video clip of the detected event
    if (event[0] - (int(args.outputpadding) * int(args.fps))) >= 1:
        start = event[0] - int(args.outputpadding) * int(args.fps)
    else:
        start = 1
    if ((int(args.outputpadding) * int(args.fps)) + event[-1]) <= totalframes:
        end = event[-1] + (int(args.outputpadding) * int(args.fps))
    else:
        end = totalframes

    if event[0] >= videoclipend:
        starttime = datetime.timedelta(seconds=(start / int(args.fps)))
        endtime = datetime.timedelta(seconds=(end / int(args.fps)))
        duration = endtime - starttime
        # print('Creating video clip from event which starts at ' + str(starttime) + ' and ends at ' + str(endtime) + '.')

        video_reportDir = os.path.join(args.reportpath, '')
        video_filename, video_file_extension = path.splitext(path.basename(videopath))
        video_clip = video_reportDir + str(video_filename) + '_' + str(start) + '.mp4'

        command = [
            FFMPEG_PATH, '-ss', str(starttime), '-i', str(currentSrcVideo),
            '-t', str(duration), video_clip, '-y', '-loglevel', '0',
        ]
        subprocess.call(command)
        return end
    else:
        return videoclipend


def load_video_filenames(relevant_path):
    included_extenstions = ['avi', 'mp4', 'asf', 'mkv']
    return [fn for fn in os.listdir(relevant_path)
            if any(fn.lower().endswith(ext) for ext in included_extenstions)]


# Loads label files, strips off carriage return
def load_labels(path):
    # load the labels and remove stuff we don't want to display
    file_data = [line.split() for line
                 in tf.gfile.GFile(path)]
    return [item[0].split(":") for item in file_data]


def load_tensor_types(path):
    # reads in the input and output tensors
    with open(path) as file:
        content = file.readlines()
    return content[0].rstrip() + ':0', content[1].rstrip() + ':0'  # return minus any extra characters we don't need
    # else:
    #     if args.modeltype =='mobilenetV1':
    #         return 'input_image', 'MobilenetV1/Predictions/Reshape_1'
    #     if args.modeltype == 'inception_resnet_v2':
    #         return 'input_image', 'InceptionResnetV2/Logits/Predictions'


def setup_reporting(passed_filename):
    path = os.path.join(args.reportpath, '')
    reportFileName = path + passed_filename + '_report.csv'
    return open(reportFileName, 'w')


def setup_logging(passed_filename):
    path = os.path.join(args.reportpath, '')
    filename = path + passed_filename + '_results.csv'
    return open(filename, 'w')


def smoothListGaussian(list, strippedXs=False, degree=2):
    window = degree * 2 - 1
    weight = np.array([1.0] * window)
    weightGauss = []
    div_odd = lambda n: (n // 2, n // 2 + 1)

    for i in range(window):
        i = i - degree + 1

        frac = i / float(window)

        gauss = 1 / (np.exp((4 * (frac)) ** 2))

        weightGauss.append(gauss)

    weight = np.array(weightGauss) * weight
    smoothed = [0.0] * (len(list) - window)

    for i in range(len(smoothed)):
        smoothed[i] = float("{0:.4f}".format(sum(np.array(list[i:i + window]) * weight) / sum(weight)))

    padfront, padback = div_odd(window)
    for i in range(0, padfront):
        smoothed.insert(0, 0.0)
    for i in range(0, padback):
        smoothed.append(0.0)

    return smoothed

def load_model(modelpath):
    if os.path.exists(modelpath):
        with tf.gfile.FastGFile(modelpath, 'rb') as file:
            graph_def1 = tf.GraphDef()
            graph_def1.ParseFromString(file.read())
            tensor_id = str(uuid.uuid4())
            return tensor_id, tf.import_graph_def(graph_def1, name=tensor_id)
    else:
        return False


def write_reports(path, data, smoothing=0):
    smootheddata = []
    logfile = setup_logging(path)
    totalrec = len(data)

    if smoothing > 0:
        # Let's find out which columns have the data vs. just labels.
        prediction_columns = []
        column_count = 0
        for item in data[0]:
            if isinstance(item, float):
                prediction_columns.append(column_count)
            column_count += 1

        # Now lets create new smoothed columns of data!
        for datacolidx in prediction_columns:
            labelcol = []
            label = str(data[0][datacolidx - 1]) + 'SM'
            for i in range(0, totalrec):
                labelcol.append(label)
            smootheddata.append(labelcol)

            newcol = []
            for row in data:
                newcol.append(row[datacolidx])
            newcol = smoothListGaussian(newcol, False, smoothing)
            smootheddata.append(newcol)

    frame_num = 1
    line_num = 0
    for row in data:
        logfile.write('%s, ' % (frame_num))
        for item in row:
            logfile.write('%s, ' % (item))
        if smoothing > 0:
            for smooth_item in smootheddata:
                logfile.write('%s, ' % (smooth_item[line_num]))

        logfile.write("\n")
        frame_num += 1
        line_num += 1

    logfile.close()


def runGraph(image_data, input_tensor, output_tensor, labels, session, session_name):
    global flagfound
    global n
    results = []

    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess1.graph.get_tensor_by_name(session_name + '/' + output_tensor)
    input_placeholder = sess1.graph.get_tensor_by_name(session_name + '/' + input_tensor)

    print('Starting analysis on ' + str(len(image_data)) + ' video frames...')

    # count the number of labels
    top_k = []
    for i in range(0, len(labels)):
        top_k.append(i)

    for image in image_data:
        n = n + 1
        predictions = session.run(softmax_tensor, {input_placeholder: image})

        data_line = []
        for node_id in top_k:
            human_string = labels[node_id][1]
            score = predictions[0][node_id]

            score = float("{0:.4f}".format(score))
            data_line.append(human_string)
            data_line.append(score)

            # save frames that are around the decision boundary so they can then be used for later model re-training.
            if args.training == True:
                if score >= 0.50 and score <= 0.80:
                    save_training_frames(n, human_string)

        results.append(data_line)
        drawProgressBar(percentage(n, len(image_data)) / 100, 40)  # --------------------- Start processing logic

    image_data = []
    return results

if path.isdir(args.videopath):
    video_files = load_video_filenames(args.videopath)
    for video_file in video_files:
        filename, file_extension = path.splitext(path.basename(video_file))
        n = 0
        flagfound = 0
        remove_video_frames()
        clean_video_path = os.path.join(args.videopath, '')
        currentSrcVideo = clean_video_path + video_file

        if args.keeptemp or args.training:
            image_data = decode_video(currentSrcVideo)
        else:
            image_data = decode_video_pipe(currentSrcVideo)

        if args.modelpath.endswith('.pb'):
            tensorpath = args.modelpath[:-3] + '-meta.txt'
            labelpath = args.modelpath[:-3] + '-labels.txt'
        else:
            tensorpath = args.modelpath
            labelpath = args.modelpath + '-labels.txt'

        loaded_labels = load_labels(labelpath)
        input_tensor, output_tensor = load_tensor_types(tensorpath)
        a_graph_name, a_graph = load_model(args.modelpath)
        sess1 = tf.Session(graph=a_graph)
        output = runGraph(image_data, input_tensor, output_tensor, loaded_labels, sess1, a_graph_name)
        write_reports(filename, output, int(args.smoothing))

else:
    filename, file_extension = path.splitext(path.basename(args.videopath))
    n = 0
    flagfound = 0
    remove_video_frames()
    currentSrcVideo = args.videopath
    if args.keeptemp or args.training:
        image_data = decode_video(currentSrcVideo)
    else:
        image_data = decode_video_pipe(currentSrcVideo)

    if args.modelpath.endswith('.pb'):
        tensorpath = args.modelpath[:-3] + '-meta.txt'
        labelpath = args.modelpath[:-3] + '-labels.txt'
    else:
        tensorpath = args.modelpath
        labelpath = args.modelpath + '-labels.txt'

    loaded_labels = load_labels(labelpath)
    input_tensor, output_tensor = load_tensor_types(tensorpath)
    a_graph_name, a_graph = load_model(args.modelpath)
    sess1 = tf.Session(graph=a_graph)
    output = runGraph(image_data, input_tensor, output_tensor, loaded_labels, sess1, a_graph_name)
    write_reports(filename, output, int(args.smoothing))

if not args.keeptemp:
    remove_video_frames()

print(' ')
stop = timeit.default_timer()
total_time = stop - start
mins, secs = divmod(total_time, 60)
hours, mins = divmod(mins, 60)
sys.stdout.write("Total running time: %d:%d:%d.\n" % (hours, mins, secs))
