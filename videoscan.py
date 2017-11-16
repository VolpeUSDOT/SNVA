import tensorflow as tf
import sys
import os
import subprocess
import time
import datetime
import os.path as path
import platform
import argparse
from glob import iglob
from shutil import copy
import timeit

# TODO: Modify detection to only process data after inference has completed.
# DONE: Modify model unpersist function to use loaded model name vs. static assignment.
# TODO: Add support for loading multiple primary and secondary models.

# set start time
# start_time = time.time()
start = timeit.default_timer()

parser = argparse.ArgumentParser(description='Process some video files using Machine Learning!')
parser.add_argument('--temppath', '-tp', dest='temppath', action='store', default='../vidtemp/',
                    help='Path to the directory where temporary files are stored.')
parser.add_argument('--trainingpath', '-rtp', dest='trainingpath', action='store', default='../retraining/',
                    help='Path to the directory where frames for retraining are stored.')
parser.add_argument('--reportpath', '-rp', dest='reportpath', action='store', default='results/',
                    help='Path to the directory where results are stored.')
parser.add_argument('--modelpath', '-mp', dest='modelpath', action='store', default='models/default.pb',
                    help='Path to the tensorflow protobuf model file.')
parser.add_argument('--labelpath', '-lp', dest='labelpath', action='store', default='models/default-labels.txt',
                    help='Path to the tensorflow model labels file.')
parser.add_argument('--labelname', '-ln', dest='labelname', action='store', default='',
                    help='Name of primary label, used to trigger secondary model (if needed).')
parser.add_argument('--smoothing', '-sm', dest='smoothing', action='store', default='2',
                    help='Apply a type of "smoothing" factor to detection results.')
parser.add_argument('--fps', '-fps', dest='fps', action='store', default='1',
                    help='Frames Per Second used to sample input video. '
                         'The higher this number the slower analysis will go. Default is 1 FPS')
parser.add_argument('--allfiles', '-a', dest='allfiles', action='store_true',
                    help='Process all video files in the directory path.')
parser.add_argument('--outputclips', '-o', dest='outputclips', action='store_true',
                    help='Output results as video clips containing searched for labelname.')
parser.add_argument('--training', '-tr', dest='training', action='store_true',
                    help='Saves predicted frames for future model retraining.')
parser.add_argument('--outputpadding', '-op', dest='outputpadding', action='store', default='45',
                    help='Number of seconds added to the start and end of created video clips.')
parser.add_argument('--filter', '-f', dest='filter', action='store', default='ALL',
                    help='Value used to filter on a label.')
parser.add_argument('--keeptemp', '-k', dest='keeptemp', action='store_true',
                    help='Keep temporary extracted video frames.')
parser.add_argument('--video_path', '-v', dest='video_path', action='store', help='Path to video file(s).')

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


def decode_video(video_path):
    video_filename, video_file_extension = path.splitext(path.basename(video_path))
    print(' ')
    print('Decoding video file ' + video_filename)
    video_temp = os.path.join(video_tempDir, str(video_filename) + '_%04d.jpg')
    command = [
        FFMPEG_PATH, '-i', video_path,
        '-vf', 'fps=' + args.fps, '-q:v', '1', '-vsync', 'vfr', video_temp, '-hide_banner', '-loglevel', '0',
    ]
    subprocess.call(command)


def create_clip(video_path, event, totalframes, videoclipend):
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
        video_filename, video_file_extension = path.splitext(path.basename(video_path))
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
    return content[0].rstrip() + ':0', content[1].rstrip() + ':0'


def setup_reporting(passed_filename):
    path = os.path.join(args.reportpath, '')
    reportFileName = path + passed_filename + '_report.csv'
    return open(reportFileName, 'w')


def setup_logging(passed_filename):
    path = os.path.join(args.reportpath, '')
    filename = path + passed_filename + '_results.csv'
    return open(filename, 'w')


# Unpersists primary graph from file
with tf.gfile.FastGFile(args.modelpath, 'rb') as f:
    graph_def1 = tf.GraphDef()
    graph_def1.ParseFromString(f.read())
    primary_graph = tf.import_graph_def(graph_def1, name='primary')

# # Unpersists secondary graph from file
# with tf.gfile.FastGFile("models/zone-features_graph.pb", 'rb') as g:
#     graph_def2 = tf.GraphDef()
#     graph_def2.ParseFromString(g.read())
#     secondary_graph = tf.import_graph_def(graph_def2, name='secondary')

# setup sessions ahead of time
sess1 = tf.Session(graph=primary_graph)


# sess2 = tf.Session(graph=secondary_graph)

# def runsecondarygraph(image_tensor):
#
#     # Feed the image_data as input to the graph and get first prediction
#     softmax_tensor = sess2.graph.get_tensor_by_name('secondary/final_result:0')
#
#     predictions = sess2.run(softmax_tensor, \
#                         {'secondary/DecodeJpeg/contents:0': image_tensor})
#
#     # secondary_predictions = predictions[0].argsort()[-len(predictions[0]):][::-1]
#     secondary_predictions = [0, 1, 2]
#     for node in secondary_predictions:
#         human_string = secondary_graph_lines[node]
#         score = predictions[0][node]
#         reportTarget.write('%s, %.5f,' % (human_string, score))
#
#     print('Processed potential construction zone in frame #' + str(n))


def runGraph(image_path, input_tensor, output_tensor):
    global flagfound
    global n

    # Read in the image_data, but sort image paths first because os.listdir results are ordered arbitrarily
    file_paths = [os.path.join(image_path, _) for _ in os.listdir(image_path)]
    file_paths.sort()
    image_data = [tf.gfile.FastGFile(_, 'rb').read() for _ in file_paths if os.path.isfile(_)]
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess1.graph.get_tensor_by_name('primary/' + output_tensor)
    input_placeholder = sess1.graph.get_tensor_by_name('primary/' + input_tensor)

    print('Starting analysis on ' + str(len(image_data)) + ' video frames...')

    event = []  # setup event tracking
    videoclipend = 0
    initial_smoothing = int(args.smoothing)  # apply a per frame smoothing factor to the data
    initial_smoothing = initial_smoothing * int(args.fps)
    smoothing = 0
    for image in image_data:
        n = n + 1
        predictions = sess1.run(softmax_tensor, {input_placeholder: image})

        top_k = [0, 1]

        for node_id in top_k:
            human_string = primary_graph_lines[node_id][1]
            score = predictions[0][node_id]
            fileTarget.write('%s, %s, %.5f, ' % (n, human_string, score))

            if args.training == True:
                if score >= 0.75 and score <= 0.90:
                    save_training_frames(n, human_string)

            if (human_string == args.labelname):  # if the label detected matches the passed label to search for
                if score < 0.95 and score >= 0.75:
                    if args.filter.upper() == 'ALL':
                        reportTarget.write('%s, %s, %.5f, %s, ' % (n, human_string, score, 'Medium'))
                        reportTarget.write('\n')

                if score >= 0.95:
                    reportTarget.write('%s, %s, %.5f, %s, ' % (n, human_string, score, 'High'))
                    # runsecondarygraph(image)
                    reportTarget.write('\n')
                    smoothing = initial_smoothing
                    event.append(n)

        if score < 0.75:
            if (smoothing == 0):
                if len(event) >= initial_smoothing:
                    # print('Event end on frame ' + str(n))
                    if args.outputclips == True:
                        videoclipend = create_clip(video_file, event, len(image_data), videoclipend)
                event = []
                smoothing = initial_smoothing
            else:
                smoothing = smoothing - 1
                # reportTarget.write('%s, %s, %.5f, %s, ' % (n, human_string, score, 'Low'))
                # reportTarget.write('\n')

    fileTarget.write("\n")
    # print('Current = ' + str(n) + '  Total = ' + str(len(image_data)))
    # print(int(percentage(n, len(image_data))))
    drawProgressBar(percentage(n, len(image_data)) / 100, 40)  # --------------------- Start processing logic
    # if only one file was passed for analysis then inject it into the passed array


if args.allfiles:
    video_files = load_video_filenames(args.video_path)
    for video_file in video_files:
        # setup reporting and search flags
        filename, file_extension = path.splitext(path.basename(video_file))
        reportTarget = setup_reporting(filename)
        fileTarget = setup_logging(filename)
        n = 0
        flagfound = 0
        remove_video_frames()
        clean_video_path = os.path.join(args.video_path, '')
        currentSrcVideo = clean_video_path + video_file
        decode_video(currentSrcVideo)
        primary_graph_lines = load_labels(args.labelpath)
        if args.modelpath.endswith('.pb'):
            tensorpath = args.modelpath[:-3] + '-model.txt'
        else:
            tensorpath = args.modelpath
        input_tensor, output_tensor = load_tensor_types(tensorpath)
        runGraph(video_tempDir, input_tensor, output_tensor)
else:
    filename, file_extension = path.splitext(path.basename(args.video_path))
    reportTarget = setup_reporting(filename)
    fileTarget = setup_logging(filename)
    n = 0
    flagfound = 0
    remove_video_frames()
    currentSrcVideo = args.video_path
    decode_video(currentSrcVideo)
    primary_graph_lines = load_labels(args.labelpath)
    if args.modelpath.endswith('.pb'):
        tensorpath = args.modelpath[:-3] + '-model.txt'
    else:
        tensorpath = args.modelpath
    input_tensor, output_tensor = load_tensor_types(tensorpath)
    runGraph(video_tempDir, input_tensor, output_tensor)

if not args.keeptemp:
    remove_video_frames()

print(' ')
# print("--- Completed in %s seconds ---" % (datetime.datetime.fromtimestamp(time.time() - start_time)).strftime('%M:%S'))
stop = timeit.default_timer()
total_time = stop - start
mins, secs = divmod(total_time, 60)
hours, mins = divmod(mins, 60)
sys.stdout.write("Total running time: %d:%d:%d.\n" % (hours, mins, secs))
