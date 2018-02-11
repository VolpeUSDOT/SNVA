import sys
import os
import os.path as path
import subprocess
import timeit
import platform
import argparse
from shutil import move
from skimage import io as skimage_io
import numpy as np

# TODO: Modify detection to only process data after inference has completed.
# TODO: Modify model unpersist function to use loaded model name vs. static assignment.
# TODO: Add support for loading multiple primary and secondary models.

parser = argparse.ArgumentParser(description='Process some video files using Machine Learning!')
parser.add_argument('--imagepath', '-i', action='store', required=True,
                    help='Path to the directory where extracted images are stored.')
parser.add_argument('--duplicatepath', '-d', action='store',
                    help='Path to the directory where duplicate images are moved after being extracted.')
parser.add_argument('--fps', '-f', action='store', default='1', help='Frames Per Second used to sample input video. '
                    'The higher this number the slower analysis will go. Default is 1 FPS')
parser.add_argument('--videopath', '-v', action='store', required=True, help='Path to video file(s).')
parser.add_argument('--allfiles', '-a', action='store_true', help='Process all video files in the directory path.')
parser.add_argument('--deinterlace', '-di', action='store_true', help='Deinterlace video frames.')
parser.add_argument('--deduplicate', '-dd', action='store_true', help='Relocate duplicate video frames.')

args = parser.parse_args()

if platform.system() == 'Windows':
    # path to ffmpeg bin
    FFMPEG_PATH = 'ffmpeg.exe'
else:
    # path to ffmpeg bin
    default_ffmpeg_path = '/usr/local/bin/ffmpeg'
    FFMPEG_PATH = default_ffmpeg_path if path.exists(default_ffmpeg_path) else '/usr/bin/ffmpeg'

# setup video temp directory for video frames
if not os.path.isdir(args.imagepath):
    os.mkdir(args.imagepath)


def decode_video(video_path):
    video_filename, video_file_extension = path.splitext(path.basename(video_path))
    print(' ')
    print('Decoding video file ' + video_filename)
    image_dir = os.path.join(args.imagepath, str(video_filename))
    if not path.isdir(image_dir):
        os.mkdir(image_dir)
    #TODO: base number of digits in file name on expected number of frames based on size of video file
    image_path = os.path.join(image_dir, str(video_filename) + '_%07d.jpg')
    command = [FFMPEG_PATH, '-i', video_path, '-vf', 'fps=' + args.fps, '-q:v', '1', '-vsync',
               'vfr', image_path, '-hide_banner', '-loglevel', '0']

    if args.deinterlace:
        command.append('-deinterlace')

    subprocess.call(command)

    if args.deduplicate:
        separate_duplicate_frames(image_dir)


def load_video_filenames(relevant_path):
    included_extenstions = ['avi', 'mp4', 'asf', 'mkv']
    return [fn for fn in os.listdir(relevant_path)
            if any(fn.lower().endswith(ext) for ext in included_extenstions)]


def separate_duplicate_frames(image_dir):
    """Videos that have been re-encoded at a higher frame rate than the original source
    will likely contain many duplicate frames. This function sequentially iterates over
    a folder containing image frames extracted from a video, under the assumption that
    the order of the frames' file names is consistent with their order in the video.

    Arguments:
        image_dir: The directory containing the extracted images.
    """
    image_names = os.listdir(image_dir)
    image_names_len = len(image_names)

    if image_names_len < 2:
        raise AssertionError('Separation of duplicate frames halted. Expected at least 2 images to exist in '
                             + image_dir + ', but found ' + str(image_names_len))

    image_names.sort()

    image_paths = [path.join(image_dir, image_name) for image_name in image_names]

    if args.duplicatepath:
        duplicates_dir = args.duplicatepath
    else:
        duplicates_dir = path.join(image_dir, 'duplicates')

    if not path.exists(duplicates_dir):
        os.mkdir(duplicates_dir)

    print('Moving duplciate frames to ' + duplicates_dir)

    left_image = skimage_io.imread(image_paths[0])

    right_image_ptr = 1
    right_image = skimage_io.imread(image_paths[right_image_ptr])

    while right_image_ptr < image_names_len - 1:
        if np.any(np.diff([left_image, right_image], axis=0)):
            left_image = right_image
        else:
            move(image_paths[right_image_ptr], duplicates_dir)

        right_image_ptr += 1
        right_image = skimage_io.imread(image_paths[right_image_ptr])

    if not np.any(np.diff([left_image, right_image], axis=0)):
        move(image_paths[right_image_ptr], duplicates_dir)

# set start time
start = timeit.default_timer()

if args.allfiles:
    video_files = load_video_filenames(args.videopath)
    for video_file in video_files:
        video_path = os.path.join(args.videopath, video_file)
        decode_video(video_path)
else:
    decode_video(args.videopath)

print(' ')
stop = timeit.default_timer()
total_time = stop - start
mins, secs = divmod(total_time, 60)
hours, mins = divmod(mins, 60)
sys.stdout.write("Total running time: %d:%d:%d.\n" % (hours, mins, secs))
