import os
import subprocess
import timeit
import platform
import argparse
from shutil import move
from skimage import io as skimage_io
import numpy as np

path = os.path

parser = argparse.ArgumentParser(description='Process some video files using Machine Learning!')
parser.add_argument('--imagepath', '-i', action='store', required=True,
                    help='Path to the directory where extracted images are stored.')
parser.add_argument('--trainingpath', '-rtp', dest='trainingpath', action='store', default='./retraining/',
                    help='Path to the directory where frames for retraining are stored.')
parser.add_argument('--fps', '-f', action='store', default='1', help='Frames Per Second used to sample input video. '
                                                                     'The higher this number the slower analysis will go. Default is 1 FPS')
parser.add_argument('--videopath', '-v', action='store', required=True, help='Path to video file(s).')
parser.add_argument('--allfiles', '-a', action='store_true', help='Process all video files in the directory path.')
parser.add_argument('--deinterlace', '-di', action='store_true', help='Deinterlace video frames.')
parser.add_argument('--scale', '-s', action='store_true')
parser.add_argument('--outputwidth', '-sw', type=int, default=299, help='Video frame output width.')
parser.add_argument('--outputheight', '-sh', type=int, default=299, help='Video frame output height.')
parser.add_argument('--crop', '-c', action='store_true',
                    help='Crop video frames to [offsetheight, offsetwidth, targetheight, targetwidth]')
parser.add_argument('--cropx', '-cx', type=int, help='x-component of top-left corner of crop.')
parser.add_argument('--cropy', '-cy', type=int, help='y-component of top-left corner of crop.')
parser.add_argument('--cropwidth', '-cw', type=int, help='x-component of bottom-right corner of crop.')
parser.add_argument('--cropheight', '-ch', type=int, help='y-component of bottom-right corner of crop.')
parser.add_argument('--epsilon', '-e', type=float, default=0.3,
                    help='If training, images with classification confidences in the range 0.5 +/- epsilon will be '
                         'stored for use in future training')
parser.add_argument('--usesymlinks', '-u', dest='usesymlinks', action='store_true',
                    help='When classifying frames, create symbolic links to the source frames instead of copies of '
                         'them.')
parser.add_argument('--deduplicate', '-dd', action='store_true', help='Relocate duplicate video frames.')

args = parser.parse_args()

if platform.system() == 'Windows':
  FFMPEG_PATH = 'ffmpeg.exe'
else:
  default_ffmpeg_path = '/usr/local/bin/ffmpeg'
  FFMPEG_PATH = default_ffmpeg_path if path.exists(default_ffmpeg_path) \
    else '/usr/bin/ffmpeg'

# setup video temp directory for video frames
if not os.path.isdir(args.imagepath):
  os.mkdir(args.imagepath)


def decode_video(video_path):
  print(' \nDecoding video')

  video_filename = path.splitext(path.basename(video_path))[0]
  image_dir = os.path.join(args.imagepath, video_filename)

  if path.exists(image_dir):
    print('image_dir {} already exists. skipping extraction'.format(image_dir))
    return

  if not path.isdir(image_dir):
    os.makedirs(image_dir)

  image_path = os.path.join(image_dir, video_filename + '_%07d.jpg')

  command = [FFMPEG_PATH, '-i', video_path]

  filter_args = []

  if args.fps:
    filter_args.append('fps={}'.format(args.fps))

  if args.crop and all([args.cropwidth > 0, args.cropheight > 0, args.cropx >= 0, args.cropy >= 0]):
    filter_args.append('crop={}:{}:{}:{}'.format(
      args.cropwidth, args.cropheight, args.cropx, args.cropy))

  if args.scale and all([args.outputwidth > 0, args.outputheight > 0]):
    filter_args.append('scale={}:{}'.format(args.outputwidth, args.outputheight))

  filter_args_len = len(filter_args)

  if filter_args_len > 0:
    command.append('-vf')
    filter = ''
    for i in range(filter_args_len - 1):
      filter += filter_args[i] + ','
    filter += filter_args[filter_args_len - 1]
    command.append(filter)

  command.extend(['-q:v', '1', '-vsync', 'vfr', image_path, '-hide_banner', '-loglevel', '0'])

  if args.deinterlace:
    command.append('-deinterlace')

  print('command: {}'.format(command))

  _ = subprocess.run(command)

  if args.deduplicate:
    separate_duplicate_frames(image_dir, video_filename)


def load_video_filenames(relevant_path):
  included_extenstions = ['avi', 'mp4', 'asf', 'mkv']
  return [fn for fn in os.listdir(relevant_path)
          if any(fn.lower().endswith(ext) for ext in included_extenstions)]


def separate_duplicate_frames(image_dir, video_filename):
  """Videos that have been re-encoded at a higher frame rate than the original source
  will likely contain many duplicate frames. This function sequentially iterates over
  a folder containing image frames extracted from a video, under the assumption that
  the order of the frames' file names is consistent with their order in the video.

  Arguments:
      image_dir: The directory containing the extracted images.
  """
  image_names = sorted(os.listdir(image_dir))
  image_names_len = len(image_names)

  if image_names_len < 2:
    raise AssertionError('Separation of duplicate frames halted. Expected at least 2 images '
                         'to exist in ' + image_dir + ', but found ' + str(image_names_len))

  image_paths = [path.join(image_dir, image_name) for image_name in image_names]

  print('Removing duplciate frames.')

  left_image_ptr = 0
  left_image = skimage_io.imread(image_paths[left_image_ptr])

  right_image_ptr = 1
  right_image = skimage_io.imread(image_paths[right_image_ptr])

  while right_image_ptr < image_names_len - 1:
    if np.any(np.diff([left_image, right_image], axis=0)):
      left_image_ptr += 1
      move(image_paths[right_image_ptr],
           path.join(image_dir, '{}_{:07d}.jpg'.format(video_filename, left_image_ptr + 1)))
      left_image = right_image
    else:
      os.remove(image_paths[right_image_ptr])

    right_image_ptr += 1
    right_image = skimage_io.imread(image_paths[right_image_ptr])

  if np.any(np.diff([left_image, right_image], axis=0)):
    left_image_ptr += 1
    move(image_paths[right_image_ptr],
         path.join(image_dir, '{}_{:07d}.jpg'.format(video_filename, left_image_ptr + 1)))
  else:
    os.remove(image_paths[right_image_ptr])


def main():
  if not path.isdir(args.imagepath):
    os.mkdir(args.imagepath)

  if not path.isdir(args.trainingpath):
    os.mkdir(args.trainingpath)

  start = timeit.default_timer()

  if path.isdir(args.videopath):
    video_files = load_video_filenames(args.videopath)
    for video_file in video_files:
      video_path = os.path.join(args.videopath, video_file)
      decode_video(video_path)
  elif path.isfile(args.videopath):
    decode_video(args.videopath)
  else:
    raise ValueError('The path {} does not exist.'.format(args.videopath))

  print('Extraction time: {}'.format(timeit.default_timer() - start))

if __name__ == '__main__':
  main()
