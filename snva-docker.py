import argparse
from os import path
import subprocess


def get_mount_path(src, dst):
  return ['--mount', 'type=bind,src={},dst={}'.format(src, dst)]


def get_num_bytes(string_list):
  return sum([len(elem) for elem in string_list]) + len(string_list) - 1


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='SHRP2 NDS Video Analytics built on TensorFlow')

  parser.add_argument('--batchsize', '-bs', default='32',
                      help='Number of concurrent neural net inputs')
  parser.add_argument('--binarizeprobs', '-b', action='store_true',
                      help='Round probs to zero or one. For distributions with '
                           ' two 0.5 values, both will be rounded up to 1.0')
  parser.add_argument('--classnamesfilepath', '-cnfp',
                      help='Path to the class ids/names text file.')
  parser.add_argument('--cpuonly', '-cpu', action='store_true', help='')
  parser.add_argument('--crop', '-c', action='store_true',
                      help='Crop video frames to [offsetheight, offsetwidth, '
                           'targetheight, targetwidth]')
  parser.add_argument('--cropheight', '-ch', default='320',
                      help='y-component of bottom-right corner of crop.')
  parser.add_argument('--cropwidth', '-cw', default='474',
                      help='x-component of bottom-right corner of crop.')
  parser.add_argument('--cropx', '-cx', default='2',
                      help='x-component of top-left corner of crop.')
  parser.add_argument('--cropy', '-cy', default='0',
                      help='y-component of top-left corner of crop.')
  parser.add_argument('--deinterlace', '-d', action='store_true',
                      help='Apply de-interlacing to video frames during '
                           'extraction.')
  parser.add_argument('--excludepreviouslyprocessed', '-epp',
                      action='store_true',
                      help='Skip processing of videos for which reports '
                           'already exist in outputpath.')
  parser.add_argument('--extracttimestamps', '-et', action='store_true',
                      help='Crop timestamps out of video frames and map them to'
                           ' strings for inclusion in the output CSV.')
  parser.add_argument('--gpumemoryfraction', '-gmf', default='0.9',
                      help='% of GPU memory available to this process.')
  parser.add_argument('--inputpath', '-ip', required=True,
                      help='Path to list of video file paths.')
  parser.add_argument('--ionodenamesfilepath', '-ifp',
                      help='Path to the io tensor names text file.')
  parser.add_argument('--loglevel', '-ll', default='info',
                      help='Defaults to \'info\'. Pass \'debug\' or \'error\' '
                           'for verbose or minimal logging, respectively.')
  parser.add_argument('--logmode', '-lm', default='verbose',
                      help='If verbose, log to file and console. If silent, '
                           'log to file only.')
  parser.add_argument('--logpath', '-l', default='logs',
                      help='Path to the directory where log files are stored.')
  parser.add_argument('--logmaxbytes', '-lmb', default=str(2**23),
                      help='File size in bytes at which the log rolls over.')
  parser.add_argument('--modelsdirpath', '-mdp',
                      default='models/work_zone_scene_detection',
                      help='Path to the parent directory of model directories.')
  parser.add_argument('--modelname', '-mn', default='inception_v3',
                      help='The square input dimensions of the neural net.')
  parser.add_argument('--numchannels', '-nc', default='3',
                      help='The fourth dimension of image batches.')
  parser.add_argument('--numprocessesperdevice', '-nppd', default='1',
                      help='The number of instances of inference to perform on '
                           'each device.')
  parser.add_argument('--protobuffilename', '-pbfn', default='model.pb',
                      help='Name of the model protobuf file.')
  parser.add_argument('--outputpath', '-op', default='reports',
                      help='Path to the directory where reports are stored.')
  parser.add_argument('--smoothprobs', '-sp', action='store_true',
                      help='Apply class-wise smoothing across video frame class'
                           ' probability distributions.')
  parser.add_argument('--smoothingfactor', '-sf', default='16',
                      help='The class-wise probability smoothing factor.')
  parser.add_argument('--timestampheight', '-th', default='16',
                      help='The length of the y-dimension of the timestamp '
                           'overlay.')
  parser.add_argument('--timestampmaxwidth', '-tw', default='160',
                      help='The length of the x-dimension of the timestamp '
                           'overlay.')
  parser.add_argument('--timestampx', '-tx', default='25',
                      help='x-component of top-left corner of timestamp '
                           '(before cropping).')
  parser.add_argument('--timestampy', '-ty', default='340',
                      help='y-component of top-left corner of timestamp '
                           '(before cropping).')
  parser.add_argument('--writeeventreports', '-wer', default='True',
                      help='Output a CVS file for each video containing one or '
                           'more feature events')
  parser.add_argument('--writeinferencereports', '-wir', default='False',
                      help='For every video, output a CSV file containing a '
                           'probability distribution over class labels, a '
                           'timestamp, and a frame number for each frame')

  args = parser.parse_args()

  if not (path.isfile(args.inputpath)
          and path.splitext(args.inputpath)[1] == '.txt'):
    raise ValueError('Expected --inputpath to be an absolute path to a text '
                     'file ending in .txt')

  docker_command_head = ['sudo', 'nvidia-docker', 'run']

  docker_command_tail = []

  docker_output_path = '/media/output'

  # example args.outputpath:
  # /media/data_1/snva_1/Reports/fhwa/shrp2_nds/test_videos/reports
  docker_command_tail.extend(get_mount_path(args.outputpath, docker_output_path))

  docker_log_path = '/media/logs'

  # example args.logs:
  # /media/data_1/snva_1/Reports/fhwa/shrp2_nds/test_videos/logs
  docker_command_tail.extend(get_mount_path(args.logpath, docker_log_path))

  docker_command_tail.append('volpeusdot/snva:v0.1.2')

  docker_input_path = '/media/input'

  snva_arguments = [
    '--batchsize', args.batchsize,
    '--cropheight', args.cropheight,
    '--cropwidth', args.cropwidth,
    '--cropx', args.cropx,
    '--cropy', args.cropy,
    '--gpumemoryfraction', args.gpumemoryfraction,
    '--inputpath', docker_input_path,
    '--loglevel', args.loglevel,
    '--logmode', args.logmode,
    '--logpath', docker_log_path,
    '--logmaxbytes', args.logmaxbytes,
    '--modelsdirpath', args.modelsdirpath,
    '--modelname', args.modelname,
    '--numchannels', args.numchannels,
    '--numprocessesperdevice', args.numprocessesperdevice,
    '--protobuffilename', args.protobuffilename,
    '--outputpath', docker_output_path,
    '--smoothingfactor', args.smoothingfactor,
    '--timestampheight', args.timestampheight,
    '--timestampmaxwidth', args.timestampmaxwidth,
    '--timestampx', args.timestampx,
    '--timestampy', args.timestampy,
    '--writeeventreports', args.writeeventreports,
    '--writeinferencereports', args.writeinferencereports]

  if args.binarizeprobs:
    snva_arguments.append('--binarizeprobs')

  if args.cpuonly:
    snva_arguments.append('--cpuonly')

  if args.crop:
    snva_arguments.append('--crop')

  if args.deinterlace:
    snva_arguments.append('--deinterlace')

  if args.excludepreviouslyprocessed:
    snva_arguments.append('--excludepreviouslyprocessed')

  if args.extracttimestamps:
    snva_arguments.append('--extracttimestamps')

  if args.smoothprobs:
    snva_arguments.append('--smoothprobs')

  if args.classnamesfilepath:
    snva_arguments.extend(['--classnamesfilepath', args.classnamesfilepath])

  if args.ionodenamesfilepath:
    snva_arguments.extend(['--ionodenamesfilepath', args.ionodenamesfilepath])

  docker_command_tail.extend(snva_arguments)

  docker_command_bodies = []

  max_bytes = 4096 * 31

  docker_command_body = []

  with open(args.inputpath, newline='') as input_file:
    file_paths = {}

    cumulative_bytes = get_num_bytes(docker_command_head + docker_command_tail)

    for line in input_file.readlines():
      clean_line = line.rstrip()

      dir_path, file_name = path.split(clean_line)

      if file_name in file_paths.keys():
        raise ValueError(
          'Expected each input video file to be listed once and have a unique '
          'path, but found {} to be listed within {} and {}'.format(
            file_name, file_paths[file_name], dir_path))
      else:
        file_paths[file_name] = dir_path

        next_appendage = get_mount_path(
          clean_line, path.join(docker_input_path, file_name))

        next_bytes = get_num_bytes(next_appendage)

        if cumulative_bytes + next_bytes < max_bytes:
          docker_command_body.extend(next_appendage)
          cumulative_bytes += next_bytes
        else:
          docker_command_bodies.append(docker_command_body)
          docker_command_body = next_appendage
          cumulative_bytes = get_num_bytes(
            docker_command_head + docker_command_tail) + next_bytes

  for docker_command_body in docker_command_bodies:
    snva_process = subprocess.run(
      docker_command_head + docker_command_body + docker_command_tail)