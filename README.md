# SHRP2 NDS Video Analytics (SNVA) v0.1.2

This repository houses the SNVA application and additional code used to develop the computer vision models at the core of SNVA. Model development code is based on [TensorFlow-Slim](https://github.com/tensorflow/models/tree/master/research/slim). The project is described in detail in our paper: [arXiv preprint arXiv:1811.04250, 2018](https://arxiv.org/abs/1811.04250).

SNVA is intended to expand the Roadway Information Database (RID)’s ability to help transportation safety researchers develop and answer research questions. The RID is the primary source of data collected as part of the FHWA’s SHRP2 Naturalistic Driving Study, including vehicle telemetry, geolocation, and roadway characteristics data. Missing from the RID are the locations of work zones driven through by NDS volunteer drivers. The app’s first release will focus on enabling/enhancing research questions related to work zone safety by using machine learning-based computer vision techniques to exhaustively and automatically detect the presence of work zone features across the entire ~1 million-hour forward-facing video data set, and then conflating that information with the RID’s time-series records. Previously, researchers depended on sparse and low fidelity 511 data provided by states that hosted the routes driven by volunteers. A successful deployment of the SNVA app will make it possible to query the RID for the exact start/stop locations, lengths, and frequencies of work zones in trip videos; a long-standing, highly desired ability within the SHRP2 community.


## Required Software Dependencies

SNVA has been tested using the following software stack:

- Ubuntu = 16.04
- Python >= 3.5
- TensorFlow = 1.8 (and its published dependencies)
- FFmpeg >= 2.8


## Optional Software Dependencies

Inference speed was observed to improve by ~10% by building TensorFlow from source and including:

- TensorRT = 3.0.4

Installation of the Docker-containerized version of SNVA has been tested using:

- NVIDIA-Docker = 2.0.3
- Docker = 18.03.1-CE


## System Requirements and Performance Expectations

SNVA is intended to run on systems with NVIDIA GPUs, but can also run in a CPU-only mode. SNVA runs ~10x faster on a single NVIDIA GeForce GTX 1080 Ti together with a 3.00GHz 10-core Intel Core i7-6950X CPU than it does on the 10-core CPU alone. For a system with N GPUs and for --numprocessesperdevice = M, SNVA will process N * M videos concurrently, but is not (at this time) designed to distribute the processing of a single video across multiple GPUs. Inference speeds depend on the particular CNN architecture used to develop the model. When tested on two GPUs against 31,535,862 video frames spanning 1,344 videos, InceptionV3 inferred class labels at 826.24 fps on average over 10:40:04 hours, MobilenetV2 (with one video processor assigned to each GPU) averaged 1491.36 fps over 05:58:20 hours, and MobilenetV2 (with two video processors assigned to each GPU) averaged 1833.1 fps over 4:53:42 hours. RAM consumption on our development machine appeared to be safely bounded above by 3.75GB per active video processor.


## To install on Ubuntu:

```shell
export SNVA_HOME=/path/to/parent/folder/of/snva.py
export FFMPEG_HOME=/path/to/parent/folder/of/ffmpeg/binary
export FFPROBE_HOME=/path/to/parent/folder/of/ffprobe/binary

cd /path/to/parent/folder/of/SNVA/repo
mkdir SNVA
git clone https://github.com/VolpeUSDOT/SNVA.git SNVA
```

## To run on Ubuntu:

```shell
python3 snva.py
  --inputpath /path/to/your/desired/video_file/source/directory/or/file \
  --modelname mobilenet_v2
```

```shell
python snva.py
  --modelname mobilenet_v2 \
  --inputpath /path/to/your/desired/video_file/source/directory/or/file \
  --outputpath /path/to/your/desired/report/directory \
  --logpath /path/to/your/desired/report/directory/log
  --batchsize 128 --loglevel debug --smoothprobs --extracttimestamps --crop \
  --writeinferencereports True
```

## To run using NVIDIA-Docker on Ubuntu (for a text file listing absolute paths to videos):

```shell
sudo nvidia-docker run \
  --mount type=bind, \
    src=/path/to/your/desired/video/source/file/or/directory,dst=/media/input \
  --mount type=bind, \
    src=/path/to/a/common/root/video/directory/when/inputpath/is/a/text/file, \
    dst=/media/root \
  --mount type=bind, \
    src=/path/to/your/desired/csv_file/destination/directory,dst=/media/output \
  --mount type=bind, \
    src=/path/to/your/desired/log_file/destination/directory,dst=/media/logs \
  volpeusdot/snva \
  --inputlistrootdirpath /common/root/path/on/the/host \
  --inputpath /media/input --outputpath /media/output --logspath /media/logs \
  --modelname inception_v3 --batchsize 64 --smoothprobs --extracttimestamps \
  --crop --writeinferencereports True
```
## To run using NVIDIA-Docker on Ubuntu (for a directory of videos):

```shell
sudo nvidia-docker run \
  --mount type=bind, \
    src=/path/to/your/desired/video_file/source/directory,dst=/media/input \
  --mount type=bind, \
    src=/path/to/your/desired/csv_file/destination/directory,dst=/media/output \
  --mount type=bind, \
    src=/path/to/your/desired/log_file/destination/directory,dst=/media/logs \
  volpeusdot/snva \
  --inputpath /media/input --outputpath /media/output --logspath /media/logs \
  --modelname inception_v3 --batchsize 64 --smoothprobs --extracttimestamps \
  --crop --writeinferencereports True
```

## To run using NVIDIA-Docker on Ubuntu (for a single video):

```shell
sudo nvidia-docker run \
  --mount type=bind, \
    src=/path/to/your/desired/video_file/source/directory/video_file_name.ext, \
    dst=/media/input/video_file_name.ext \
  --mount type=bind, \
    src=/path/to/your/desired/csv_file/destination/directory,dst=/media/output \
  --mount type=bind, \
    src=/path/to/your/desired/log_file/destination/directory,dst=/media/logs \
  volpeusdot/snva \
  --inputpath /media/input/ --outputpath /media/output --logspath /media/logs \
  --modelname inception_v3 --batchsize 64 --smoothprobs --extracttimestamps \
  --crop --writeinferencereports True
```


## Usage

Flag | Short Flag | Properties | Description
:------:|:---------------:|:---------------------:|:-----------:
--batchsize|-bs|type=int, default=32|Number of concurrent neural net inputs
--binarizeprobs|-b|action=store_true|Round probs to zero or one. For distributions with two 0.5 values, both will be rounded up to 1.0
--classnamesfilepath|-cnfp||Path to the class ids/names text file
--cpuonly|-cpu|action=store_true|Useful for systems without an NVIDIA GPU
--crop|-c|action=store_true|Crop video frames to [offsetheight, offsetwidth, targetheight, targetwidth]
--cropheight|-ch|type=int, default=320|y-component of bottom-right corner of crop
--cropwidth|-cw|type=int, default=474|x-component of bottom-right corner of crop
--cropx|-cx|type=int, default=2|x-component of top-left corner of crop
--cropy|-cy|type=int, default=0|y-component of top-left corner of crop
--deinterlace|-d|action=store_true|Apply de-interlacing to video frames during extraction
--excludepreviouslyprocessed|-epp|action=store_true|Skip processing of videos for which reports already exist in outputpath
--extracttimestamps|-et|action=store_true|Crop timestamps out of video frames and map them to strings for inclusion in the output CSV
--gpumemoryfraction|-gmf|type=float, default=0.9|% of GPU memory available to this process
--inputpath|-ip|required=True|Path to a single video file, a folder containing video files, or a text file that lists absolute video file paths
--inputlistrootdirpath|-ilrdp|Path to the common root directory shared by video file paths listed in the text file specified using --inputpath
--ionodenamesfilepath|-ifp|Path to the io tensor names text file
--loglevel|-ll|default=info|Defaults to 'info'. Pass 'debug' or 'error' for verbose or minimal logging, respectively
--logmode|-lm|default=verbose|If verbose, log to file and console. If silent, log to file only
--logpath|-l|default=logs|Path to the directory where log files are stored
--logmaxbytes|-lmb|type=int|default=2**23|File size in bytes at which the log rolls over
--modelsdirpath|-mdp|default=models/work_zone_scene_detection|Path to the parent directory of model directories
--modelname|-mn|required=True|The square input dimensions of the neural net
--numchannels|-nc|type=int, default=3|The fourth dimension of image batches
--numprocessesperdevice|-nppd|type=int, default=1|The number of instances of inference to perform on each device
--protobuffilename|-pbfn|default=model.pb|Name of the model protobuf file
--outputpath|-op|default=reports|Path to the directory where reports are stored
--smoothprobs|-sp|action=store_true|Apply class-wise smoothing across video frame class probability distributions
--smoothingfactor|-sf|type=int, default=16|The class-wise probability smoothing factor
--timestampheight|-th|type=int, default=16|The length of the y-dimension of the timestamp overlay
--timestampmaxwidth|-tw|type=int, default=160|The length of the x-dimension of the timestamp overlay
--timestampx|-tx|type=int, default=25|x-component of top-left corner of timestamp (before cropping)
--timestampy|-ty|type=int, default=340|y-component of top-left corner of timestamp (before cropping)
--writeeventreports|-wer|type=bool, default=True|Output a CVS file for each video containing one or more feature events
--writeinferencereports|-wir|type=bool, default=False|For every video, output a CSV file containing a probability distribution over class labels, a timestamp, and a frame number for each frame


## Troubleshooting and Additional Considerations

If a timestamp cannot be interpreted, a -1 will be written in its place in the output CSV.

While inference speed has been observed to monotonically increase with batch size, it is important to not exceed the GPU's memory capacity. The SNVA app does not automatically determine the optimal batch size for maximum inference speed. It is best to discover the optimal batch size by testing the app on a small sample of videos (say ~15) starting at a relatively low batch size, then iteratively incrementing the batch size while monitoring GPU memory utilization (e.g. using the NVIDIA X Server Settings GUI app or nvidia-smi CLI app: nvidia-smi --query-compute-apps=process_name,pid,used_gpu_memory --format=csv) and also observing the cumulative analysis duration printed at the end of each run. GPU memory is set to be dynamically allocated, so one should monitor its usage over time to increase the chance of observing peak utilization.

When terminating the app using ctrl-c, there may be a delay while the app terminates gracefully.

When terminating the dockerized app, use ctrl-c to let the app terminate gracefully before invoking the nvidia-docker stop command (which actually shouldn't be needed).

Windows is not officially supported but may be used with minor code tweaks.

When using Docker, some extraneous C++ output is passed to the host machine's console that is not actually logged to file and is not intended to be seen. Consider this a bug and ignore it.
## License

[MIT](https://opensource.org/licenses/MIT)
