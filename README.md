# SHRP2 NDS Video Analytics (SNVA) v0.2

This repository houses the SNVA application and additional code used to develop the computer vision models at the core of SNVA. Model development code is based on [TensorFlow-Slim](https://github.com/tensorflow/models/tree/master/research/slim). 

v0.1 of the project is described in detail in our paper: [arXiv preprint arXiv:1811.04250, 2018](https://arxiv.org/abs/1811.04250). If you were directed here by our paper, the v0.1 code may be found [here](https://github.com/VolpeUSDOT/SNVA/tree/v0.1.2).

SNVA is intended to expand the Roadway Information Database (RID)’s ability to help transportation safety researchers develop and answer research questions. The RID is the primary source of data collected as part of the FHWA’s SHRP2 Naturalistic Driving Study, including vehicle telemetry, geolocation, and roadway characteristics data. Missing from the RID are the locations of work zones driven through by NDS volunteer drivers. The app’s first release will focus on enabling/enhancing research questions related to work zone safety by using machine learning-based computer vision techniques to exhaustively and automatically detect the presence of work zone features across the entire ~1 million-hour forward-facing video data set, and then conflating that information with the RID’s time-series records. Previously, researchers depended on sparse and low fidelity 511 data provided by states that hosted the routes driven by volunteers. A successful deployment of the SNVA app will make it possible to query the RID for the exact start/stop locations, lengths, and frequencies of work zones in trip videos; a long-standing, highly desired ability within the SHRP2 community.

## Architecture

SNVA v0.2 is intended to run in a networked environment, and is comprised of three main components:

### Control Node

Manages the assignment of tasks to other working nodes.  For more details view [here](ControlNode/README.md).

### Analyzer Node

The anaylzer node is a tf-serving 2.1 instance built from the official docker image.  For more details, view [here](https://www.tensorflow.org/tfx/serving/docker).

### Processor Node

The processor node is assigned videos by the Control Node.  It then handles making inference requests to the analyzer node, as well as pre/post processing and writing the results.  The rest of this document describes the Processor Node.

## Required Software Dependencies

SNVA has been tested using the following software stack:

- Ubuntu = 16.04
- Python >= 3.5
- TensorFlow = 2.1 (and its published dependencies)
- TensorBoard = 2.1
- FFmpeg >= 2.8
- websockets
- numpy
- scikit-image

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
  -et --modelname desired_model_name \
  -cnh controlodeHostOrIP \
  -l /path/to/your/desired/log/directory  \
  --modelsdirpath /path/to/your/model/directory \
  -msh analyzerHostOrIP \
  -ip /path/to/directory/containing/your/video/files \
  --writeinferencereports True
```

## To run using NVIDIA-Docker on Ubuntu (for a text file listing absolute paths to videos):

```shell
	sudo docker run \
    --runtime=nvidia 
    --mount type=bind,\
    src=/path/to/your/model/directory,dst=/usr/model \
    --mount type=bind,\
    src=/path/to/your/desired/output/directory,dst=/usr/output 
    --mount type=bind,\
    src=/path/to/directory/containing/your/vidoe/files,dst=/usr/videos 
    --mount type=bind,\
    src=/path/to/your/desired/log/directory,dst=/usr/logs\
    snva-processor -et -cpu -cnh controlnodeHoseOrIP -msh analzyerHostOrIP -wir true 
    --modelname desired_model_name
```

## Model directory structure

The model directory should contain subdirectories for each available model. The specific model to use is specified by the '--modelname' argument, which should match the name of one of these directories. 

This directory should also contain two nonstandard files. The first, class_names.txt, should be saved in the model directory itself. It will contain a list mapping numeric class id values to the appropriate string class name, in the format 'id:class_name'. Items should be separated by newlines. The processor will use this to parse model output.

The second file is the input_size.txt file, which should be saved in the subdirectory for each modelname. It will contain a single numeric value between 224 and 299, and indicates the square input dimension of the neural net.

## Usage

Flag | Short Flag | Properties | Description
:------:|:---------------:|:---------------------:|:-----------:
--batchsize|-bs|type=int, default=32|Number of concurrent neural net inputs
--binarizeprobs|-b|action=store_true|Round probs to zero or one. For distributions with two 0.5 values, both will be rounded up to 1.0
--classnamesfilepath|-cnfp||Path to the class ids/names text file
--numprocesses|-np|type=int, default=3|Number of videos to process at one time
--crop|-c|action=store_true|Crop video frames to [offsetheight, offsetwidth, targetheight, targetwidth]
--cropheight|-ch|type=int, default=320|y-component of bottom-right corner of crop
--cropwidth|-cw|type=int, default=474|x-component of bottom-right corner of crop
--cropx|-cx|type=int, default=2|x-component of top-left corner of crop
--cropy|-cy|type=int, default=0|y-component of top-left corner of crop
--deinterlace|-d|action=store_true|Apply de-interlacing to video frames during extraction
--extracttimestamps|-et|action=store_true|Crop timestamps out of video frames and map them to strings for inclusion in the output CSV
--gpumemoryfraction|-gmf|type=float, default=0.9|% of GPU memory available to this process
--inputpath|-ip|required=True|Path to a directory containing the video files to be processed
--ionodenamesfilepath|-ifp|Path to the io tensor names text file
--loglevel|-ll|default=info|Defaults to 'info'. Pass 'debug' or 'error' for verbose or minimal logging, respectively
--logmode|-lm|default=verbose|If verbose, log to file and console. If silent, log to file only
--logpath|-l|default=logs|Path to the directory where log files are stored
--logmaxbytes|-lmb|type=int|default=2**23|File size in bytes at which the log rolls over
--modelsdirpath|-mdp|default=models/work_zone_scene_detection|Path to the parent directory of model directories
--modelname|-mn|required=True|The subdirectory of modelsdirpath to use
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
--controlnodehost|-cnh|default=localhost:8080|Control Node, colon-separated hostname or IP and Port
--modelserverhost|-msh|default=0.0.0.0:8500|Tensorflow Serving Instance, colon-separated hostname or IP and Port
--processormode|-pm|default=workzone|Indicates what model pipeline to use: 'workzone', 'signalstate', or 'weather'
--writebbox|-bb|action=store_true|Create JSON files with raw bounding box coordinates when run in 'signalstate' mode


## Troubleshooting and Additional Considerations

If a timestamp cannot be interpreted, a -1 will be written in its place in the output CSV.

While inference speed has been observed to monotonically increase with batch size, it is important to not exceed the GPU's memory capacity. The SNVA app does not automatically determine the optimal batch size for maximum inference speed. It is best to discover the optimal batch size by testing the app on a small sample of videos (say ~15) starting at a relatively low batch size, then iteratively incrementing the batch size while monitoring GPU memory utilization (e.g. using the NVIDIA X Server Settings GUI app or nvidia-smi CLI app: nvidia-smi --query-compute-apps=process_name,pid,used_gpu_memory --format=csv) and also observing the cumulative analysis duration printed at the end of each run. GPU memory is set to be dynamically allocated, so one should monitor its usage over time to increase the chance of observing peak utilization.

When terminating the app using ctrl-c, there may be a delay while the app terminates gracefully.

When terminating the dockerized app, use ctrl-c to let the app terminate gracefully before invoking the nvidia-docker stop command (which actually shouldn't be needed).

Windows is not officially supported but may be used with minor code tweaks.

When using Docker, some extraneous C++ output is passed to the host machine's console that is not actually logged to file and is not intended to be seen. Consider this a bug and ignore it.
## License

[MIT](https://opensource.org/licenses/MIT)

## Reference

```
@article{SNVA2018,
  title={Detecting Work Zones in SHRP 2 NDS Videos Using Deep Learning Based Computer Vision},
  author={Abodo, Rittmuller, Sumner, Berthaume},
  journal={arXiv preprint arXiv:1811.04250},
  year={2018}
}
```
