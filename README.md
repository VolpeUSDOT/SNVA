# SHRP2 NDS Video Analytics (SNVA)

This repository houses the SNVA application and additional code used to develop the computer vision models at the core of SNVA. Model development code is based on TensorFlow-Slim.


## Motivation



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

## System Requirements

SNVA is intended to run on systems with NVIDIA GPUs, but can also run in a CPU-only mode. SNVA runs ~10x faster on a single NVIDIA GeForce GTX 1080 Ti together with a 3.00GHz 10-core Intel Core i7-6950X CPU than it does on the 10-core CPU alone. For a system with N GPUs, SNVA will process N videos concurrently, but is not (at this time) designed to distribute the processing of a single video across multiple GPUs. Inference speeds depend on the particular CNN architecture used to develop the model. When tested on two GPUs against ~32,000,000 video frames spanning ~1,350 videos, InceptionV3 inferred class labels at ~860 fps, whereas MobilenetV2 operated at ~1520 fps, taking 10.75 and 6 hours to complete, respectively.


## Installation



## To run using NVIDIA-Docker on Ubuntu:

sudo nvidia-docker run \
  --mount type=bind, \
    src=/path/to/your/desired/video_file/source/directory,dst=/media/input \
  --mount type=bind, \
    src=/path/to/your/desired/csv_file/destination/directory,dst=/media/output \
  --mount type=bind, \
    src=/path/to/your/desired/log_file/destination/directory,dst=/media/logs \
  volpeusdot/snva \
  --inputpath /media/input --outputpath /media/output --logspath /media/logs \
  --modelname inception_v3 --batchsize 32 --smoothprobs --binarizeprobs


## To run in an ordinary Ubuntu environment:

python3 snva.py
  --inputpath /media/input --outputpath /media/output --logspath /media/logs \
  --modelname inception_v3 --batchsize 32 --smoothprobs --binarizeprobs


## Additional Usage and Troubleshooting

While inference speed has been observed to monotonically increase with batch size, it is important to not exceed the GPU's memory capacity. The SNVA app does not currently manage memory utilization. It is best to discover the optimal batch size by starting to run for a breif period at a relatively low batch size, then iteratively incrementing the batch size while monitoring GPU memory utilization (e.g. using the nvidia-smi CLI app or NVIDIA X Server Settings GUI app).

When terminating the app using ctrl-c, there may be a delay while the app terminates gracefully.

When terminating the dockerized app, use ctrl-c to let the app terminate gracefully before invoking the nvidia-docker stop command.

Windows is not officially supported but may be used with minor code tweaks.


## License

MIT
