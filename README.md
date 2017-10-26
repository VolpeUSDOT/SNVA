# TF-WRAPPER
#### Using Tensorflow & Inception to Detect Features in Video
## Version 2.0.2

An automation "wrapper" based on TF-SLIM to make it easy to detect various features in video using Tensorflow, FFMPEG, and various versions of the Inception neural network.

## Features

 - Ability to detect & train for various features in both images and video.
 - Can be trained using Inception V3, V4, and Resnet V2.
 - Runs under both Linux and Windows (Python 3.6.x).
 - Silly fast, even on CPU for detection. Using an GTX 1060 Nvidia GPU it can perform analysis of a two-hour video in less than 7 minutes even using 
 Inception Resnet V2.

## Requirements

You will need Tensorflow, FFMPEG, and Python 2.7.x or Python 3.6.x (Windows).

## Installation

No formal "installation" is required beyond making a copy of this directory on your local system.

```bash
$ git clone git@github.com:VolpeUSDOT/tf-wrapper.git
$ cd tf-wrapper
$ python run-tests.py
```

## Commands

### videoscan.py

Search video for features in video. Creates/overwrites `[videofilename]-results.csv`.

#### Simple example:

`videoscan.py` __`--modelpath` models/mymodel.pb `--labelpath` models\mylabelsfilename.txt `--reportpath` ..\example-reports
`--labelname` mylabel [myvideofile.avi]__

#### Complex example:

`videoscan.py` __`--modelpath` models/mymodel.pb `--labelpath` models\mylabelsfilename.txt `--reportpath` ..\example-reports
`--labelname` mylabel `--fps` 5 `--allfiles` `--outputclips` `--smoothing` 2 `--training` [/path/to/video/files]__

#### Additional Switches & Options

`--modelpath` Path to the tensorflow protobuf model file.
<br>`--labelpath` Path to the tensorflow model labels file.
<br>`--labelname` Name of primary label, used to trigger secondary model (if needed).
<br>`--reportpath` Path to the directory where reports/output are stored.
<br>`--temppath` Path to the directory where temporary files are stored.
<br>`--trainingpath` Path to the directory where detected frames for retraining are stored.
<br>`--smoothing` Apply a type of "smoothing" factor to detection results. (Range = 1-6)
<br>`--fps` Frames Per Second used to sample input video. The higher this number the slower analysis will go. (Default is 1 FPS)
<br>`--allfiles` Process all video files in the directory path.
<br>`--outputclips` Output results as video clips containing searched for labelname.
<br>`--training` Saves predicted frames for future model retraining.
<br>`--outputpadding` Number of seconds added to the start and end of created video clips.
<br>`--filter` Value used to filter on a label. [Depricated]
<br>`--keeptemp` Keep temporary extracted video frames stored in path specified by `--temppath`

#### Perform predictions on a single image.

`detection.py` __image.jpg__

#### Run accuracy tests against the Tensorflow model, creates/overwrites `test-results.txt`.

`run-tests.py`

## License

MIT
