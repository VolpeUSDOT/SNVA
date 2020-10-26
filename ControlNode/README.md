# SNVA Control Node

The Control Node for the SNVA 2.0 Architecture Design. The control node is responsible for monitoring and assigning work to processor nodes.

## Requirements

- Node.js 12.15.0 LTS

## Installation

Download the application and in its directory run
```
npm install 
```
to download all required dependencies

## How to Use

To start the control node use the following command:

```
node app.js -p /path/to/list/of/videos.txt
```

To start via docker use:
```
sudo docker run --mount type=bind,src=/path/to/list/of/videos.txt,dst=/usr/config/Paths.txt --mount type=bind,src=/path/to/list/of/Nodes.json,dst=/usr/config/Nodes.json --mount type=bind,src=/path/to/log/directory,dst=/usr/logs -d bsumner/control-node --paths /usr/config/Paths.txt --nodes /usr/config/Nodes.json --logDir /usr/logs -a 1
```

The list of videos should contain a set of paths of videos to process separated by newlines.

Once the Control Node has started, a WebSocket connection may be opened by a processor node at path "/registerProcess".  Once a processor is registered, the control node will begin to issue requests to process videos.  Once the provided input list is exhausted, the control node will issue shutdown commands to all processors and stop.


Flag | Short Flag | Properties | Description
:------:|:---------------:|:---------------------:|:-----------:
--input|-i|type=string, default='./videopaths.txt'|Text File containing a list of video paths separated by newlines
--nodes|-n|type=string, default='./nodes.json| JSON file containing a list of nodes to use as analyzers or processors. Should be an array of objects formatted as: {"node":"nodeLocation", "gpuEnabled":"true\|false"}. Functionality based on this argument is incomplete.
--analyzerCount|-a|type=int, default=2|Number of analyzer nodes to generate. Functionality based on this argument is incomplete.
--logDir|-l|type=string, default=./logs|Directory to save log files.
--port|-p|type=int, default=8081|Port which server should listen on.
