# SNVA Control Node

The Control Node for the SNVA v0.2.2 Architecture Design. The control node is responsible for monitoring and assigning work to processor nodes.

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
node app.js -i /path/to/list/of/videos.txt
```

To start via docker use:
```
sudo docker run --mount type=bind,src=/path/to/list/of/videos.txt,dst=/usr/config/Input.txt  --mount type=bind,src=/path/to/log/directory,dst=/usr/logs --mount type=bind,src=/path/to/output/directory,dst=/usr/output -d control-node --inputFile /usr/config/Input.txt --logDir /usr/logs --outputPath /usr/output/outputList.txt
```

The list of videos should contain a list of video file names within the directory passed to your processor node, separated by newlines.

Once the Control Node has started, a WebSocket connection may be opened by a processor node at path "/registerProcess".  Once a processor is registered it will begin to request tasks from the control node, which will assign it videos from the provided list.  Once the provided input list is exhausted, the control node will wait for all processors to finish work, issue shutdown commands to them, and finally stop itself. As processors report vidoes complete, the control node will update an output file which contains the names of the processed videos along with the location of any reports produced by the processor node.


Flag | Short Flag | Properties | Description
:------:|:---------------:|:---------------------:|:-----------:
--inputFile|-i|type=string, default='./videopaths.txt'|Text File containing a list of video paths separated by newlines
--outputPath|-op|type=string, default='./outputList.txt'|Text file to write a list of processed videos and their output locations to
--nodes|-n|type=string, default='./nodes.json| JSON file containing a list of nodes to use as analyzers or processors. Should be an array of objects formatted as: {"node":"nodeLocation", "gpuEnabled":"true\|false"}. Functionality based on this argument is incomplete.
--analyzerCount|-a|type=int, default=2|Number of analyzer nodes to generate. Functionality based on this argument is incomplete.
--logDir|-l|type=string, default=./logs|Directory to save log files.
--port|-p|type=int, default=8081|Port which server should listen on.

## GUI

A web-based monitoring GUI is available at \<deploymentIp\>:\<port\>/snvaStatus. This will display the status of all connected Processor nodes, as well as the number of videos remaining in the processing queue. The page will automatically update as the status changes.
