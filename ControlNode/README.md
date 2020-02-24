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

The list of videos should contain a set of paths of videos to process separated by newlines.

Once the Control Node has started, a WebSocket connection may be opened by a processor node at path "/registerProcess".  Once a processor is registered, the control node will begin to issue requests to process videos.  Once the provided input list is exhausted, the control node will issue shutdown commands to all processors and stop.
