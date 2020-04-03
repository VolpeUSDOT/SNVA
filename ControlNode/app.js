const WebSocket = require('ws');
const yargs = require('yargs');
const fs = require('fs');
const VideoManager = require('./videoPathManager.js');

// Length of time (in ms) to wait before running a status check on nodes
const statusCheckFreq = 300000;
// Length of time (in ms) a node has to respond to a status request before it is considered dead
const statusTimeoutLength = 600000;
// List of processor nodes currently active
var processorNodes = [];
// List of analyzer (tfserving) nodes
var analyzerNodes = [];
// Completed videos and their output files
var completed = {};
//TODO Make these cl params
var numAnalyzer = 2;
var procPerAnalyzer = -1;


const actionTypes = {
    con_success: "CONNECTION_SUCCESS",
    process: "PROCESS",
    stat_req: "STATUS_REQUEST",
    shutdown: "SHUTDOWN",
    req_video: "REQUEST_VIDEO",
    cease_req: "CEASE_REQUESTS",
    stat_rep: "STATUS_REPORT",
    complete: "COMPLETE",
    error: "ERROR"
};

console.log("Starting...");

// Configure command line arguments
const argv = yargs
    .option('paths', {
                    alias: 'p',
                    description: 'Path to a file containing a list of videos to process',
                    default: './videopaths.txt',
                    type: 'string'
                })
    .option('outputPath', {
                    alias: 'op',
                    description: 'Text file to write a list of processed videos and their output locations to',
                    default: './outputList.txt',
                    type: 'string'
                })
    .option('nodes', {
                alias: 'n',
                description: 'JSON File containing a list of nodes to use for analysis and processing',
                default: './nodes.json',
                type: 'string'
            })
    .option('analyzerCount', {
                alias: 'a',
                description: 'Number of analyzer nodes to start.  Will prioritize GPU enabled nodes.',
                default: 2,
                type: 'int'
            })
    .help()
    .alias('help', 'h')
    .argv;

console.log("Provided with path file: %s", argv.paths);
// Read paths from file into memory
VideoManager.readInputPaths(argv.paths);

numAnalyzer = argv.analyzerCount;

var rawNodes = fs.readFileSync(argv.nodes);
var nodeList = JSON.parse(rawNodes);

var gpuNodes = nodeList.filter(function(n) { return n.gpuEnabled == true;});
var cpuNodes = nodeList.filter(function(n) { return n.gpuEnabled != true;});

if (numAnalyzer >= nodeList.length) {
    console.log("Insufficient nodes provided to create %d analyzer nodes", numAnalyzer);
    process.exit();
}

// Start analyzer nodes
for (var i=0;i<numAnalyzer;i++) {
    var node = gpuNodes.pop();
    if (node === undefined)
        node = cpuNodes.pop();
    startAnalyzer(node.node);
}

// TODO Start Processor node
// TODO Initialize Logging
// TODO Record list of processed videos and write to file

const wws = new WebSocket.Server({
    port: 8080,
    path: '/registerProcess'
});

wws.on('connection', function connection(ws) {
    ws.on('message', function incoming(message) {
        console.log('Received: %s', message);
        parseMessage(message, ws);
    });
    ws.on('close', onSocketDisconnect(ws));
    initializeConnection(ws);
});

const statusInterval = setInterval(function checkStatus() {
    console.log("Status check");
    for (var ip in processorNodes) {
        var node = processorNodes[ip];
        if (node.statusRequested) {
            if (new Date().getTime() - node.statusRequested > statusTimeoutLength) {
                // TODO Handle a dead connection: kill old, start new, add in-progress video back into queue
                console.log("Connection with %s lost", ip);
            }
        } else {
            requestStatus(node.websocket);
        }
    }
}, statusCheckFreq);

function startAnalyzer(node) {
    // TODO Actually start an analyzer node via docker
    var analyzerInfo = {
        path: node,
        numVideos: 0
    };
    analyzerNodes.push(analyzerInfo);
}

function initializeConnection(ws) {
    var ip = ws._socket.remoteAddress;
    var timestamp = new Date().getTime();
    console.log("Connection opened with address: %s", ip);
    var socketConnection = {
        websocket: ws,
        started: timestamp,
        lastResponse: timestamp,
        videos: []
    };
    processorNodes[ip] = socketConnection;
    sendRequest({action: actionTypes.con_success}, ws);
}

// If a processor loses connection, clean up outstanding tasks
function onSocketDisconnect(ws) {
    return function(code, reason) {
        var ip = ws._socket.remoteAddress;
        console.log("WS at %s disconnected with Code:%s and Reason:%s", 
            ip, code, reason);
        processorNodes[ip].videos.forEach(function(video) {
            VideoManager.addVideo(video.path);
        });
        delete processorNodes[ip];
        if (processorNodes.length == 0)
            // TODO Start up new processors, or end.
            console.log("All processors disconnected.");
    };    
}

function parseMessage(message, ws) {
    var msgObj;
    var ip = ws._socket.remoteAddress;
    processorNodes[ip].lastResponse = new Date().getTime();
    try {
        msgObj = JSON.parse(message);
    } catch (e) {
        console.log("Invalid Input");
        return;
    }
    switch(msgObj.action) {
        case actionTypes.req_video:
            console.log("Video requested");
            sendNextVideo(ws);
            break;
        case actionTypes.stat_rep:
            console.log("Status Reported");
            processStatusReport(msgObj, ws);
            break;
        case actionTypes.complete:
            console.log("Task Complete");
            processTaskComplete(msgObj, ws);
            break;
        case actionTypes.error:
            console.log("Error Reported");
            handleProcError(msgObj, ws);
            break;
        default:
            console.log("Invalid Input");
            // TODO Determine how to handle bad input
    }
}

function sendNextVideo(ws) {
    var nextVideoPath = VideoManager.nextVideo();
    // If there is no 'next video', work may stop
    var requestMessage;
    if (nextVideoPath == null) {
        requestMessage = {
            action: actionTypes.cease_req,
        };
        sendRequest(requestMessage, ws);
        return;
    }
    var analyzer = getBalancedAnalyzer();
    var analyzerPath = "";
    if (analyzer != null) {
        analyzerPath = analyzer.path;
        analyzer.numVideos++;
    }

    var ip = ws._socket.remoteAddress;
    var videoInfo = {
        path: nextVideoPath,
        analyzer: analyzerPath
    };
    processorNodes[ip].videos.push(videoInfo);
    // TODO validate path is real?
    requestMessage = {
        action: actionTypes.process,
        analyzer: analyzer.path,
        path: nextVideoPath
    };
    sendRequest(requestMessage, ws);
}

function processStatusReport(msg, ws) {
    // TODO Handle status report
    console.log("Status Reported: %s", msg);
    var ip = ws._socket.remoteAddress;
    delete processorNodes[ip].statusRequested;
}

function processTaskComplete(msgObj, ws) {
    var ip = ws._socket.remoteAddress;
    var video = msgObj.video;
    if (video == null) {
        // TODO Handle malformed input
        return;
    }
    var index = processorNodes[ip].videos.findIndex((videoItem) => videoItem.path == video);
    if (index == -1) {
        // Video path not assigned to this ws
        // TODO Handle malformed input
        return;
    }
    var analyzerPath = processorNodes[ip].videos[index].analyzer;
    // Decrement our video counter
    for (var analyzer of analyzerNodes) {
        if (analyzer.path == analyzerPath) {
            analyzer.numVideos--;
            break;
        }
    }
    processorNodes[ip].videos.splice(index, 1);
    var outputPath = msgObj.output;
    if (outputPath == null)
        outputPath = "Not Reported";
    completed[video] = outputPath;
    checkProcessorComplete(ws);
}

function checkProcessorComplete(ws) {
    if (!VideoManager.isComplete())
        return;
    var ip = ws._socket.remoteAddress;
    if (processorNodes[ip].videos.length == 0)
        shutdownProcessor(ws);
}

function shutdownProcessor(ws) {
    var msg = {
        action: actionTypes.shutdown
    };
    sendRequest(msg, ws);
    var ip = ws._socket.remoteAddress;
    delete processorNodes[ip];
    if (processorNodes.length == 0)
        shutdownControlNode();
}

function shutdownControlNode() {
    var output = fs.openSync(argv.outputPath, "w");
    Object.keys(completed).forEach(e => fs.writeSync(output, e + ": " + completed[e] + "\n"));
    fs.closeSync(output);
    process.exit();
}

function handleProcError(errorMsg, ws) {
    if (errorMsg.description != null)
        console.log("An error occured: %s", errorMsg.description);
    console.log("An error occured: Cause unknown");
    // TODO determine potential errors/behavior in each case
        // Video not found
        // No analyzer found
}

function requestStatus(ws) {
    // TODO Determine if it is suffient to simply ping/pong here?
    var msg = {
        action: actionTypes.stat_req
    };
    sendRequest(msg, ws);
    var ip = ws._socket.remoteAddress;
    processorNodes[ip].statusRequested = new Date().getTime();
}

function sendRequest(msgObj, ws) {
    ws.send(JSON.stringify(msgObj));
}

// Return the analyzer node with the fewest videos assigned
function getBalancedAnalyzer() {
    if (analyzerNodes.length == 0)
        return null;
    var min = analyzerNodes[0];
    analyzerNodes.forEach(function(a) {
        if (a.numVideos < min.numVideos)
            min = a;
    });
    return min;
}