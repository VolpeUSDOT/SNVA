const WebSocket = require('ws');
const yargs = require('yargs');
const fs = require('fs');
const path = require('path');
const winston = require('winston');
const http = require('http');
const url = require('url');
var express = require('express')
const VideoManager = require('./modules/videoPathManager.js');
const DockerManager = require('./modules/dockerManager.js');

// Length of time (in ms) to wait before running a status check on nodes
const statusCheckFreq = 300000;
// Length of time (in ms) a node has to reconnect before it is considered dead
const reconnectTimer = 600000;
// Length of time a processor has to confirm it received a process request
const processTimer = 60000;
// List of processor nodes currently active
var processorNodes = {};
// List of timeouts from disconnects - we can't store in the above since they don't serialize
var timeouts = {};
// List of analyzer (tfserving) nodes
var analyzerNodes = [];
// Completed videos and their output files
var completed = {};
// Sent requests, pending acknowledgment from processor, and their timeouts
var pending = {};
// Number of analyzer nodes to create
var numAnalyzer = 2;
// Number of processor ndoes
var procPerAnalyzer = -1;
// Processor ID - we just count up
var nextProcessorId = 0;

const actionTypes = {
    con_success: "CONNECTION_SUCCESS",
    process: "PROCESS",
    req_rec: "REQUEST_RECEIVED",
    stat_req: "STATUS_REQUEST",
    shutdown: "SHUTDOWN",
    req_video: "REQUEST_VIDEO",
    cease_req: "CEASE_REQUESTS",
    resume_req: "RESUME_REQUESTS",
    stat_rep: "STATUS_REPORT",
    complete: "COMPLETE",
    error: "ERROR"
};

// Configure command line arguments
const argv = yargs
    .option('inputFile', {
                    alias: 'i',
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
                type: 'string'
            })
    .option('analyzerCount', {
                alias: 'a',
                description: 'Number of analyzer nodes to start.  Will prioritize GPU enabled nodes.',
                default: 2,
                type: 'int'
            })
    .option('logDir', {
                alias: 'l',
                description: 'Log directory. If run on a distributed environment, this should be a network location accessible by all nodes',
                default: './logs',
                type: 'string'
            })
    .option('port', {
        alias: 'p',
        description: 'Port which server should listen on',
        default: 8081,
        type: 'int'
    })
    .help()
    .alias('help', 'h')
    .argv;

const logger = winston.createLogger({
    level: 'debug',
    format: winston.format.combine(
        winston.format.timestamp({
          format: 'YYYY-MM-DD hh:mm:ss A ZZ'
        }),
        winston.format.json()
      ),
    defaultMeta: { service: 'user-service' },
    transports: [
        //
        // - Write all logs with level `error` and below to `error.log`
        // - Write all logs with level `info` and below to `combined.log`
        //
        new winston.transports.File({ filename: argv.logDir + '/error.log', level: 'error'}),
        new winston.transports.File({ filename: argv.logDir + '/combined.log'})
    ],
    exceptionHandlers: [
        new winston.transports.File({ filename: argv.logDir + '/exceptions.log', timestamp: true, maxsize: 1000000 })
    ]
    });

logger.info("Starting control node...");

logger.info("Provided with path file: " + argv.inputFile);
logger.info("Provided with node file: " + argv.nodes);
// Read paths from file into memory
VideoManager.readInputPaths(argv.inputFile);

numAnalyzer = argv.analyzerCount;

var nodeList = [];
// Placeholder functionality to automatically start other nodes; incomplete
if (argv.nodes != null) {
    var rawNodes = fs.readFileSync(argv.nodes);
    nodeList = JSON.parse(rawNodes);

    var gpuNodes = nodeList.filter(function(n) { return n.gpuEnabled == true;});
    var cpuNodes = nodeList.filter(function(n) { return n.gpuEnabled != true;});

    if (numAnalyzer >= nodeList.length) {
        logger.error("Insufficient nodes provided to create " + numAnalyzer + " analyzer nodes");
        process.exit();
    }

    // Start analyzer nodes
    for (var i=0;i<numAnalyzer;i++) {
        var node = gpuNodes.pop();
        if (node === undefined)
            node = cpuNodes.pop();
        startAnalyzer(node.node);
    }

    // Create processors
    var remaining = gpuNodes.concat(cpuNodes);
    var numToCreate;
    if (procPerAnalyzer == -1)
        numToCreate = remaining.length;
    else
        numToCreate = numAnalyzer * procPerAnalyzer;
    for (var i=0;i<numToCreate;i++) {
        var node = remaining.pop();
        if (node == null)
            return;
        startProcessor(node.node);
    }
}

var app = express()

app.use(express.static('web'))

const server = http.createServer(app);

const wws = new WebSocket.Server({
    noServer: true,
    path: '/registerProcess'
});

const guiWws = new WebSocket.Server({
    noServer: true,
    path: '/snvaStatus'
});

guiWws.on('connection', function guiConn(ws) {
    // One-way connection; we don't need to accept any messages
    // Client should handle reconnect if needed
    ws.send(getGuiInfo());
});

wws.on('connection', function connection(ws, req) {
    ws.on('message', function incoming(message) {
        logger.info('Received from ' + ws.id + ': ' + message);
        parseMessage(message, ws);
    });

    ws.on("error", function(error) {
        // Manage error here
        logger.error(error);
    });

    ws.on('pong', function test(ms) {
        var id = ws.id;
        var diff = parseInt(new Date().getTime()) - parseInt(ms.toString());
        logger.info("Latency to " + id + ": " + diff + " ms");
    });
    
    const parameters = url.parse(req.url, true);
    var id = parameters.query.id;

    var date = new Date().getTime();
    ws.ping(new Date().getTime());
    ws.on('close', onSocketDisconnect(ws));
    initializeConnection(ws, id);
});

server.on('upgrade', function upgrade(request, socket, head) {
    const pathname = url.parse(request.url).pathname;
    if (pathname === '/snvaStatus') {
      guiWws.handleUpgrade(request, socket, head, function done(ws) {
        guiWws.emit('connection', ws, request);
      });
    } else if (pathname === '/registerProcess') {
      wws.handleUpgrade(request, socket, head, function done(ws) {
        wws.emit('connection', ws, request);
      });
    } else {
      socket.destroy();
    }
});
  
server.listen(argv.port);

function getGuiInfo() {
    return JSON.stringify({
        'videosRemaining': VideoManager.getCount(),
        'processorInfo': processorNodes
    });
}

function broadcastStatus() {
    guiWws.clients.forEach((client) => {
        client.send(getGuiInfo());
    });
}

function startAnalyzer(node) {
    // TODO Actually start an analyzer node via docker
    //DockerManager.startAnalyzer(node);
    var analyzerInfo = {
        path: node,
        numVideos: 0
    };
    analyzerNodes.push(analyzerInfo);
}

function startProcessor(node) {
    var logDir = argv.logDir + "/" + node;
    logDir = path.normalize(logDir);
    //if (!fs.existsSync(logDir))
        //fs.mkdirSync(logDir, {recursive: true});
    // TODO Start processor in docker, passing logDir as arg
}

function initializeConnection(ws, id) {
    if (id != null) {
        onReconnect(ws, id);
    } else {
        var ip = ws._socket.remoteAddress;
        var timestamp = new Date().getTime();
        logger.info("Connection opened with address: " + ip);
        var id = "Processor" + (nextProcessorId++).toString();
        logger.info("Assigned ID: " + id);
        ws.id = id;
        var socketConnection = {
            started: timestamp,
            lastResponse: timestamp,
            videos: [],
            ip: ip
        };
        processorNodes[id] = socketConnection;
        broadcastStatus();
    }
    sendRequest({action: actionTypes.con_success, id: id}, ws);
}

function onReconnect(ws, id) {
    ws.id = id;
    logger.info(id + " reconnected");
    clearTimeout(timeouts[id]);
    processorNodes[id].timeoutId = null;
    processorNodes[id].disconnect = false;
    broadcastStatus();
}

// If a processor loses connection, clean up outstanding tasks
function onSocketDisconnect(ws) {  
    return function(code, reason) {
        var id = ws.id;
        processorNodes[id].disconnect = true;
        logger.debug("WS " + id + " disconnected with Code:" + code + " and Reason:" + reason);
        timeouts[id] = setTimeout(onReconnectFail(id), reconnectTimer);
        broadcastStatus();
    };
}

function onReconnectFail(id) {
    return function() {
        logger.debug("WS " + id + " failed to reconnect");
        processorNodes[id].videos.forEach(function(video) {
            VideoManager.addVideo(video.path);
            removeVideoFromProcessor(id, video.path);
        });
        processorNodes[id].closed = true;
        if (!haveActiveProcessors())
            // TODO Start up new processors, or end.
            logger.debug("All processors disconnected.");
        delete timeouts[id];
        broadcastStatus();
    };
}

function haveActiveProcessors() {
    for (var proc in processorNodes) {
        if (!processorNodes[proc].closed)
            return true;
    }
    return false;
}

function parseMessage(message, ws) {
    var msgObj;
    var id = ws.id;
    logger.debug("Parsing message from " + id);
    processorNodes[id].lastResponse = new Date().getTime();
    // Sometimes a disconnect event fires server-side erroneously and the client continues to send messages 
    // In that case, log it and mark as reconnected
    if (processorNodes[id].disconnect) {
        logger.warn("Erroneous disconnect reported for " + id);
        onReconnect(ws, id);
    }
    if (processorNodes[id].closed) {
        logger.warn("Message received from closed processor " + id + ": requesting shutdown");
        shutdownProcessor(ws)
        return
    }
    
    try {
        msgObj = JSON.parse(message);
    } catch (e) {
        logger.debug("Invalid JSON Sent by " + id);
        return;
    }
    switch(msgObj.action) {
        case actionTypes.req_video:
            logger.info("Video requested by " + id);
            sendNextVideo(ws);
            break;
        case actionTypes.req_rec:
            logger.info("Confirm request received by " + id);
            processReceived(msgObj, ws);
            break;
        case actionTypes.stat_rep:
            logger.info("Status Reported by " + id);
            processStatusReport(msgObj, ws);
            break;
        case actionTypes.complete:
            logger.info("Task Complete by " + id);
            processTaskComplete(msgObj, ws);
            break;
        case actionTypes.error:
            logger.info("Error Reported by " + id);
            handleProcError(msgObj, ws);
            break;
        default:
            logger.info("Invalid Input by " + id);
            // TODO Determine how to handle bad input
    }
    broadcastStatus();
}

function sendNextVideo(ws) {
    var nextVideoPath = VideoManager.nextVideo();
    var id = ws.id;
    // If there is no 'next video', work may stop
    var requestMessage;
    if (nextVideoPath == null) {
        logger.info("No videos remaining; telling " + id + " to cease requests");
        requestMessage = {
            action: actionTypes.cease_req,
        };
        processorNodes[id].idle = true;
        sendRequest(requestMessage, ws);
        return;
    }
    logger.info("Sending video to " + id + ": " + nextVideoPath);
    var analyzer = getBalancedAnalyzer();
    var analyzerPath = "";
    if (analyzer != null) {
        analyzerPath = analyzer.path;
        analyzer.numVideos++;
    }

    var videoInfo = {
        path: nextVideoPath,
        analyzer: analyzerPath,
        //time: getTime()
    };
    pending[nextVideoPath] = setTimeout(processTimeout(nextVideoPath, ws), processTimer);
    processorNodes[id].videos.push(videoInfo);
    // TODO validate path is real?
    requestMessage = {
        action: actionTypes.process,
        analyzer: analyzerPath,
        path: nextVideoPath
    };
    sendRequest(requestMessage, ws);
}

function processReceived(msgObj, ws) {
    logger.debug("Clearing timeout for " + msgObj.video);
    clearTimeout(pending[msgObj.video]);
    pending[msgObj.video] = null;
}

function processTimeout(video, ws) {
    var id = ws.id;
    return function() {
        // Processor did not acknoweldge receipt of video
        // Remove video from processor
        logger.info("Connection " + id + " did not verify receipt of request to process " + video);
        removeVideoFromProcessor(id, video);
        // Return it to the queue, and clean up the pending request
        VideoManager.addVideo(video);
        pending[video] = null;
    }
}

function processStatusReport(msg, ws) {
    // TODO Handle status report
    logger.info("Status Reported: " + msg);
    var id = ws.id;
    delete processorNodes[id].statusRequested;
}

function processTaskComplete(msgObj, ws) {
    var id = ws.id;
    var video = msgObj.video;
    if (video == null) {
        logger.error("Completed video not specified by " + id);
        return;
    }
    removeVideoFromProcessor(id, video);
    var outputPath = msgObj.output;
    if (outputPath == null)
        outputPath = "Not Reported";
    completed[video] = outputPath;
    checkProcessorComplete(ws);
}

function removeVideoFromProcessor(id, video) {
    var index = processorNodes[id].videos.findIndex((videoItem) => videoItem.path == video);
    if (index == -1) {
        // Video path not assigned to this ws
        logger.error("Node reported on video it was not assigned: " + id);
        // TODO Handle malformed input
        return;
    }
    var analyzerPath = processorNodes[id].videos[index].analyzer;
    // Decrement our video counter
    for (var analyzer of analyzerNodes) {
        if (analyzer.path == analyzerPath) {
            analyzer.numVideos--;
            break;
        }
    }
    //logger.info("%s processed video %s in %d ms", ip, video, getTime() - processorNodes[id].videos[index].time);
    processorNodes[id].videos.splice(index, 1);
}

function checkProcessorComplete(ws) {
    var id = ws.id;
    if (!VideoManager.isComplete()) {
        // If we still have tasks in queue, and this processor has ceased requests, make it resume
        if (processorNodes[id].idle === true && processorNodes[id].closed != true) {
            var msg = {
                action: actionTypes.resume_req
            };
            sendRequest(msg, ws);
        }
        return;
    }
    if (processorNodes[id].videos.length == 0)
        shutdownProcessor(ws);
}

function shutdownProcessor(ws) {
    var msg = {
        action: actionTypes.shutdown
    };
    sendRequest(msg, ws);
    var id = ws.id;
    logger.info("Requesting shutdown from " + id);
    processorNodes[id].closed = true;
    if (!haveActiveProcessors())
        shutdownControlNode();
}

function shutdownControlNode() {
    broadcastStatus();
    var output = fs.openSync(argv.outputPath, "w");
    Object.keys(completed).forEach(e => fs.writeSync(output, e + ": " + completed[e] + "\n"));
    fs.closeSync(output);
    logger.info("Shutting down");
    // Logger will need time to finish, and does not seem to properly fire an event when it does
    // TODO find a better solution to this
    setTimeout(process.exit, 5000);
}

function handleProcError(errorMsg, ws) {
    if (errorMsg.description != null)
        logger.debug("An error occured: " + errorMsg.description);
    logger.debug("An error occured: Cause unknown");
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
    var id = ws.id;
    processorNodes[id].statusRequested = new Date().getTime();
}

function sendRequest(msgObj, ws) {
    var id = ws.id;
    logger.debug("Sending request to " + id + ": " + JSON.stringify(msgObj));
    ws.send(JSON.stringify(msgObj), {}, function() {logger.debug("Request Sent");});
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