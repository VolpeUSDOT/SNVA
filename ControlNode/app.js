const WebSocket = require('ws');
const yargs = require('yargs');
const fs = require('fs');
const readline = require('readline');

// Length of time (in ms) to wait before running a status check on nodes
const statusCheckFreq = 300000;
// Length of time (in ms) a node has to respond to a status request before it is considered dead
const statusTimeoutLength = 600000;
// List of processor nodes currently active
var processorNodes = [];
// List of file paths to process
var toProcess = [];
// Completed videos and their output files
var completed = {};

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
    .help()
    .alias('help', 'h')
    .argv;

console.log("Provided with path file: %s", argv.paths);
// Read paths from file into memory
readline.createInterface({
    input: fs.createReadStream(argv.paths),
    terminal: false
}).on('line', function(line) {
    toProcess.push(line);
});

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
            //TODO Handle Error response
            break;
        default:
            console.log("Invalid Input");
            // TODO Determine how to handle bad input
    }
}

function processTaskComplete(msgObj, ws) {
    var ip = ws._socket.remoteAddress;
    var video = msgObj.video;
    if (video == null) {
        // TODO Handle malformed input
        return;
    }
    var index = processorNodes[ip].videos.indexOf(video);
    if (index == -1) {
        // Video path not assigned to this ws
        // TODO Handle malformed input
        return;
    }
    processorNodes[ip].videos.splice(index, 1);
    var outputPath = msgObj.output;
    if (outputPath == null)
        outputPath = "Not Reported";
    completed[video] = outputPath;
    checkProcessorComplete(ws);
}

function sendNextVideo(ws) {
    var nextVideoPath = nextVideo();
    // If there is no 'next video', work may stop
    if (nextVideoPath == null) {
        var requestMessage = {
            action: actionTypes.cease_req,
        };
        sendRequest(requestMessage, ws);
        return;
    }
    var ip = ws._socket.remoteAddress;
    processorNodes[ip].videos.push(nextVideoPath);
    // TODO validate path is real?
    var requestMessage = {
        action: actionTypes.process,
        path: nextVideoPath,
    };
    sendRequest(requestMessage, ws);
}

function handleProcError(errorMsg, ws) {
    if (errorMsg.description != null)
        console.log("An error occured: %s", errorMsg.description);
    console.log("An error occured: Cause unknown");
    // TODO determine potential errors/behavior in each case
        // Video not found
        // No analyzer found
}

function processStatusReport(msg, ws) {
    // TODO Handle status report
    console.log("Status Reported: %s", msg);
    var ip = ws._socket.remoteAddress;
    delete processorNodes[ip].statusRequested;
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

function checkProcessorComplete(ws) {
    if (toProcess.length > 0)
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
    //fs.writeFileSync("outputData.txt", JSON.stringify(completed), 'utf8');
    Object.keys(completed).forEach(e => fs.writeSync(output, e + ": " + completed[e] + "\n"));
    fs.closeSync(output);
    process.exit();
}

function sendRequest(msgObj, ws) {
    ws.send(JSON.stringify(msgObj));
}

function nextVideo() {
    if (toProcess.length > 0)
        return toProcess.pop();
    return null;
}