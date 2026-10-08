const WebSocket = require('ws');
const yargs = require('yargs');
const fs = require('fs');
const path = require('path');
const winston = require('winston');
const https = require('https');
const tls = require('tls');
const crypto = require('crypto');
const url = require('url');
var express = require('express');
const VideoManager = require('./modules/videoPathManager.js');
const DockerManager = require('./modules/dockerManager.js');

// Length of time (in ms) to wait before running a status check on nodes
const statusCheckFreq = 300000;
// Length of time (in ms) a node has to reconnect before it is considered dead
const reconnectTimer = 600000;
// Length of time a processor has to confirm it received a process request
const processTimer = 60000;
// List of processor nodes currently active
var processorNodes = Object.create(null);
// Credentials and live sockets must never be included in GUI status responses.
var connections = Object.create(null);
// List of timeouts from disconnects - we can't store in the above since they don't serialize
var timeouts = Object.create(null);
// List of analyzer (tfserving) nodes
var analyzerNodes = [];
// Completed videos and their output files
var completed = Object.create(null);
// Keep per-attempt receipts private for idempotent completion recovery during this run.
var completionReceipts = Object.create(null);
// Sent requests, pending acknowledgment from processor, and their timeouts
var pending = Object.create(null);
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
    complete_accepted: "COMPLETE_ACCEPTED",
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
    .option('host', {
        description: 'Interface on which the HTTPS server should listen',
        default: '0.0.0.0',
        type: 'string'
    })
    .option('tlsCert', {
        description: 'PEM server certificate file',
        type: 'string',
        demandOption: true
    })
    .option('tlsKey', {
        description: 'PEM server private key file',
        type: 'string',
        demandOption: true
    })
    .option('tlsCa', {
        description: 'PEM CA certificate bundle trusted for processor and GUI client certificates',
        type: 'string',
        demandOption: true
    })
    .option('allowedOrigin', {
        description: 'Exact canonical HTTPS origin allowed for browser WebSocket connections',
        type: 'string',
        demandOption: true
    })
    .help()
    .alias('help', 'h')
    .argv;

// Validate origin and TLS before loading work, creating logs, or starting any nodes.
const allowedOrigin = validateAllowedOrigin(argv.allowedOrigin);
const tlsOptions = {
    cert: readTlsFile('tlsCert'),
    key: readTlsFile('tlsKey'),
    ca: readTlsFile('tlsCa'),
    minVersion: 'TLSv1.2',
    requestCert: true,
    rejectUnauthorized: true
};
var caText = tlsOptions.ca.toString();
var caCertificates = caText.match(/-----BEGIN CERTIFICATE-----[\s\S]*?-----END CERTIFICATE-----/g);
if (!caCertificates || caText.replace(/-----BEGIN CERTIFICATE-----[\s\S]*?-----END CERTIFICATE-----/g, '').trim())
    throw new Error('--tlsCa must contain PEM CA certificates');
caCertificates.forEach(function(cert) { crypto.createPublicKey(cert); });
tls.createSecureContext(tlsOptions);
if (typeof argv.host !== 'string' || !argv.host.trim() ||
        !Number.isInteger(argv.port) || argv.port < 0 || argv.port > 65535)
    throw new Error('Invalid --host or --port');

function validateAllowedOrigin(origin) {
    if (typeof origin !== 'string')
        throw new Error('--allowedOrigin must be an exact canonical HTTPS origin');
    var parsed = new url.URL(origin);
    if (parsed.protocol !== 'https:' || !parsed.hostname || parsed.username || parsed.password ||
            parsed.origin !== origin)
        throw new Error('--allowedOrigin must be an exact canonical HTTPS origin without a path, query, or fragment');
    return origin;
}

function readTlsFile(option) {
    var filename = argv[option];
    if (typeof filename !== 'string' || !filename.trim() || !fs.statSync(filename).isFile())
        throw new Error('--' + option + ' must name a readable file');
    var contents = fs.readFileSync(filename);
    if (!contents.length)
        throw new Error('--' + option + ' must not be empty');
    return contents;
}

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

var app = express();

app.use(express.static('web'));

const server = https.createServer(tlsOptions, app);

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
    ws.on('close', onSocketDisconnect(ws));
    if (initializeConnection(ws, parameters.query.id, parameters.query.reconnectToken,
            clientFingerprint(req.socket)))
        ws.ping(new Date().getTime().toString());
});

server.on('upgrade', function upgrade(request, socket, head) {
    if (!socket.authorized || !clientFingerprint(socket)) {
        socket.destroy();
        return;
    }
    const pathname = url.parse(request.url).pathname;
    var origin = request.headers.origin;
    var originHeaders = 0;
    (request.rawHeaders || []).forEach(function(header, index) {
        if (index % 2 === 0 && header.toLowerCase() === 'origin')
            originHeaders++;
    });
    if (originHeaders > 1 || (origin !== undefined &&
            (typeof origin !== 'string' || origin !== allowedOrigin)) ||
            (pathname === '/snvaStatus' && origin !== allowedOrigin)) {
        socket.destroy();
        return;
    }
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
  
server.on('error', function(error) {
    logger.error('HTTPS listener failed: ' + error.message);
    process.exit(1);
});
server.listen(argv.port, argv.host);

function getGuiInfo() {
    return JSON.stringify({
        'videosRemaining': VideoManager.getCount(),
        'processorInfo': processorNodes
    });
}

function broadcastStatus() {
    guiWws.clients.forEach((client) => {
        if (client.readyState === WebSocket.OPEN)
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

function clientFingerprint(socket) {
    if (!socket || !socket.authorized || typeof socket.getPeerCertificate !== 'function')
        return null;
    var cert = socket.getPeerCertificate();
    return cert && cert.raw ? crypto.createHash('sha256').update(cert.raw).digest('hex') : null;
}

function isCurrentSocket(ws) {
    return typeof ws.id === 'string' &&
        Object.prototype.hasOwnProperty.call(connections, ws.id) &&
        Object.prototype.hasOwnProperty.call(processorNodes, ws.id) && connections[ws.id].socket === ws;
}

function initializeConnection(ws, id, reconnectToken, fingerprint) {
    if (typeof fingerprint !== 'string' ||
            (id === undefined && reconnectToken !== undefined)) {
        ws.close(1008, 'Invalid reconnect credentials');
        return false;
    }
    if (id !== undefined) {
        if (!onReconnect(ws, id, reconnectToken, fingerprint)) {
            ws.close(1008, 'Invalid reconnect credentials');
            return false;
        }
    } else {
        var ip = ws._socket.remoteAddress;
        var timestamp = new Date().getTime();
        logger.info("Connection opened with address: " + ip);
        id = "Processor" + (nextProcessorId++).toString();
        logger.info("Assigned ID: " + id);
        ws.id = id;
        var socketConnection = {
            started: timestamp,
            lastResponse: timestamp,
            videos: [],
            ip: ip
        };
        processorNodes[id] = socketConnection;
        connections[id] = {
            socket: ws,
            reconnectToken: crypto.randomBytes(32).toString('hex'),
            fingerprint: fingerprint
        };
        broadcastStatus();
    }
    sendRequest({action: actionTypes.con_success, id: id,
        reconnectToken: connections[id].reconnectToken}, ws);
    if (processorNodes[id].closed)
        sendRequest({action: actionTypes.shutdown}, ws);
    return true;
}

function onReconnect(ws, id, reconnectToken, fingerprint) {
    if (typeof id !== 'string' || !/^Processor(?:0|[1-9][0-9]*)$/.test(id) ||
            typeof reconnectToken !== 'string' || !/^[a-f0-9]{64}$/.test(reconnectToken) ||
            !Object.prototype.hasOwnProperty.call(processorNodes, id) ||
            !Object.prototype.hasOwnProperty.call(connections, id))
        return false;
    var previous = connections[id];
    if ((processorNodes[id].closed && !previous.terminal) || previous.fingerprint !== fingerprint ||
            !crypto.timingSafeEqual(Buffer.from(previous.reconnectToken, 'hex'), Buffer.from(reconnectToken, 'hex')))
        return false;
    ws.id = id;
    connections[id] = {socket: ws, reconnectToken: previous.reconnectToken, fingerprint: fingerprint,
        terminal: previous.terminal};
    logger.info(id + " reconnected");
    if (timeouts[id]) {
        clearTimeout(timeouts[id].timer);
        delete timeouts[id];
    }
    if (!processorNodes[id].closed)
        processorNodes[id].disconnect = false;
    // Replace pending timers as well as the socket, so queued old callbacks are harmless.
    processorNodes[id].videos.forEach(function(video) {
        var request = pending[video.assignmentId];
        if (request) {
            clearTimeout(request.timer);
            armProcessTimeout(video, ws, request.expiresAt);
        }
    });
    if (previous.socket !== ws && previous.socket.readyState === WebSocket.OPEN)
        previous.socket.close(1008, 'Connection superseded');
    if (!processorNodes[id].closed)
        broadcastStatus();
    return true;
}

// If a processor loses connection, clean up outstanding tasks
function onSocketDisconnect(ws) {  
    return function(code, reason) {
        if (!isCurrentSocket(ws) || processorNodes[ws.id].closed)
            return;
        var id = ws.id;
        if (timeouts[id])
            return;
        processorNodes[id].disconnect = true;
        logger.debug("WS " + id + " disconnected with Code:" + code + " and Reason:" + reason);
        var timeout = {socket: ws};
        timeouts[id] = timeout;
        timeout.timer = setTimeout(onReconnectFail(id, timeout), reconnectTimer);
        broadcastStatus();
    };
}

function onReconnectFail(id, timeout) {
    return function() {
        if (timeouts[id] !== timeout || !isCurrentSocket(timeout.socket) ||
                processorNodes[id].closed || !processorNodes[id].disconnect)
            return;
        logger.debug("WS " + id + " failed to reconnect");
        processorNodes[id].videos.slice().forEach(function(video) {
            if (removeVideoFromProcessor(id, video.path, video.assignmentId))
                VideoManager.addVideo(video.path);
        });
        processorNodes[id].closed = true;
        delete connections[id];
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
    if (!isCurrentSocket(ws) || ws.readyState !== WebSocket.OPEN)
        return;
    var msgObj;
    var id = ws.id;
    logger.debug("Parsing message from " + id);
    try {
        msgObj = JSON.parse(message);
    } catch (e) {
        logger.debug("Invalid JSON Sent by " + id);
        return;
    }
    if (!msgObj || typeof msgObj !== 'object' || Array.isArray(msgObj) ||
            !Object.prototype.hasOwnProperty.call(msgObj, 'action') || typeof msgObj.action !== 'string')
        return;
    if ((msgObj.action === actionTypes.req_rec || msgObj.action === actionTypes.complete) &&
            (typeof msgObj.video !== 'string' || !msgObj.video.length ||
             typeof msgObj.assignmentId !== 'string' || !/^[a-f0-9]{64}$/.test(msgObj.assignmentId)))
        return;
    if (msgObj.action === actionTypes.complete && typeof msgObj.output !== 'string')
        return;
    if (msgObj.action === actionTypes.error && msgObj.description !== undefined && typeof msgObj.description !== 'string')
        return;
    if (processorNodes[id].closed && msgObj.action !== actionTypes.complete)
        return;
    switch(msgObj.action) {
        case actionTypes.req_video:
            logger.info("Video requested by " + id);
            sendNextVideo(ws);
            break;
        case actionTypes.req_rec:
            logger.info("Confirm request received by " + id);
            if (!processReceived(msgObj, ws))
                return;
            break;
        case actionTypes.stat_rep:
            logger.info("Status Reported by " + id);
            processStatusReport(msgObj, ws);
            break;
        case actionTypes.complete:
            logger.info("Task Complete by " + id);
            if (!processTaskComplete(msgObj, ws))
                return;
            break;
        case actionTypes.error:
            logger.info("Error Reported by " + id);
            handleProcError(msgObj, ws);
            break;
        default:
            logger.info("Invalid Input by " + id);
            // TODO Determine how to handle bad input
            return;
    }
    processorNodes[id].lastResponse = new Date().getTime();
    broadcastStatus();
}

function sendNextVideo(ws) {
    if (!isCurrentSocket(ws) || processorNodes[ws.id].closed || ws.readyState !== WebSocket.OPEN)
        return;
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
        assignmentId: crypto.randomBytes(32).toString('hex')
    };
    processorNodes[id].videos.push(videoInfo);
    processorNodes[id].idle = false;
    armProcessTimeout(videoInfo, ws, new Date().getTime() + processTimer);
    // TODO validate path is real?
    requestMessage = {
        action: actionTypes.process,
        analyzer: analyzerPath,
        path: nextVideoPath,
        assignmentId: videoInfo.assignmentId
    };
    sendRequest(requestMessage, ws);
}

function processReceived(msgObj, ws) {
    if (!isCurrentSocket(ws) || processorNodes[ws.id].closed)
        return false;
    var request = pending[msgObj.assignmentId];
    if (!request || request.owner !== ws.id || request.socket !== ws || request.video !== msgObj.video ||
            !processorNodes[ws.id].videos.some(function(video) {
                return video.path === msgObj.video && video.assignmentId === msgObj.assignmentId;
            }))
        return false;
    logger.debug("Clearing timeout for " + msgObj.video);
    clearTimeout(request.timer);
    delete pending[msgObj.assignmentId];
    return true;
}

function armProcessTimeout(video, ws, expiresAt) {
    var request = {owner: ws.id, socket: ws, video: video.path,
        assignmentId: video.assignmentId, expiresAt: expiresAt};
    pending[video.assignmentId] = request;
    request.timer = setTimeout(processTimeout(request), Math.max(0, expiresAt - new Date().getTime()));
}

function processTimeout(request) {
    return function() {
        if (pending[request.assignmentId] !== request || !isCurrentSocket(request.socket) ||
                processorNodes[request.owner].closed)
            return;
        logger.info("Connection " + request.owner + " did not verify receipt of request to process " + request.video);
        if (!removeVideoFromProcessor(request.owner, request.video, request.assignmentId))
            return;
        VideoManager.addVideo(request.video);
        broadcastStatus();
    };
}

function processStatusReport(msg, ws) {
    if (!isCurrentSocket(ws) || processorNodes[ws.id].closed)
        return;
    // TODO Handle status report
    logger.info("Status Reported: " + JSON.stringify(msg));
    var id = ws.id;
    delete processorNodes[id].statusRequested;
}

function processTaskComplete(msgObj, ws) {
    if (!isCurrentSocket(ws))
        return false;
    var id = ws.id;
    var video = msgObj.video;
    var accepted = completionReceipts[msgObj.assignmentId];
    if (accepted) {
        if (accepted.owner === id && accepted.video === video && accepted.output === msgObj.output)
            sendRequest(accepted.receipt, ws);
        // Replays must not refresh liveness or trigger GUI/counter updates.
        return false;
    }
    if (processorNodes[id].closed)
        return false;
    if (!removeVideoFromProcessor(id, video, msgObj.assignmentId))
        return false;
    completed[video] = msgObj.output;
    var receipt = {action: actionTypes.complete_accepted, video: video, assignmentId: msgObj.assignmentId};
    completionReceipts[msgObj.assignmentId] = {owner: id, video: video, output: msgObj.output, receipt: receipt};
    sendRequest(receipt, ws);
    checkProcessorComplete(ws);
    return true;
}

function removeVideoFromProcessor(id, video, assignmentId) {
    if (!Object.prototype.hasOwnProperty.call(processorNodes, id))
        return false;
    var index = processorNodes[id].videos.findIndex((videoItem) =>
        videoItem.path === video && videoItem.assignmentId === assignmentId);
    if (index == -1) {
        // Video path not assigned to this ws
        logger.error("Node reported on video it was not assigned: " + id);
        // TODO Handle malformed input
        return false;
    }
    var request = pending[assignmentId];
    if (request && request.owner === id && request.video === video) {
        clearTimeout(request.timer);
        delete pending[assignmentId];
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
    return true;
}

function checkProcessorComplete(ws) {
    if (!isCurrentSocket(ws) || processorNodes[ws.id].closed)
        return;
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
    if (!isCurrentSocket(ws) || processorNodes[ws.id].closed)
        return;
    var msg = {
        action: actionTypes.shutdown
    };
    sendRequest(msg, ws);
    var id = ws.id;
    logger.info("Requesting shutdown from " + id);
    processorNodes[id].closed = true;
    connections[id].terminal = true;
    if (timeouts[id]) {
        clearTimeout(timeouts[id].timer);
        delete timeouts[id];
    }
    if (!haveActiveProcessors())
        shutdownControlNode();
}

function shutdownControlNode() {
    broadcastStatus();
    var output = fs.openSync(argv.outputPath, "w");
    Object.keys(completed).forEach(function(video) {
        fs.writeSync(output, JSON.stringify({video: video, output: completed[video]}) + '\n');
    });
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
    if (!isCurrentSocket(ws) || processorNodes[ws.id].closed || ws.readyState !== WebSocket.OPEN)
        return;
    // TODO Determine if it is suffient to simply ping/pong here?
    var msg = {
        action: actionTypes.stat_req
    };
    sendRequest(msg, ws);
    var id = ws.id;
    processorNodes[id].statusRequested = new Date().getTime();
}

function sendRequest(msgObj, ws) {
    if (!isCurrentSocket(ws) || ws.readyState !== WebSocket.OPEN)
        return;
    if (processorNodes[ws.id].closed && msgObj.action !== actionTypes.con_success &&
            msgObj.action !== actionTypes.shutdown && msgObj.action !== actionTypes.complete_accepted)
        return;
    var id = ws.id;
    logger.debug("Sending " + msgObj.action + " to " + id);
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
