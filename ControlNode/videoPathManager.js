const fs = require('fs');
const readline = require('readline');

var toProcess = [];

exports.readInputPaths = readInputPaths;
exports.nextVideo = nextVideo;
exports.addVideo = addVideo;
exports.isComplete = isComplete;

function readInputPaths(pathFile) {
    readline.createInterface({
        input: fs.createReadStream(pathFile),
        terminal: false
    }).on('line', function(line) {
        toProcess.push(line);
    });
}

function nextVideo() {
    if (toProcess.length > 0)
        return toProcess.pop();
    return null;
}

function addVideo(videoPath) {
    toProcess.push(videoPath);
}

function isComplete() {
    return toProcess.length <= 0;
}