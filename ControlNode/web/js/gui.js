$(document).ready(function() {

    var url = "ws://" + window.location.hostname + ":" + window.location.port + "/snvaStatus";
    const socket = new WebSocket(url);

    socket.addEventListener('message', function (event) {
        console.log('Message from server ', event.data);
        updateTable(JSON.parse(event.data));
    });
});

// Processors we are currently tracking
var processorList = [];

function updateTable(data) {
    console.log("data", data);
    console.log("Updating");
    console.log("Remaining", data.videosRemaining);
    $("#videosRemaining").text(data.videosRemaining);
    for (var id in data.processorInfo) {
        console.log("ID", id);
        var procData = parseProcessorData(data.processorInfo[id]);
        console.log("ProcData", procData);
        if (processorList.includes(id)) {
            console.log("Updating");
            updateRow(id, procData);
        } else {
            console.log("Updating");
            createRow(id, procData);
            processorList.push(id);
        }
    }
}

function parseProcessorData(data) {
    var procData = {};
    procData.ip = data.ip;
    procData.videoCount = data.videos.length;
    if (data.closed) {
        procData.status = "<span style='color:red;'>Closed</span>";
    } else if (data.disconnect) {
        procData.status = "<span style='color:orange;'>Disconnected - Awaiting Reconnect</span>";
    } else {
        procData.status = "<span style='color:green;'>Running</span>";
    }
    return procData;
}

function createRow(id, processorInfo) {
    var tableRow = "<tr>";
        // Name
        tableRow += "<td>" + id + "</td>";
        // IP
        tableRow += "<td>" + processorInfo.ip + "</td>";
        // Videos Assigned
        tableRow += "<td><span id=" + id + "Assigned>" + processorInfo.videoCount + "</span></td>";
        // Status
        tableRow += "<td><div id=" + id + "Status>" + processorInfo.status + "</div></td>";
        tableRow += "</tr>";
        $("#processorTableBody").append(tableRow);
}

function updateRow(id, processorInfo) {
    $("#" + id + "Assigned").text(processorInfo.videoCount);
    $("#" + id + "Status").html(processorInfo.status);
}

// Parse received data
// Compare to existing
// Update and create new rows if necessary