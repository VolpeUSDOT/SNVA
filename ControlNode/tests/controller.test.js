'use strict';

const assert = require('assert');
const crypto = require('crypto');
const EventEmitter = require('events');
const fs = require('fs');
const path = require('path');
const url = require('url');
const vm = require('vm');
const source = fs.readFileSync(path.join(__dirname, '..', 'app.js'), 'utf8');
var passed = 0;

function test(name, run) {
    run();
    passed++;
    console.log('PASS ' + name);
}

function controller(options) {
    options = options || {};
    const queue = (options.videos || []).slice();
    const timers = [];
    const effects = {readPaths: 0, loggers: 0, writes: '', servers: 0};
    const argv = Object.assign({tlsCert: 'cert.pem', tlsKey: 'key.pem', tlsCa: 'ca.pem',
        allowedOrigin: 'https://localhost:8081',
        host: '0.0.0.0', port: 8081, logDir: 'logs', inputFile: 'videos.txt',
        outputPath: './outputList.txt', analyzerCount: 2}, options.argv);
    const yargs = {option: function() { return this; }, help: function() { return this; },
        alias: function() { return this; }, argv: argv};
    const logger = {info: function() {}, error: function() {}, debug: function() {}, warn: function() {}};
    const server = new EventEmitter();
    server.listen = function(port, host) { effects.listen = [port, host]; };
    function Socket() {
        EventEmitter.call(this);
        this._socket = {remoteAddress: '127.0.0.1'};
        this.readyState = Socket.OPEN;
        this.sent = [];
        this.closed = [];
    }
    Socket.prototype = Object.create(EventEmitter.prototype);
    Socket.prototype.send = function(data, opts, callback) {
        this.sent.push(JSON.parse(data));
        if (callback) callback();
    };
    Socket.prototype.close = function(code, reason) {
        this.closed.push([code, reason]);
        this.readyState = 3;
        this.emit('close', code, reason);
    };
    Socket.prototype.ping = function() {};
    Socket.OPEN = 1;
    const webServers = [];
    Socket.Server = function() {
        const instance = new EventEmitter();
        instance.clients = new Set();
        instance.handleUpgrade = function(req, socket, head, callback) {
            effects.upgraded = true;
            callback(new Socket());
        };
        webServers.push(instance);
        return instance;
    };
    function express() { return {use: function() {}}; }
    express.static = function() {};
    const mocks = {
        ws: Socket,
        yargs: yargs,
        fs: {
            statSync: function(name) {
                if (options.missingFile === name) throw new Error('ENOENT');
                return {isFile: function() { return options.directory !== name; }};
            },
            readFileSync: function(name) {
                if (options.emptyFile === name) return Buffer.alloc(0);
                return Buffer.from(name === 'ca.pem' ? (options.ca ||
                    '-----BEGIN CERTIFICATE-----\nvalid\n-----END CERTIFICATE-----') : 'valid');
            },
            openSync: function() { return 1; },
            writeSync: function(fd, data) { effects.writes += data; },
            closeSync: function() {}
        },
        path: path,
        crypto: Object.assign({}, crypto, {createPublicKey: function() {}}),
        tls: {createSecureContext: function(opts) {
            effects.tls = opts;
            if (options.invalidTls) throw new Error('Invalid TLS credentials');
        }},
        https: {createServer: function(opts) {
            effects.servers++;
            effects.https = opts;
            return server;
        }},
        url: url,
        express: express,
        winston: {
            createLogger: function() { effects.loggers++; return logger; },
            format: {combine: function() {}, timestamp: function() {}, json: function() {}},
            transports: {File: function() {}}
        },
        './modules/videoPathManager.js': {
            readInputPaths: function() { effects.readPaths++; },
            nextVideo: function() { return queue.length ? queue.pop() : null; },
            addVideo: function(video) { queue.push(video); },
            getCount: function() { return queue.length; },
            isComplete: function() { return queue.length === 0; }
        },
        './modules/dockerManager.js': {}
    };
    const context = vm.createContext({require: function(name) {
        assert(Object.prototype.hasOwnProperty.call(mocks, name), 'Unexpected dependency: ' + name);
        return mocks[name];
    }, Buffer: Buffer, process: {exit: function(code) { effects.exit = code; }},
    setTimeout: function(callback, delay) {
        const timer = {callback: callback, delay: delay, cleared: false};
        timers.push(timer);
        return timer;
    }, clearTimeout: function(timer) { if (timer) timer.cleared = true; }});
    try {
        // CommonJS permits the existing top-level return in node provisioning.
        vm.runInContext('var scope = this; (function() {\n' + source +
            '\nObject.assign(scope, {processorNodes: processorNodes, connections: connections,' +
            'timeouts: timeouts, pending: pending, completed: completed, completionReceipts: completionReceipts,' +
            'analyzerNodes: analyzerNodes, sendNextVideo: sendNextVideo, requestStatus: requestStatus,' +
            'initializeConnection: initializeConnection, onSocketDisconnect: onSocketDisconnect,' +
            'parseMessage: parseMessage, getGuiInfo: getGuiInfo, startAnalyzer: startAnalyzer});\n})();',
            context, {filename: 'app.js'});
    } catch (error) {
        if (!options.startupFails) throw error;
        return {error: error, effects: effects};
    }
    assert(!options.startupFails, 'Expected startup to fail');
    function connect(id, token, fingerprint) {
        const ws = new Socket();
        ws.on('close', context.onSocketDisconnect(ws));
        context.initializeConnection(ws, id, token, fingerprint === undefined ? 'certificate-one' : fingerprint);
        return ws;
    }
    return {context: context, effects: effects, queue: queue, timers: timers, connect: connect,
        Socket: Socket, server: server, webServers: webServers,
        message: function(ws, msg) { context.parseMessage(JSON.stringify(msg), ws); }};
}

function state(c) {
    return JSON.stringify({processors: c.context.processorNodes, completed: c.context.completed,
        receipts: c.context.completionReceipts,
        pending: Object.keys(c.context.pending), timeouts: Object.keys(c.context.timeouts),
        analyzers: c.context.analyzerNodes, queue: c.queue, writes: c.effects.writes});
}

function assignment(c, ws) {
    c.message(ws, {action: 'REQUEST_VIDEO'});
    const request = ws.sent[ws.sent.length - 1];
    assert.strictEqual(request.action, 'PROCESS');
    assert(/^[a-f0-9]{64}$/.test(request.assignmentId));
    return request;
}

function ack(request) {
    return {action: 'REQUEST_RECEIVED', video: request.path, assignmentId: request.assignmentId};
}

function complete(request, output) {
    return {action: 'COMPLETE', video: request.path, assignmentId: request.assignmentId, output: output};
}

test('TLS requires scalar readable files and validates before any state startup', function() {
    const cases = [
        {argv: {tlsCert: undefined}}, {argv: {tlsKey: ['key.pem', 'key.pem']}},
        {argv: {tlsCa: ''}}, {missingFile: 'cert.pem'}, {directory: 'key.pem'},
        {emptyFile: 'ca.pem'}, {ca: 'not a certificate'}, {invalidTls: true},
        {argv: {port: '8081'}}, {argv: {host: ['localhost', '0.0.0.0']}},
        {argv: {allowedOrigin: undefined}}, {argv: {allowedOrigin: ['https://localhost:8081']}},
        {argv: {allowedOrigin: 'http://localhost:8081'}}, {argv: {allowedOrigin: 'wss://localhost:8081'}},
        {argv: {allowedOrigin: 'https://localhost:8081/'}}, {argv: {allowedOrigin: 'https://localhost:8081/path'}},
        {argv: {allowedOrigin: 'https://localhost:8081?query'}}, {argv: {allowedOrigin: 'https://localhost:8081#fragment'}},
        {argv: {allowedOrigin: 'https://user@localhost:8081'}}, {argv: {allowedOrigin: 'https://LOCALHOST:8081'}},
        {argv: {allowedOrigin: 'https://localhost:443'}}, {argv: {allowedOrigin: 'null'}},
        {argv: {allowedOrigin: ' https://localhost:8081'}}, {argv: {allowedOrigin: 'https://localhost:8081\n'}}
    ];
    cases.forEach(function(options) {
        const c = controller(Object.assign({startupFails: true}, options));
        assert(c.error);
        assert.strictEqual(c.effects.readPaths, 0);
        assert.strictEqual(c.effects.loggers, 0);
        assert.strictEqual(c.effects.servers, 0);
    });
    const c = controller();
    assert.deepStrictEqual(c.effects.listen, [8081, '0.0.0.0']);
    assert.strictEqual(c.effects.https.minVersion, 'TLSv1.2');
    assert.strictEqual(c.effects.https.requestCert, true);
    assert.strictEqual(c.effects.https.rejectUnauthorized, true);
    controller({argv: {allowedOrigin: 'https://localhost'}});
    controller({argv: {allowedOrigin: 'https://[::1]:8081'}});
});

test('all client-keyed registries have no prototype and GUI omits secrets/sockets', function() {
    const c = controller();
    c.connect();
    ['processorNodes', 'connections', 'timeouts', 'pending', 'completed', 'completionReceipts'].forEach(function(name) {
        assert.strictEqual(Object.getPrototypeOf(c.context[name]), null);
    });
    const gui = JSON.parse(c.context.getGuiInfo());
    assert.strictEqual(Object.keys(gui.processorInfo).length, 1);
    assert(!/reconnectToken|fingerprint|socket/.test(JSON.stringify(gui)));
});

test('unknown, malformed, prototype and duplicate reconnect IDs/tokens are rejected without mutation', function() {
    const c = controller();
    const owner = c.connect();
    const success = owner.sent[0];
    assert.strictEqual(success.action, 'CONNECTION_SUCCESS');
    assert(/^[a-f0-9]{64}$/.test(success.reconnectToken));
    const invalid = [null, '', 0, {}, [], ['Processor0', 'Processor0'], '__proto__',
        'constructor', 'prototype', 'toString', 'Processor99', 'Processor00', 'Processor-1'];
    invalid.forEach(function(id) {
        const before = state(c);
        const ws = c.connect(id, success.reconnectToken);
        assert.strictEqual(ws.closed[0][0], 1008);
        assert.strictEqual(state(c), before);
    });
    [undefined, null, '', {}, [], [success.reconnectToken, success.reconnectToken], '0'.repeat(64)].forEach(function(token) {
        const before = state(c);
        assert.strictEqual(c.connect(success.id, token).closed[0][0], 1008);
        assert.strictEqual(state(c), before);
    });
    assert.strictEqual(c.connect(undefined, success.reconnectToken).closed[0][0], 1008);
    assert.strictEqual(c.connect(success.id, success.reconnectToken, 'different-certificate').closed[0][0], 1008);
    assert.strictEqual(vm.runInContext('Object.prototype.disconnect', c.context), undefined);
});

test('duplicate query fields are rejected through real URL parsing and connection callbacks', function() {
    const c = controller();
    const owner = c.connect();
    const success = owner.sent[0];
    ['/registerProcess?id=Processor0&id=Processor0&reconnectToken=' + success.reconnectToken,
        '/registerProcess?id=Processor0&reconnectToken=' + success.reconnectToken +
            '&reconnectToken=' + success.reconnectToken,
        '/registerProcess?id=__proto__'].forEach(function(address) {
        const ws = new c.Socket();
        const before = state(c);
        c.webServers[0].emit('connection', ws, {url: address, socket: {
            authorized: true, getPeerCertificate: function() { return {raw: Buffer.from('cert')}; }
        }});
        assert.strictEqual(ws.closed[0][0], 1008);
        ws.emit('message', '{"action":"REQUEST_VIDEO"}');
        assert.strictEqual(state(c), before);
    });
});

test('both GUI and processor upgrades reject missing certificate authorization', function() {
    const c = controller();
    ['/snvaStatus', '/registerProcess'].forEach(function(endpoint) {
        var destroyed = false;
        c.server.emit('upgrade', {url: endpoint}, {authorized: false,
            destroy: function() { destroyed = true; }}, Buffer.alloc(0));
        assert(destroyed);
        assert(!c.effects.upgraded);
    });
});

test('browser upgrades require exact scalar Origin; GUI requires Origin and CLI processor may omit it', function() {
    function upgrade(endpoint, origin, rawHeaders) {
        const c = controller();
        var destroyed = false;
        const socket = {authorized: true, getPeerCertificate: function() { return {raw: Buffer.from('cert')}; },
            destroy: function() { destroyed = true; }};
        c.server.emit('upgrade', {url: endpoint, socket: socket,
            headers: origin === undefined ? {} : {origin: origin}, rawHeaders: rawHeaders}, socket, Buffer.alloc(0));
        return {destroyed: destroyed, upgraded: c.effects.upgraded};
    }
    ['/snvaStatus', '/registerProcess'].forEach(function(endpoint) {
        ['https://attacker.invalid', 'null', '', 'https://localhost:8081/', 'http://localhost:8081',
            ['https://localhost:8081'], {}, 'https://localhost:8082', 'https://LOCALHOST:8081'].forEach(function(origin) {
            const result = upgrade(endpoint, origin);
            assert(result.destroyed);
            assert(!result.upgraded);
        });
        const duplicated = upgrade(endpoint, 'https://localhost:8081',
            ['Origin', 'https://localhost:8081', 'ORIGIN', 'https://localhost:8081']);
        assert(duplicated.destroyed);
        assert(!duplicated.upgraded);
        const valid = upgrade(endpoint, 'https://localhost:8081', ['Origin', 'https://localhost:8081']);
        assert(!valid.destroyed);
        assert(valid.upgraded);
    });
    assert(upgrade('/snvaStatus').destroyed);
    assert(upgrade('/registerProcess').upgraded);
});

test('authenticated reconnect replaces socket and makes stale close/message/timers inert', function() {
    const c = controller({videos: ['next.mp4', 'owned.mp4']});
    const old = c.connect();
    const request = assignment(c, old);
    const oldPending = c.context.pending[request.assignmentId];
    old.close(1006, 'disconnected');
    const disconnect = c.context.timeouts[old.id];
    const replacement = c.connect(old.id, old.sent[0].reconnectToken);
    assert.strictEqual(replacement.sent[0].id, old.id);
    assert.strictEqual(replacement.sent[0].reconnectToken, old.sent[0].reconnectToken);
    assert.strictEqual(c.context.processorNodes[old.id].disconnect, false);
    assert(disconnect.timer.cleared);
    assert(oldPending.timer.cleared);
    assert.strictEqual(c.context.pending[request.assignmentId].socket, replacement);
    assert.strictEqual(c.context.pending[request.assignmentId].expiresAt, oldPending.expiresAt);
    const before = state(c);
    old.readyState = c.Socket.OPEN;
    c.message(old, {action: 'REQUEST_VIDEO'});
    c.message(old, complete(request, 'forged'));
    c.context.onSocketDisconnect(old)(1006, 'late close');
    disconnect.timer.callback();
    oldPending.timer.callback();
    assert.strictEqual(state(c), before);
    c.message(replacement, ack(request));
    assert.strictEqual(c.context.pending[request.assignmentId], undefined);
    c.message(replacement, complete(request, 'valid'));
    assert.strictEqual(c.context.completed[request.path], 'valid');
});

test('live socket takeover closes old connection and cannot schedule a stale disconnect', function() {
    const c = controller();
    const old = c.connect();
    const replacement = c.connect(old.id, old.sent[0].reconnectToken);
    assert.strictEqual(old.closed[0][1], 'Connection superseded');
    assert.strictEqual(c.context.connections[old.id].socket, replacement);
    assert.strictEqual(Object.keys(c.context.timeouts).length, 0);
});

test('ACK and COMPLETE require exact owner, full path and per-attempt ID', function() {
    const c = controller({videos: ['remaining', 'nested/a/video.mp4', 'nested/b/video.mp4']});
    c.context.startAnalyzer('analyzer');
    const owner = c.connect();
    const stranger = c.connect();
    const first = assignment(c, owner);
    const second = assignment(c, stranger);
    assert.notStrictEqual(first.assignmentId, second.assignmentId);
    assert.strictEqual(c.context.analyzerNodes[0].numVideos, 2);
    const pending = c.context.pending[first.assignmentId];
    const before = state(c);
    c.message(stranger, ack(first));
    c.message(stranger, complete(first, 'forged'));
    c.message(owner, Object.assign(ack(first), {video: 'video.mp4'}));
    c.message(owner, Object.assign(complete(first, 'forged'), {video: second.path}));
    c.message(owner, Object.assign(complete(first, 'forged'), {assignmentId: second.assignmentId}));
    assert.strictEqual(state(c), before);
    assert(!pending.timer.cleared);
    c.message(owner, ack(first));
    assert(pending.timer.cleared);
    const acknowledged = state(c);
    c.message(owner, ack(first));
    pending.timer.callback();
    assert.strictEqual(state(c), acknowledged);
    c.message(owner, complete(first, 'first output'));
    assert.strictEqual(c.context.completed[first.path], 'first output');
    assert.strictEqual(c.context.analyzerNodes[0].numVideos, 1);
    const finished = state(c);
    c.message(owner, complete(first, 'overwrite'));
    c.message(stranger, complete(first, 'overwrite'));
    assert.strictEqual(state(c), finished);
    c.message(stranger, complete(second, 'second output'));
    assert.strictEqual(c.context.completed[second.path], 'second output');
    assert.strictEqual(c.context.analyzerNodes[0].numVideos, 0);
});

test('completion before ACK atomically cancels timer and rejects later ACK/replay', function() {
    const c = controller({videos: ['remaining', 'video.mp4']});
    const owner = c.connect();
    const request = assignment(c, owner);
    const pending = c.context.pending[request.assignmentId];
    c.message(owner, complete(request, 'output'));
    assert(pending.timer.cleared);
    const before = state(c);
    pending.timer.callback();
    c.message(owner, ack(request));
    c.message(owner, complete(request, 'replacement'));
    assert.strictEqual(state(c), before);
});

test('missing completion receipt can be retried after reconnect without any state mutation', function() {
    const c = controller({videos: ['remaining', 'owned']});
    c.context.startAnalyzer('analyzer');
    const owner = c.connect();
    const stranger = c.connect();
    const request = assignment(c, owner);
    const message = complete(request, 'original output');
    c.message(owner, message);
    const receipt = owner.sent.pop(); // Simulate application loss after the controller accepts COMPLETE.
    assert.deepStrictEqual(receipt, {action: 'COMPLETE_ACCEPTED', video: request.path,
        assignmentId: request.assignmentId});
    assert.strictEqual(c.context.completionReceipts[request.assignmentId].owner, owner.id);
    owner.close(1006, 'receipt lost');
    const replacement = c.connect(owner.id, owner.sent[0].reconnectToken);
    c.context.processorNodes[owner.id].lastResponse = -1;
    const before = state(c);
    const sent = replacement.sent.length;
    c.message(replacement, message);
    assert.deepStrictEqual(replacement.sent[sent], receipt);
    assert.strictEqual(state(c), before);
    c.message(replacement, message);
    assert.deepStrictEqual(replacement.sent[sent + 1], receipt);
    assert.strictEqual(state(c), before);
    const acceptedCount = replacement.sent.length;
    c.message(replacement, Object.assign({}, message, {output: 'changed'}));
    c.message(replacement, Object.assign({}, message, {video: 'remaining'}));
    c.message(replacement, Object.assign({}, message, {assignmentId: '0'.repeat(64)}));
    c.message(stranger, message);
    owner.readyState = c.Socket.OPEN;
    c.message(owner, message);
    assert.strictEqual(replacement.sent.length, acceptedCount);
    assert.strictEqual(stranger.sent.length, 1);
    assert.strictEqual(state(c), before);
    const gui = JSON.stringify(JSON.parse(c.context.getGuiInfo()));
    assert(!gui.includes('original output'));
    assert(!gui.includes('completionReceipts'));
});

test('terminal reconnect recovers lost SHUTDOWN and receipts but never reopens work or status', function() {
    const c = controller({videos: ['other', 'finished']});
    c.context.startAnalyzer('analyzer');
    const owner = c.connect();
    const other = c.connect();
    const first = assignment(c, owner);
    const second = assignment(c, other);
    const pending = c.context.pending[first.assignmentId];
    const message = complete(first, 'terminal output');
    c.message(owner, message);
    assert.deepStrictEqual(owner.sent.slice(-2).map(function(msg) { return msg.action; }),
        ['COMPLETE_ACCEPTED', 'SHUTDOWN']);
    const receipt = owner.sent[owner.sent.length - 2];
    owner.close(1006, 'receipt and shutdown lost');
    assert.strictEqual(Object.keys(c.context.timeouts).length, 0);
    assert.strictEqual(c.context.connections[owner.id].terminal, true);
    assert.strictEqual(c.context.processorNodes[other.id].closed, undefined);
    const before = state(c);
    const timers = c.timers.length;
    const replacement = c.connect(owner.id, owner.sent[0].reconnectToken);
    assert.deepStrictEqual(replacement.sent.map(function(msg) { return msg.action; }),
        ['CONNECTION_SUCCESS', 'SHUTDOWN']);
    assert.strictEqual(state(c), before);
    assert.strictEqual(c.timers.length, timers);
    c.context.processorNodes[owner.id].lastResponse = -1;
    c.context.processorNodes[owner.id].statusRequested = 123;
    const terminal = state(c);
    c.message(replacement, message);
    assert.deepStrictEqual(replacement.sent[2], receipt);
    assert.strictEqual(state(c), terminal);
    c.message(replacement, {action: 'REQUEST_VIDEO'});
    c.message(replacement, {action: 'STATUS_REPORT'});
    c.message(replacement, {action: 'ERROR', description: 'ignored'});
    c.message(replacement, ack(first));
    c.message(replacement, complete(second, 'foreign'));
    c.message(replacement, Object.assign({}, message, {output: 'changed'}));
    c.message(replacement, complete({path: 'new work', assignmentId: '0'.repeat(64)}, 'new'));
    c.context.sendNextVideo(replacement);
    c.context.requestStatus(replacement);
    c.context.onSocketDisconnect(owner)(1006, 'stale close');
    pending.timer.callback();
    assert.strictEqual(state(c), terminal);
    assert.strictEqual(replacement.sent.length, 3);
    assert.strictEqual(c.timers.length, timers);
    replacement.close(1006, 'terminal lost again');
    const again = c.connect(owner.id, owner.sent[0].reconnectToken);
    assert.strictEqual(again.sent[1].action, 'SHUTDOWN');
    assert.strictEqual(state(c), terminal);
});

test('ACK expiry requeues once and stale attempt cannot affect a reassignment', function() {
    const c = controller({videos: ['video.mp4']});
    c.context.startAnalyzer('analyzer');
    const old = c.connect();
    const newer = c.connect();
    const first = assignment(c, old);
    const pending = c.context.pending[first.assignmentId];
    pending.timer.callback();
    assert.strictEqual(c.queue.length, 1);
    assert.strictEqual(c.context.analyzerNodes[0].numVideos, 0);
    const second = assignment(c, newer);
    assert.strictEqual(second.path, first.path);
    assert.notStrictEqual(second.assignmentId, first.assignmentId);
    const before = state(c);
    pending.timer.callback();
    c.message(old, ack(first));
    c.message(old, complete(first, 'stale'));
    c.message(newer, complete(first, 'stale'));
    assert.strictEqual(state(c), before);
    c.message(newer, ack(second));
    c.message(newer, complete(second, 'current'));
    assert.strictEqual(c.context.completed[first.path], 'current');
    assert.strictEqual(c.context.analyzerNodes[0].numVideos, 0);
});

test('disconnect expiry requeues every assignment once and cancels all ACK timers', function() {
    const c = controller({videos: ['one', 'two', 'three']});
    c.context.startAnalyzer('analyzer');
    const ws = c.connect();
    const requests = [assignment(c, ws), assignment(c, ws), assignment(c, ws)];
    const pending = requests.map(function(request) { return c.context.pending[request.assignmentId]; });
    ws.close(1006, 'lost');
    const timeout = c.context.timeouts[ws.id];
    timeout.timer.callback();
    assert.strictEqual(c.queue.length, 3);
    assert.strictEqual(c.context.processorNodes[ws.id].videos.length, 0);
    assert.strictEqual(c.context.analyzerNodes[0].numVideos, 0);
    assert.strictEqual(c.context.processorNodes[ws.id].closed, true);
    assert.strictEqual(c.context.connections[ws.id], undefined);
    assert.strictEqual(Object.keys(c.context.pending).length, 0);
    const before = state(c);
    pending.forEach(function(request) { assert(request.timer.cleared); request.timer.callback(); });
    timeout.timer.callback();
    assert.strictEqual(state(c), before);
    assert.strictEqual(c.connect(ws.id, ws.sent[0].reconnectToken).closed[0][0], 1008);
});

test('expired processors cannot replay retained receipts or authenticate again', function() {
    const c = controller({videos: ['remaining', 'completed']});
    const ws = c.connect();
    const request = assignment(c, ws);
    const message = complete(request, 'accepted');
    c.message(ws, message);
    ws.close(1006, 'expired');
    c.context.timeouts[ws.id].timer.callback();
    assert(c.context.completionReceipts[request.assignmentId]);
    assert.strictEqual(c.context.connections[ws.id], undefined);
    const before = state(c);
    const count = ws.sent.length;
    ws.readyState = c.Socket.OPEN;
    c.message(ws, message);
    assert.strictEqual(ws.sent.length, count);
    assert.strictEqual(c.connect(ws.id, ws.sent[0].reconnectToken).closed[0][0], 1008);
    assert.strictEqual(state(c), before);
});

test('malformed scalar/object/array/field schemas never mutate assignments or completion state', function() {
    const c = controller({videos: ['remaining', 'owned']});
    const ws = c.connect();
    const request = assignment(c, ws);
    const malformed = [null, [], true, 1, 'string', {}, {action: null}, {action: []},
        Object.assign(ack(request), {video: null}), Object.assign(ack(request), {video: ['owned']}),
        Object.assign(ack(request), {video: {toString: null}}), Object.assign(ack(request), {assignmentId: []}),
        Object.assign(ack(request), {assignmentId: {}}), Object.assign(complete(request), {output: null}),
        Object.assign(complete(request), {output: {toString: null}}),
        Object.assign(complete(request), {output: []}), {action: 'ERROR', description: {}},
        {action: 'COMPLETE', video: '__proto__', assignmentId: '0'.repeat(64), output: 'bad'},
        {action: 'COMPLETE', video: 'constructor', assignmentId: '0'.repeat(64), output: 'bad'}];
    const before = state(c);
    malformed.forEach(function(msg) { c.message(ws, msg); assert.strictEqual(state(c), before); });
    c.context.parseMessage('{ invalid JSON', ws);
    c.context.parseMessage('{"__proto__":{"action":"REQUEST_VIDEO"}}', ws);
    assert.strictEqual(state(c), before);
    // Valid ignored status data must not invoke attacker-supplied object coercion.
    c.message(ws, {action: 'STATUS_REPORT', toString: null, valueOf: null});
    assert.strictEqual(c.context.processorNodes[ws.id].videos.length, 1);
});

test('full shutdown ledger is JSONL with prototype keys and injected CR/LF/delimiters escaped', function() {
    const paths = ['__proto__', 'constructor', 'nested/a/same.mp4', 'nested/b/same.mp4',
        'line\r\nvideo: "quote"\\end\u0000'];
    const outputs = Object.create(null);
    const c = controller({videos: paths});
    c.context.startAnalyzer('analyzer');
    const ws = c.connect();
    paths.forEach(function() {
        const request = assignment(c, ws);
        const output = 'output: "x"\\path\r\n{"video":"forged"}\t\u0000';
        outputs[request.path] = output;
        c.message(ws, complete(request, output));
    });
    assert.strictEqual(ws.sent[ws.sent.length - 1].action, 'SHUTDOWN');
    assert.strictEqual(ws.sent[ws.sent.length - 2].action, 'COMPLETE_ACCEPTED');
    assert.strictEqual(Object.keys(c.context.completionReceipts).length, paths.length);
    assert.strictEqual(c.context.processorNodes[ws.id].closed, true);
    assert.strictEqual(c.context.analyzerNodes[0].numVideos, 0);
    assert.strictEqual(Object.keys(c.context.pending).length, 0);
    const lines = c.effects.writes.trim().split('\n');
    assert.strictEqual(lines.length, paths.length);
    lines.forEach(function(line) {
        assert(!line.includes('\r'));
        const entry = JSON.parse(line);
        assert.deepStrictEqual(Object.keys(entry), ['video', 'output']);
        assert.strictEqual(entry.output, outputs[entry.video]);
        delete outputs[entry.video];
    });
    assert.strictEqual(Object.keys(outputs).length, 0);
    const before = state(c);
    c.context.onSocketDisconnect(ws)(1000, 'shutdown');
    assert.strictEqual(state(c), before);
    assert.strictEqual(c.timers[c.timers.length - 1].delay, 5000);
    const timers = c.timers.length;
    const replacement = c.connect(ws.id, ws.sent[0].reconnectToken);
    assert.strictEqual(replacement.sent[1].action, 'SHUTDOWN');
    const accepted = c.context.completionReceipts[Object.keys(c.context.completionReceipts)[0]];
    c.message(replacement, {action: 'COMPLETE', video: accepted.video,
        assignmentId: accepted.receipt.assignmentId, output: accepted.output});
    assert.deepStrictEqual(replacement.sent[2], JSON.parse(JSON.stringify(accepted.receipt)));
    assert.strictEqual(state(c), before);
    assert.strictEqual(c.timers.length, timers);
});

console.log(passed + ' controller unit tests passed');
