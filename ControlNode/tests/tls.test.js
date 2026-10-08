'use strict';

const assert = require('assert');
const childProcess = require('child_process');
const fs = require('fs');
const https = require('https');
const net = require('net');
const os = require('os');
const path = require('path');
const WebSocket = require('ws');
const root = path.join(__dirname, '..');
const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'snva-controller-tls-'));
var server;
var serverOutput = '';
const channels = [];
var passed = 0;

function openssl(args) {
    childProcess.execFileSync(process.env.OPENSSL || 'openssl', args, {cwd: temporary, stdio: 'pipe'});
}

function fixture(name) { return path.join(temporary, name); }
function contents(name) { return fs.readFileSync(fixture(name)); }

function createCa(name) {
    openssl(['req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-keyout', name + '.key',
        '-out', name + '.pem', '-days', '1', '-subj', '/CN=' + name,
        '-addext', 'basicConstraints=critical,CA:TRUE', '-addext', 'keyUsage=critical,keyCertSign,cRLSign']);
}

function createCertificate(name, ca, serverCertificate) {
    openssl(['req', '-new', '-newkey', 'rsa:2048', '-nodes', '-keyout', name + '.key',
        '-out', name + '.csr', '-subj', '/CN=' + (serverCertificate ? 'localhost' : name)]);
    fs.writeFileSync(fixture(name + '.ext'), 'basicConstraints=critical,CA:FALSE\n' +
        'keyUsage=critical,digitalSignature,keyEncipherment\n' +
        'extendedKeyUsage=' + (serverCertificate ? 'serverAuth' : 'clientAuth') + '\n' +
        (serverCertificate ? 'subjectAltName=DNS:localhost,IP:127.0.0.1\n' : ''));
    openssl(['x509', '-req', '-in', name + '.csr', '-CA', ca + '.pem', '-CAkey', ca + '.key',
        '-CAcreateserial', '-out', name + '.pem', '-days', '1', '-extfile', name + '.ext']);
}

function tlsOptions(name) {
    const options = {ca: contents('ca.pem'), rejectUnauthorized: true, minVersion: 'TLSv1.2'};
    if (name) {
        options.cert = contents(name + '.pem');
        options.key = contents(name + '.key');
    }
    return options;
}

function httpsRequest(port, options) {
    return new Promise(function(resolve, reject) {
        const request = https.get(Object.assign({hostname: 'localhost', port: port,
            path: '/snvaGui.html', agent: false, timeout: 2000}, options), function(response) {
            response.resume();
            response.on('end', function() { resolve(response.statusCode); });
        });
        request.on('error', reject);
        request.on('timeout', function() { request.destroy(new Error('HTTPS request timed out')); });
    });
}

function timeout(promise, description) {
    return new Promise(function(resolve, reject) {
        const timer = setTimeout(function() { reject(new Error('Timed out: ' + description)); }, 5000);
        promise.then(function(value) { clearTimeout(timer); resolve(value); },
            function(error) { clearTimeout(timer); reject(error); });
    });
}

async function rejects(promise, description) {
    var failed = false;
    try { await timeout(promise, description); } catch (error) {
        assert(!/^Timed out:/.test(error.message), error.message);
        failed = true;
    }
    assert(failed, 'Expected rejection: ' + description);
}

async function test(name, run) {
    await run();
    passed++;
    console.log('PASS ' + name);
}

function unusedPort() {
    return new Promise(function(resolve, reject) {
        const socket = net.createServer();
        socket.on('error', reject);
        socket.listen(0, '127.0.0.1', function() {
            const port = socket.address().port;
            socket.close(function() { resolve(port); });
        });
    });
}

function connection(address, options) {
    const ws = new WebSocket(address, options);
    const inbox = [];
    var waiter;
    var closed;
    const channel = {
        ws: ws,
        open: new Promise(function(resolve, reject) {
            ws.once('open', resolve);
            ws.once('error', reject);
        }),
        close: new Promise(function(resolve) {
            ws.once('close', function(code) { closed = code; resolve(code); });
        }),
        received: [],
        next: function() {
            if (inbox.length) return Promise.resolve(inbox.shift());
            if (closed !== undefined) return Promise.reject(new Error('WebSocket closed: ' + closed));
            return timeout(new Promise(function(resolve, reject) { waiter = {resolve: resolve, reject: reject}; }),
                'WebSocket message');
        },
        send: function(value) { ws.send(JSON.stringify(value)); }
    };
    ws.on('message', function(data) {
        const value = JSON.parse(data);
        channel.received.push(value);
        if (waiter) { const current = waiter; waiter = null; current.resolve(value); }
        else inbox.push(value);
    });
    ws.on('error', function(error) {
        if (waiter) { const current = waiter; waiter = null; current.reject(error); }
    });
    ws.on('close', function(code) {
        if (waiter) { const current = waiter; waiter = null; current.reject(new Error('WebSocket closed: ' + code)); }
    });
    channels.push(channel);
    return channel;
}

async function ready(port) {
    const deadline = Date.now() + 10000;
    while (Date.now() < deadline) {
        if (server.exitCode !== null) throw new Error('Controller exited before startup: ' + serverOutput);
        try { return await httpsRequest(port, tlsOptions('client')); } catch (error) {
            await new Promise(function(resolve) { setTimeout(resolve, 50); });
        }
    }
    throw new Error('Controller did not become ready: ' + serverOutput);
}

async function invalidReconnect(base, query, client) {
    const channel = connection(base + query, tlsOptions(client || 'client'));
    await timeout(channel.open, 'rejected reconnect upgrade');
    assert.strictEqual(await timeout(channel.close, 'rejected reconnect close'), 1008);
}

async function run() {
    // All keys and certificates are generated for this run and removed on completion.
    createCa('ca');
    createCa('untrusted-ca');
    createCertificate('server', 'ca', true);
    createCertificate('client', 'ca', false);
    createCertificate('other-client', 'ca', false);
    createCertificate('untrusted-client', 'untrusted-ca', false);
    fs.mkdirSync(fixture('logs'));
    fs.writeFileSync(fixture('videos.txt'), 'nested/a/same.mp4\nnested/b/same.mp4\n__proto__\n');
    fs.writeFileSync(fixture('empty.pem'), '');
    fs.writeFileSync(fixture('invalid.pem'), 'not a certificate');
    const tlsArgs = ['--tlsCert', fixture('server.pem'), '--tlsKey', fixture('server.key'),
        '--tlsCa', fixture('ca.pem'), '--inputFile', fixture('videos.txt'), '--logDir', fixture('logs')];
    const port = await unusedPort();
    const allowedOrigin = 'https://localhost:' + port;
    const args = tlsArgs.concat(['--allowedOrigin', allowedOrigin]);

    await test('real process fails closed for missing/invalid origin and TLS credentials before state startup', async function() {
        const cases = [[], ['--tlsCert', fixture('server.pem')],
            tlsArgs, args.concat(['--allowedOrigin', allowedOrigin]),
            args.concat(['--tlsCert', fixture('server.pem')]),
            args.concat(['--tlsKey', fixture('missing.key')]),
            args.map(function(value) { return value === fixture('ca.pem') ? fixture('empty.pem') : value; }),
            args.map(function(value) { return value === fixture('ca.pem') ? fixture('invalid.pem') : value; }),
            args.map(function(value) { return value === fixture('server.key') ? fixture('client.key') : value; })];
        ['http://localhost:8081', 'wss://localhost:8081', 'https://localhost:8081/',
            'https://localhost:8081?query', 'https://localhost:8081#fragment',
            'https://user@localhost:8081', 'https://LOCALHOST:8081', 'https://localhost:443',
            'null', ' https://localhost:8081'].forEach(function(origin) {
            cases.push(tlsArgs.concat(['--allowedOrigin', origin]));
        });
        cases.forEach(function(options) {
            const result = childProcess.spawnSync(process.execPath, ['app.js'].concat(options),
                {cwd: root, encoding: 'utf8', timeout: 5000});
            assert(!result.error, String(result.error));
            assert.notStrictEqual(result.status, 0, 'Invalid TLS configuration succeeded');
            assert.strictEqual(fs.readdirSync(fixture('logs')).length, 0, 'Startup state changed before TLS validation');
        });
    });

    server = childProcess.spawn(process.execPath, ['app.js'].concat(args,
        ['--host', '127.0.0.1', '--port', String(port), '--outputPath', fixture('outputList.txt')]),
        {cwd: root, stdio: ['ignore', 'pipe', 'pipe']});
    server.stdout.on('data', function(data) { serverOutput += data; });
    server.stderr.on('data', function(data) { serverOutput += data; });
    assert.strictEqual(await ready(port), 200);
    const processBase = 'wss://localhost:' + port + '/registerProcess';
    const guiBase = 'wss://localhost:' + port + '/snvaStatus';

    await test('HTTPS and both WebSocket endpoints require trusted mutual TLS; no plaintext fallback', async function() {
        assert.strictEqual(await httpsRequest(port, tlsOptions('client')), 200);
        await rejects(httpsRequest(port, tlsOptions()), 'HTTPS without client certificate');
        await rejects(httpsRequest(port, tlsOptions('untrusted-client')), 'HTTPS with untrusted client');
        await rejects(httpsRequest(port, Object.assign(tlsOptions('client'), {ca: contents('untrusted-ca.pem')})),
            'HTTPS with untrusted server CA');
        await rejects(httpsRequest(port, Object.assign(tlsOptions('client'), {servername: 'wrong-host'})),
            'HTTPS hostname validation');
        await rejects(httpsRequest(port, Object.assign(tlsOptions('client'), {minVersion: 'TLSv1', maxVersion: 'TLSv1.1'})),
            'TLS below 1.2');
        for (const endpoint of [processBase, guiBase]) {
            await rejects(connection(endpoint, tlsOptions()).open, endpoint + ' without client certificate');
            await rejects(connection(endpoint, tlsOptions('untrusted-client')).open, endpoint + ' with untrusted client');
        }
        await rejects(connection('ws://localhost:' + port + '/registerProcess', {handshakeTimeout: 2000}).open,
            'plaintext WebSocket');
    });

    await test('hostile/malformed Origin is refused on GUI and processor; exact configured Origin is accepted', async function() {
        for (const endpoint of [processBase, guiBase]) {
            for (const origin of ['https://attacker.invalid', 'null', allowedOrigin + '/',
                allowedOrigin.replace('https:', 'http:'), allowedOrigin + '?query'])
                await rejects(connection(endpoint, Object.assign(tlsOptions('client'), {origin: origin})).open,
                    endpoint + ' with origin ' + origin);
            await rejects(connection(endpoint, Object.assign(tlsOptions('client'),
                {headers: {Origin: [allowedOrigin, allowedOrigin]}})).open, endpoint + ' with duplicate Origin');
        }
        await rejects(connection(guiBase, tlsOptions('client')).open, 'GUI without Origin');
        const browserProcessor = connection(processBase + '?id=unknown',
            Object.assign(tlsOptions('client'), {origin: allowedOrigin}));
        await timeout(browserProcessor.open, 'same-origin browser processor');
        assert.strictEqual(await timeout(browserProcessor.close, 'same-origin upgrade then invalid reconnect'), 1008);
    });

    const gui = connection(guiBase, Object.assign(tlsOptions('client'), {origin: allowedOrigin}));
    await timeout(gui.open, 'GUI connection');
    const initialGui = await gui.next();
    assert.strictEqual(initialGui.videosRemaining, 3);
    const owner = connection(processBase, tlsOptions('client'));
    await timeout(owner.open, 'processor connection');
    const success = await owner.next();
    assert.strictEqual(success.action, 'CONNECTION_SUCCESS');
    assert(/^[a-f0-9]{64}$/.test(success.reconnectToken));

    await test('actual reconnects reject prototype keys, duplicate fields, wrong tokens and wrong certificate identity', async function() {
        const token = success.reconnectToken;
        for (const query of ['?id=unknown', '?id=__proto__', '?id=constructor', '?id=toString', '?id=',
            '?id=' + success.id, '?reconnectToken=' + token,
            '?id=' + success.id + '&reconnectToken=' + '0'.repeat(64),
            '?id=' + success.id + '&id=' + success.id + '&reconnectToken=' + token,
            '?id=' + success.id + '&reconnectToken=' + token + '&reconnectToken=' + token])
            await invalidReconnect(processBase, query);
        await invalidReconnect(processBase, '?id=' + success.id + '&reconnectToken=' + token, 'other-client');
        assert.strictEqual(server.exitCode, null);
    });

    await test('valid reconnect retains ownership, terminal receipt retries are idempotent, and shutdown ledger is JSONL', async function() {
        owner.send({action: 'REQUEST_VIDEO'});
        const request = await owner.next();
        assert.strictEqual(request.action, 'PROCESS');
        assert.strictEqual(request.path, '__proto__');
        const replacement = connection(processBase + '?id=' + success.id + '&reconnectToken=' +
            success.reconnectToken, tlsOptions('client'));
        await timeout(replacement.open, 'authenticated reconnect');
        const reconnected = await replacement.next();
        assert.strictEqual(reconnected.id, success.id);
        assert.strictEqual(reconnected.reconnectToken, success.reconnectToken);
        assert.strictEqual(await timeout(owner.close, 'superseded socket close'), 1008);
        const foreign = connection(processBase, tlsOptions('other-client'));
        await timeout(foreign.open, 'second processor');
        const foreignSuccess = await foreign.next();
        foreign.send({action: 'REQUEST_VIDEO'});
        const second = await foreign.next();
        assert.strictEqual(second.action, 'PROCESS');
        assert.strictEqual(second.path, 'nested/b/same.mp4');
        foreign.send({action: 'REQUEST_RECEIVED', video: request.path, assignmentId: request.assignmentId});
        foreign.send({action: 'COMPLETE', video: request.path, assignmentId: request.assignmentId, output: 'forged'});
        replacement.send({action: 'REQUEST_RECEIVED', video: request.path, assignmentId: request.assignmentId});
        replacement.send({action: 'REQUEST_RECEIVED', video: request.path, assignmentId: request.assignmentId});
        const output = 'output: "quote"\\path\r\n{"video":"forged"}\t\u0000';
        replacement.send({action: 'COMPLETE', video: request.path, assignmentId: request.assignmentId, output: output});
        const firstReceipt = await replacement.next();
        assert.deepStrictEqual(firstReceipt, {action: 'COMPLETE_ACCEPTED', video: request.path,
            assignmentId: request.assignmentId});
        // Discard the first receipt at the application boundary and retry the exact payload.
        replacement.send({action: 'COMPLETE', video: request.path, assignmentId: request.assignmentId, output: output});
        assert.deepStrictEqual(await replacement.next(), firstReceipt);
        replacement.send({action: 'REQUEST_VIDEO'});
        const third = await replacement.next();
        assert.strictEqual(third.action, 'PROCESS');
        assert.strictEqual(third.path, 'nested/a/same.mp4');
        replacement.send({action: 'COMPLETE', video: request.path, assignmentId: request.assignmentId, output: 'replayed'});
        replacement.send({action: 'COMPLETE', video: 'same.mp4', assignmentId: third.assignmentId, output: 'basename-forged'});
        replacement.send({action: 'COMPLETE', video: second.path, assignmentId: second.assignmentId, output: 'foreign-forged'});
        foreign.send({action: 'COMPLETE', video: second.path, assignmentId: second.assignmentId, output: 'second'});
        const secondReceipt = await foreign.next();
        assert.deepStrictEqual(secondReceipt, {action: 'COMPLETE_ACCEPTED', video: second.path,
            assignmentId: second.assignmentId});
        assert.strictEqual((await foreign.next()).action, 'SHUTDOWN');
        // Simulate losing terminal messages, then reconnect while another processor still owns work.
        foreign.ws.close();
        await timeout(foreign.close, 'terminal processor disconnect');
        const terminal = connection(processBase + '?id=' + foreignSuccess.id + '&reconnectToken=' +
            foreignSuccess.reconnectToken, tlsOptions('other-client'));
        await timeout(terminal.open, 'terminal reconnect');
        assert.strictEqual((await terminal.next()).action, 'CONNECTION_SUCCESS');
        assert.strictEqual((await terminal.next()).action, 'SHUTDOWN');
        terminal.send({action: 'REQUEST_VIDEO'});
        terminal.send({action: 'STATUS_REPORT'});
        terminal.send({action: 'COMPLETE', video: request.path, assignmentId: request.assignmentId, output: output});
        terminal.send({action: 'COMPLETE', video: second.path, assignmentId: second.assignmentId, output: 'changed'});
        terminal.send({action: 'COMPLETE', video: second.path, assignmentId: second.assignmentId, output: 'second'});
        assert.deepStrictEqual(await terminal.next(), secondReceipt);
        replacement.send({action: 'COMPLETE', video: third.path, assignmentId: third.assignmentId, output: 'third'});
        const thirdReceipt = await replacement.next();
        assert.deepStrictEqual(thirdReceipt, {action: 'COMPLETE_ACCEPTED', video: third.path,
            assignmentId: third.assignmentId});
        assert.strictEqual((await replacement.next()).action, 'SHUTDOWN');
        const status = await gui.next();
        assert(!JSON.stringify(status).includes('reconnectToken'));
        const expected = Object.create(null);
        expected[request.path] = output;
        expected[second.path] = 'second';
        expected[third.path] = 'third';
        // The second final reconnect must work within the bounded five-second exit window.
        replacement.ws.close();
        await timeout(replacement.close, 'final processor disconnect');
        const final = connection(processBase + '?id=' + success.id + '&reconnectToken=' +
            success.reconnectToken, tlsOptions('client'));
        await timeout(final.open, 'final terminal reconnect');
        assert.strictEqual((await final.next()).action, 'CONNECTION_SUCCESS');
        assert.strictEqual((await final.next()).action, 'SHUTDOWN');
        final.send({action: 'COMPLETE', video: third.path, assignmentId: third.assignmentId, output: 'third'});
        assert.deepStrictEqual(await final.next(), thirdReceipt);
        final.send({action: 'REQUEST_VIDEO'});
        final.send({action: 'STATUS_REPORT'});
        const exited = new Promise(function(resolve) { server.once('exit', resolve); });
        await new Promise(function(resolve, reject) {
            const timer = setTimeout(function() { reject(new Error('Normal shutdown did not exit')); }, 8000);
            exited.then(function(code) { clearTimeout(timer); assert.strictEqual(code, 0); resolve(); });
        });
        const lines = contents('outputList.txt').toString().trim().split('\n');
        assert.strictEqual(lines.length, 3);
        lines.forEach(function(line) {
            assert(!line.includes('\r'));
            const entry = JSON.parse(line);
            assert.deepStrictEqual(Object.keys(entry), ['video', 'output']);
            assert.strictEqual(entry.output, expected[entry.video]);
            delete expected[entry.video];
        });
        assert.strictEqual(Object.keys(expected).length, 0);
        assert.deepStrictEqual(terminal.received.map(function(msg) { return msg.action; }),
            ['CONNECTION_SUCCESS', 'SHUTDOWN', 'COMPLETE_ACCEPTED']);
        assert.deepStrictEqual(final.received.map(function(msg) { return msg.action; }),
            ['CONNECTION_SUCCESS', 'SHUTDOWN', 'COMPLETE_ACCEPTED']);
    });
    console.log(passed + ' live TLS integration tests passed');
}

run().catch(function(error) {
    console.error(error.stack);
    if (serverOutput) console.error(serverOutput);
    process.exitCode = 1;
}).then(function() {
    channels.forEach(function(channel) { channel.ws.terminate(); });
    if (server && server.exitCode === null) server.kill();
    // Node 11-compatible recursive cleanup; no committed TLS fixture keys.
    function remove(directory) {
        fs.readdirSync(directory).forEach(function(name) {
            const filename = path.join(directory, name);
            if (fs.lstatSync(filename).isDirectory()) remove(filename);
            else fs.unlinkSync(filename);
        });
        fs.rmdirSync(directory);
    }
    remove(temporary);
});
