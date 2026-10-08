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
node app.js -i /path/to/list/of/videos.txt --tlsCa /secure/control/ca.pem \
  --tlsCert /secure/control/cert.pem --tlsKey /secure/control/key.pem \
  --allowedOrigin https://control.example.org:8081
```

To start via docker use:
```
sudo docker run -p 8081:8081 \
  --mount type=bind,src=/path/to/list/of/videos.txt,dst=/usr/config/Input.txt,readonly \
  --mount type=bind,src=/path/to/log/directory,dst=/usr/logs \
  --mount type=bind,src=/path/to/output/directory,dst=/usr/output \
  --mount type=bind,src=/secure/control,dst=/run/secrets/tls,readonly \
  -d control-node --inputFile /usr/config/Input.txt --logDir /usr/logs \
  --outputPath /usr/output/outputList.txt --tlsCa /run/secrets/tls/ca.pem \
  --tlsCert /run/secrets/tls/cert.pem --tlsKey /run/secrets/tls/key.pem \
  --allowedOrigin https://control.example.org:8081
```

The list of videos contains paths relative to the processor input directory, separated by newlines. Nested paths are retained as assignment identities; two videos in different directories may share a basename.

The server requires TLS 1.2 or newer and a client certificate signed by `--tlsCa` for both processors and GUI users. Provision a dedicated SNVA client CA and protect private keys outside the repository; any certificate issued by this CA grants access to these endpoints. See the [deployment instructions](../README.md#deployment) for certificate names, inference TLS, and read-only mounts. Startup fails if credentials are missing or invalid; no plaintext fallback exists.

Processors connect at `wss://<control-host>:8081/registerProcess`. Registration returns an ID and secret `reconnectToken`; reconnect must provide both query parameters using the same client certificate. Unknown IDs, invalid credentials, duplicate query fields, and stale sockets are rejected without mutating run state. Upgrade processors and control together; legacy clients are intentionally not supported by the secured protocol.

Each `PROCESS` message carries a fresh `assignmentId`. `REQUEST_RECEIVED` and `COMPLETE` must echo both this ID and the exact dispatched video path. Only the owning processor and current assignment can acknowledge or complete work; stale/replayed messages cannot clear another assignment's timeout or fabricate completion records.

Retain completion results until the controller replies with `COMPLETE_ACCEPTED` containing the same path and assignment ID. An identical authenticated retry resends the receipt without changing ledger state. Completed processors can reconnect to recover their receipt and terminal `SHUTDOWN`; expired sessions remain revoked. Browser requests must carry the exact HTTPS origin configured by `--allowedOrigin`; GUI connections without Origin and foreign-Origin processor requests are rejected. Native Python clients omit Origin.

After all work finishes, the controller requests processor shutdown and writes the final `outputList.txt` as JSON Lines, not the previous colon-delimited format. Each physical line is a JSON object with `video` and `output` strings; parse it with a JSON parser. Control characters and delimiters are safely escaped. `output` remains the processor's existing textual representation of its report-location list.


Flag | Short Flag | Properties | Description
:------:|:---------------:|:---------------------:|:-----------:
--inputFile|-i|type=string, default='./videopaths.txt'|Text File containing a list of video paths separated by newlines
--outputPath|-op|type=string, default='./outputList.txt'|JSON Lines manifest written at successful shutdown
--nodes|-n|type=string, default='./nodes.json| JSON file containing a list of nodes to use as analyzers or processors. Should be an array of objects formatted as: {"node":"nodeLocation", "gpuEnabled":"true\|false"}. Functionality based on this argument is incomplete.
--analyzerCount|-a|type=int, default=2|Number of analyzer nodes to generate. Functionality based on this argument is incomplete.
--logDir|-l|type=string, default=./logs|Directory to save log files.
--port|-p|type=int, default=8081|Port which server should listen on.
--host||type=string, default=0.0.0.0|Listening interface (not a client connection hostname)
--tlsCa||required=True|PEM CA bundle trusted for processor and GUI client certificates
--tlsCert||required=True|PEM server certificate chain with a SAN matching the control hostname
--tlsKey||required=True|PEM private key matching the server certificate
--allowedOrigin||required=True|Exact canonical HTTPS GUI origin, e.g. https://control.example.org:8081 (no trailing slash; omit default :443)

## GUI

A web-based monitoring GUI is available at `https://<control-host>:<port>/snvaGui.html`. Import a trusted GUI client certificate and its private key into the browser/OS certificate store first; also trust the control server CA. The page uses the same HTTPS origin for its `wss://.../snvaStatus` connection, including default ports and IPv6. HTTP pages and clients without a trusted certificate are not supported. The status stream includes assigned work but never reconnect credentials or live socket state.

Run `npm ci` followed by `npm test` to execute dependency-free fake-timer unit tests and live HTTPS/WebSocket child-process regression tests. OpenSSL must be available; test certificates/keys are generated in temporary directories and removed afterward. No model or GPU deployment is needed.
