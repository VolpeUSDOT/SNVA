import asyncio
import argparse
import json
import ssl
import websockets as ws

parser = argparse.ArgumentParser()
parser.add_argument('--host', default='localhost')
parser.add_argument('--port', type=int, default=8081)
parser.add_argument('--tlsCert', required=True)
parser.add_argument('--tlsKey', required=True)
parser.add_argument('--tlsCa', required=True)
args = parser.parse_args()
tls_context = ssl.create_default_context(cafile=args.tlsCa)
if hasattr(ssl, 'TLSVersion'):
  tls_context.minimum_version = ssl.TLSVersion.TLSv1_2
else:
  tls_context.options |= ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1
tls_context.load_cert_chain(args.tlsCert, args.tlsKey)

async def run():
  async with ws.connect('wss://{}:{}/registerProcess'.format(args.host, args.port),
                        ssl=tls_context) as connection:
    response = await connection.recv()
    response = json.loads(response)
    print({'action': response.get('action'), 'id': response.get('id')})

    if response['action'] != 'CONNECTION_SUCCESS':
      raise ConnectionError(
        'control node connection failed with response: {}'.format(response))

    # Retain both id and reconnectToken for authenticated reconnect query parameters.

    print('requesting video')
    request = json.dumps({'action':'REQUEST_VIDEO'})
    await connection.send(request)

    print('reading response')
    response = await connection.recv()
    response = json.loads(response)

    print(response)

    if response['action'] == 'PROCESS':
      await connection.send(json.dumps({
        'action': 'REQUEST_RECEIVED',
        'video': response['path'],
        'assignmentId': response['assignmentId']}))
      # COMPLETE must echo the same original path and assignmentId, plus string output.
      # Retain/replay it until matching COMPLETE_ACCEPTED, buffering other messages.

try:
  asyncio.get_event_loop().run_until_complete(run())
except Exception as e:
  print(e)
  exit()


