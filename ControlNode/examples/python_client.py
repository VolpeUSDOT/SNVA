import asyncio
import json
import websockets as ws

async def run():
  async with ws.connect('ws://localhost:8080/registerProcess') as connection:
    response = await connection.recv()
    response = json.loads(response)
    print(response)

    if response['action'] != 'CONNECTION_SUCCESS':
      raise ConnectionError(
        'control node connection failed with response: {}'.format(response))

    print('requesting video')
    request = json.dumps({'action':'REQUEST_VIDEO'})
    await connection.send(request)

    print('reading response')
    response = await connection.recv()
    response = json.loads(response)

    print(response)

    # if response['action'] != 'PROCESS':
    #   raise ConnectionError(
    #     'control node connection failed with response: {}'.format(response))

asyncio.get_event_loop().run_until_complete(run())
try:
  asyncio.get_event_loop().run_until_complete(run())
except Exception as e:
  print(e)
  exit()


