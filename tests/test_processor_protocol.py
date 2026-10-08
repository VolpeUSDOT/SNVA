"""Exercise the control-client lifecycle without media, TensorFlow or sockets."""

import ast
import asyncio
from collections import deque
from contextlib import ExitStack
import importlib.util
import inspect
import json
import logging
from logging.handlers import QueueHandler
import multiprocessing
import os
from pathlib import Path
from queue import Empty, Queue
import ssl
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock
from urllib.parse import parse_qs, urlsplit


ROOT = Path(__file__).resolve().parents[1]


class ConnectionClosed(Exception):
  pass


class ScriptExhausted(AssertionError):
  """Fail immediately instead of hanging on an unexpected receive/retry."""


class FakeQueue(Queue):
  def __init__(self):
    super().__init__()
    self.closed = False
    self.feeder_joined = False

  def close(self):
    self.closed = True

  def join_thread(self):
    self.feeder_joined = True

  def cancel_join_thread(self):
    pass


class FakeThread:
  def __init__(self, harness, target, args, **kwargs):
    self.harness = harness
    self.target = target
    self.args = args
    self.started = False
    self.joined = False
    self.alive = False
    self.daemon = kwargs.get('daemon', False)

  def start(self):
    self.started = True
    self.alive = True

  def join(self, timeout=None):
    self.joined = True
    self.harness.events.append(('forwarder-join', self))
    # Execute the real forwarding function only when its sentinel is available.
    with self.args[1].mutex:
      can_drain = None in self.args[1].queue
    if can_drain:
      self.target(*self.args)
      self.alive = False

  def is_alive(self):
    return self.alive


class FakeProcess:
  def __init__(self, harness, target, name, args):
    self.harness = harness
    self.target = target
    self.name = name
    self.args = args
    self.pid = 100000 + len(harness.processes)
    self.started = False
    self.joined = False
    self.alive = False
    self.terminated = False
    self.exitcode = None

  def start(self):
    self.started = True
    self.alive = True
    self.target(*self.args)

  def join(self, timeout=None):
    self.joined = True
    self.harness.events.append(('worker-join', self))

  def is_alive(self):
    return self.alive

  def terminate(self):
    self.terminated = True
    self.alive = False
    self.exitcode = -15
    self.harness.events.append(('worker-terminate', self))
    self.args[8].put(logging.LogRecord(
      'synthetic.worker', logging.WARNING, __file__, 1,
      'late worker log during termination', (), None))

  def kill(self):
    self.terminate()


class FakeConnection:
  def __init__(self, messages, fail_complete_on=None, on_receive=None,
               accept_completions=True, on_send=None):
    self.messages = deque(messages)
    self.fail_complete_on = fail_complete_on
    self.complete_count = 0
    self.on_receive = on_receive
    self.accept_completions = accept_completions
    self.on_send = on_send
    self.received = []
    self.attempts = []
    self.sent = []
    self.entered = False
    self.exited = False

  async def __aenter__(self):
    self.entered = True
    return self

  async def __aexit__(self, *args):
    self.exited = True

  async def recv(self):
    if not self.messages:
      raise ScriptExhausted('main() received beyond the scripted commands')
    message = self.messages.popleft()
    if isinstance(message, BaseException):
      raise message
    self.received.append(message)
    if self.on_receive is not None:
      self.on_receive(message)
    return json.dumps(message)

  async def send(self, payload):
    message = json.loads(payload)
    self.attempts.append(message)
    if message['action'] == 'COMPLETE':
      self.complete_count += 1
      if self.complete_count == self.fail_complete_on:
        raise ConnectionClosed('synthetic disconnect during COMPLETE')
    self.sent.append(message)
    if self.on_send is not None:
      self.on_send(message)
    if message['action'] == 'COMPLETE' and self.accept_completions:
      # A receipt cannot overtake commands already sent by the controller.
      self.messages.append(accepted(message['video'], message['assignmentId']))


def connected(connection_id='processor-7', token='reconnect-secret'):
  return {'action': 'CONNECTION_SUCCESS', 'id': connection_id,
          'reconnectToken': token}


def assignment(video_path, assignment_id):
  return {'action': 'PROCESS', 'path': video_path,
          'assignmentId': assignment_id}


def accepted(video_path, assignment_id):
  return {'action': 'COMPLETE_ACCEPTED', 'video': video_path,
          'assignmentId': assignment_id}


class MainHarness:
  def __init__(self, test, mode='workzone', num_processes=3, deferred=False):
    self.temp_dir = tempfile.TemporaryDirectory()
    test.addCleanup(self.temp_dir.cleanup)
    home = Path(self.temp_dir.name)
    model = home / 'models' / 'synthetic-model'
    model.mkdir(parents=True)
    (model / 'input_size.txt').write_text('224\n', encoding='ascii')
    (home / 'models' / 'class_names.txt').write_text('0:synthetic\n',
                                                   encoding='ascii')
    (home / 'input').mkdir()
    self.processes = []
    self.threads = []
    self.worker_calls = []
    self.deferred = deferred
    self.connections = deque()
    self.connect_calls = []
    self.sleep_calls = 0
    self.wait_calls = 0
    self.wait_timeouts = []
    self.clock = 0
    self.duration_messages = []
    self.events = []
    self.factory_contexts = []

    io_module = ModuleType('utils.io')
    io_module.IO = SimpleNamespace(
      read_class_names=mock.Mock(return_value={0: 'synthetic'}),
      get_processing_duration=self.processing_duration)
    self.io = io_module.IO
    processor_module = ModuleType('utils.processor')
    processor_module.process_video = self.workzone_worker
    processor_module.process_video_signalstate = self.signalstate_worker
    ws_module = ModuleType('websockets')
    ws_module.connect = self.connect
    ws_module.exceptions = SimpleNamespace(ConnectionClosed=ConnectionClosed)
    spec = importlib.util.spec_from_file_location(
      '_snva_protocol_test', ROOT / 'snva.py')
    self.module = importlib.util.module_from_spec(spec)
    websocket_logger = logging.getLogger('websockets')
    old_level = websocket_logger.level
    test.addCleanup(websocket_logger.setLevel, old_level)
    with mock.patch.dict(sys.modules, {'utils.io': io_module,
                                      'utils.processor': processor_module,
                                      'websockets': ws_module}):
      spec.loader.exec_module(self.module)

    self.module.snva_home = str(home)
    self.module.snva_version_string = 'synthetic-test'
    self.module.log_level = logging.DEBUG
    self.module.log_queue = FakeQueue()
    self.module.main_interrupt_queue = FakeQueue()
    # Use production parser defaults so new required integration options surface.
    tree = ast.parse((ROOT / 'snva.py').read_text(encoding='utf-8'))
    entrypoint = next(node for node in tree.body if isinstance(node, ast.If)
                      and isinstance(node.test, ast.Compare)
                      and isinstance(node.test.left, ast.Name)
                      and node.test.left.id == '__name__')
    parser_nodes = []
    for node in entrypoint.body:
      if (isinstance(node, ast.Assign)
          and any(isinstance(target, ast.Name) and target.id == 'args'
                  for target in node.targets)):
        break
      parser_nodes.append(node)
    namespace = {'argparse': self.module.argparse}
    exec(compile(ast.Module(body=parser_nodes, type_ignores=[]),
                 str(ROOT / 'snva.py'), 'exec'), namespace)
    self.module.args = namespace['parser'].parse_args([
      '--inputpath', str(home / 'input'),
      '--modelsdirpath', 'models', '--modelname', 'synthetic-model',
      '--outputpath', str(home / 'reports'),
      '--controlnodehost', 'wss://controller.example:8443',
      '--modelserverhost', 'inference.example:8500',
      '--numprocesses', str(num_processes), '--processormode', mode,
      '--tls-ca', str(home / 'ca.pem'),
      '--tls-cert', str(home / 'client.pem'),
      '--tls-key', str(home / 'private-key.pem')])
    self.context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    self.context_factory = mock.Mock(return_value=self.context)
    self.kill = mock.Mock(side_effect=ProcessLookupError)

  def processing_duration(self, duration, message):
    self.duration_messages.append(message)
    return '{} {:.2f}s'.format(message, duration)

  def tick(self):
    self.clock += 1
    return self.clock

  def workzone_worker(self, *args):
    self.worker_calls.append(('workzone', args))
    self.worker_finished(args)

  def signalstate_worker(self, *args):
    self.worker_calls.append(('signalstate', args))
    self.worker_finished(args)

  def worker_finished(self, args):
    if not self.deferred:
      self.finish(args)

  def finish(self, args):
    args[7].put({'return_code': 'success', 'return_value': 12,
                 'analysis_duration': 0.25,
                 'output_locations': 'reports/{}.csv'.format(len(self.worker_calls))})
    args[8].put(None)
    for child in self.processes:
      if child.args[7] is args[7]:
        child.alive = False
        child.exitcode = 0

  def process(self, target, name, args):
    child = FakeProcess(self, target, name, args)
    self.processes.append(child)
    return child

  def thread(self, target, args, **kwargs):
    thread = FakeThread(self, target, args, **kwargs)
    self.threads.append(thread)
    return thread

  def connect(self, url, **kwargs):
    self.connect_calls.append((url, kwargs))
    if not self.connections:
      raise ScriptExhausted('main() reconnected beyond the scripted connections')
    connection = self.connections.popleft()
    if isinstance(connection, BaseException):
      raise connection
    return connection

  def context_process(self, context, *args, **kwargs):
    self.factory_contexts.append(('Process', context.get_start_method()))
    return self.process(*args, **kwargs)

  def context_queue(self, context, *args, **kwargs):
    self.factory_contexts.append(('Queue', context.get_start_method()))
    return FakeQueue()

  async def no_sleep(self, duration):
    self.sleep_calls += 1
    if self.sleep_calls > 20:
      raise ScriptExhausted('main() stalled while draining worker result queues')

  async def immediate_wait(self, awaitable, timeout):
    self.wait_calls += 1
    self.wait_timeouts.append(timeout)
    return await awaitable

  async def run(self, *connections):
    self.connections.extend(connections)
    with ExitStack() as patches:
      for name, replacement in (('Process', self.process), ('Thread', self.thread),
                                ('Queue', FakeQueue),
                                ('create_client_context', self.context_factory),
                                ('time', self.tick)):
        patches.enter_context(mock.patch.object(self.module, name, replacement,
                                                 create=True))
      # Preserve real context selection while replacing only its heavy factories.
      for method in multiprocessing.get_all_start_methods():
        context_type = type(multiprocessing.get_context(method))
        patches.enter_context(mock.patch.object(
          context_type, 'Process',
          lambda context, *args, **kwargs: self.context_process(context, *args, **kwargs)))
        patches.enter_context(mock.patch.object(
          context_type, 'Queue',
          lambda context, *args, **kwargs: self.context_queue(context, *args, **kwargs)))
      patches.enter_context(mock.patch.object(self.module.signal, 'signal'))
      patches.enter_context(mock.patch.object(self.module.os, 'kill', self.kill))
      patches.enter_context(mock.patch.object(self.module.asyncio, 'sleep', self.no_sleep))
      patches.enter_context(mock.patch.object(self.module.asyncio, 'wait_for', self.immediate_wait))
      patches.enter_context(mock.patch.dict(os.environ, {
        'FFMPEG_HOME': '/fake/ffmpeg', 'FFPROBE_HOME': '/fake/ffprobe'}))
      try:
        await self.module.main()
      finally:
        self.events.append(('main-exit', None))


class ProcessorProtocolTests(unittest.IsolatedAsyncioTestCase):
  async def test_nested_paths_same_basename_echo_original_assignment(self):
    harness = MainHarness(self)
    assignments = [assignment('camera-a/day/clip.mp4', 'assignment-a'),
                   assignment('camera-b/night/clip.mp4', 'assignment-b')]
    connection = FakeConnection([
      connected(), *assignments, {'action': 'SHUTDOWN'}])
    with self.assertLogs(level=logging.DEBUG):
      await harness.run(connection)

    acknowledgments = [message for message in connection.sent
                       if message['action'] == 'REQUEST_RECEIVED']
    completions = [message for message in connection.sent
                   if message['action'] == 'COMPLETE']
    expected = [(message['path'], message['assignmentId'])
                for message in assignments]
    self.assertEqual([(message['video'], message['assignmentId'])
                      for message in acknowledgments], expected)
    self.assertEqual([(message['video'], message['assignmentId'])
                      for message in completions], expected)
    self.assertEqual([(message['video'], message['assignmentId'])
                      for message in connection.received
                      if message['action'] == 'COMPLETE_ACCEPTED'], expected)
    self.assertEqual(len(harness.processes), 2)
    self.assertEqual([child.name for child in harness.processes], ['clip', 'clip'])
    for child, command in zip(harness.processes, assignments):
      self.assertEqual(child.args[0],
                       os.path.join(harness.module.args.inputpath, command['path']))
      self.assertTrue(child.started)
      self.assertTrue(child.joined)
      self.assertTrue(child.args[7].closed)
    self.assertTrue(all(thread.started and thread.joined
                        for thread in harness.threads))
    self.assertTrue(connection.exited)
    self.assertIn('2 videos and 24 frames', harness.duration_messages[-1])

  async def test_both_worker_modes_receive_trailing_model_tls_arguments(self):
    for mode in ('workzone', 'signalstate'):
      with self.subTest(mode=mode):
        harness = MainHarness(self, mode=mode)
        connection = FakeConnection([
          connected(), assignment('nested/clip.mp4', 'tls-' + mode),
          {'action': 'SHUTDOWN'}])
        with self.assertLogs(level=logging.DEBUG):
          await harness.run(connection)
        self.assertEqual(len(harness.worker_calls), 1)
        worker_mode, args = harness.worker_calls[0]
        self.assertEqual(worker_mode, mode)
        self.assertEqual(args[-4:], (mode, harness.module.args.tls_ca,
                                     harness.module.args.tls_cert,
                                     harness.module.args.tls_key))
        self.assertEqual(args[5], 'inference.example:8500')
        self.assertEqual(args[6], 224)
        # Bind the real worker signature without importing its TensorFlow stack.
        tree = ast.parse((ROOT / 'utils' / 'processor.py').read_text(encoding='utf-8'))
        function_name = ('process_video_signalstate' if mode == 'signalstate'
                         else 'process_video')
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == function_name)
        function.body = [ast.Return(value=ast.Constant(value=None))]
        ast.fix_missing_locations(function)
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]),
                     str(ROOT / 'utils' / 'processor.py'), 'exec'), namespace)
        bound = inspect.signature(namespace[function_name]).bind(*args)
        self.assertEqual(bound.arguments['tls_ca'], harness.module.args.tls_ca)
        self.assertEqual(bound.arguments['tls_cert'], harness.module.args.tls_cert)
        self.assertEqual(bound.arguments['tls_key'], harness.module.args.tls_key)

  async def test_wss_context_and_reconnect_query_do_not_log_token(self):
    harness = MainHarness(self)
    connection_id = 'processor/id +7'
    token = 'private-reconnect/secret?+&='
    rotated_token = 'rotated-reconnect/secret?+&='
    first = FakeConnection([connected(connection_id, token), ConnectionClosed()])
    second = FakeConnection([
      connected(connection_id, rotated_token), ConnectionClosed()])
    third = FakeConnection([
      connected(connection_id, rotated_token), {'action': 'SHUTDOWN'}])
    with self.assertLogs(level=logging.DEBUG) as logs:
      await harness.run(first, second, third)

    self.assertEqual(len(harness.connect_calls), 3)
    for index, (url, kwargs) in enumerate(harness.connect_calls):
      parsed = urlsplit(url)
      self.assertEqual(parsed.scheme, 'wss')
      self.assertEqual(parsed.netloc, 'controller.example:8443')
      self.assertEqual(parsed.path, '/registerProcess')
      self.assertIs(kwargs['ssl'], harness.context)
      expected_query = {} if index == 0 else {
        'id': [connection_id],
        'reconnectToken': [token if index == 1 else rotated_token]}
      self.assertEqual(parse_qs(parsed.query), expected_query)
    harness.context_factory.assert_called_once_with(
      harness.module.args.tls_ca, harness.module.args.tls_cert,
      harness.module.args.tls_key)
    rendered_logs = '\n'.join(logs.output)
    self.assertNotIn(token, rendered_logs)
    self.assertNotIn(rotated_token, rendered_logs)
    for url, _ in harness.connect_calls[1:]:
      self.assertNotIn(urlsplit(url).query, rendered_logs)
    self.assertTrue(all(connection.exited for connection in (first, second, third)))

  async def test_idle_received_process_command_is_not_discarded(self):
    harness = MainHarness(self, deferred=True)
    commands = [assignment('first/clip.mp4', 'idle-first'),
                assignment('second/clip.mp4', 'idle-second')]

    def finish_on_shutdown(message):
      if message['action'] == 'SHUTDOWN':
        for _, args in harness.worker_calls:
          harness.finish(args)

    connection = FakeConnection([
      connected(), commands[0], {'action': 'CEASE_REQUESTS'}, commands[1],
      {'action': 'SHUTDOWN'}], on_receive=finish_on_shutdown)
    with self.assertLogs(level=logging.DEBUG):
      await harness.run(connection)

    self.assertEqual(len(harness.worker_calls), 2)
    self.assertGreaterEqual(harness.wait_calls, 1)
    self.assertEqual([message['assignmentId'] for message in connection.sent
                      if message['action'] == 'REQUEST_RECEIVED'],
                     ['idle-first', 'idle-second'])
    self.assertEqual([message['assignmentId'] for message in connection.sent
                      if message['action'] == 'COMPLETE'],
                     ['idle-first', 'idle-second'])
    # Idle receives unsolicited commands without making another video request.
    self.assertEqual(sum(message['action'] == 'REQUEST_VIDEO'
                         for message in connection.sent), 2)
    self.assertFalse(connection.messages)

  async def test_idle_resume_returns_to_requesting_work(self):
    harness = MainHarness(self)
    connection = FakeConnection([
      connected(), {'action': 'CEASE_REQUESTS'}, {'action': 'RESUME_REQUESTS'},
      assignment('resumed/clip.mp4', 'resumed-assignment'),
      {'action': 'SHUTDOWN'}])
    with self.assertLogs(level=logging.DEBUG):
      await harness.run(connection)
    self.assertEqual([message['action'] for message in connection.sent],
                     ['REQUEST_VIDEO', 'REQUEST_VIDEO', 'REQUEST_RECEIVED',
                      'REQUEST_VIDEO', 'COMPLETE'])
    self.assertEqual(len(harness.worker_calls), 1)

  async def test_idle_timeout_drains_completed_worker_then_receives_resume(self):
    harness = MainHarness(self)
    connection = FakeConnection([
      connected(), assignment('nested/clip.mp4', 'idle-timeout'),
      {'action': 'CEASE_REQUESTS'}, asyncio.TimeoutError(),
      {'action': 'RESUME_REQUESTS'}, {'action': 'SHUTDOWN'}])
    with self.assertLogs(level=logging.DEBUG):
      await harness.run(connection)
    self.assertEqual(harness.wait_timeouts.count(1), 1)
    self.assertEqual([message['action'] for message in connection.sent],
                     ['REQUEST_VIDEO', 'REQUEST_RECEIVED', 'REQUEST_VIDEO',
                       'COMPLETE'])
    self.assertTrue(harness.processes[0].joined)
    self.assertFalse(connection.messages)

  async def test_disconnect_during_complete_replays_result_after_reconnect(self):
    harness = MainHarness(self, num_processes=1)
    command = assignment('nested/finished.mp4', 'completed-assignment')
    first = FakeConnection([connected(), command], fail_complete_on=1)
    second = FakeConnection([connected(), {'action': 'SHUTDOWN'}])
    with self.assertLogs(level=logging.DEBUG):
      await harness.run(first, second)
    attempted = [message for message in first.attempts
                 if message['action'] == 'COMPLETE']
    delivered = [message for message in second.sent
                 if message['action'] == 'COMPLETE']
    self.assertEqual(len(attempted), 1)
    self.assertEqual(delivered, attempted)
    self.assertEqual(len(harness.worker_calls), 1)
    self.assertTrue(harness.processes[0].joined)
    self.assertIn('1 videos and 12 frames', harness.duration_messages[-1])

  async def test_disconnect_during_shutdown_drain_reconnects_before_exit(self):
    harness = MainHarness(self)
    first = FakeConnection([
      connected(), assignment('nested/finished.mp4', 'shutdown-assignment'),
      {'action': 'SHUTDOWN'}], fail_complete_on=1)
    second = FakeConnection([connected()])
    with self.assertLogs(level=logging.DEBUG):
      await harness.run(first, second)
    self.assertEqual(len(harness.connect_calls), 2)
    attempted = [message for message in first.attempts
                 if message['action'] == 'COMPLETE']
    delivered = [message for message in second.sent
                 if message['action'] == 'COMPLETE']
    self.assertEqual(delivered, attempted)
    self.assertTrue(harness.processes[0].joined)
    self.assertFalse(any(message['action'] == 'REQUEST_VIDEO'
                         for message in second.sent))

  async def test_successful_send_without_receipt_replays_after_connection_loss(self):
    harness = MainHarness(self, num_processes=1)
    command = assignment('nested/finished.mp4', 'unconfirmed-assignment')

    def lose_receipt(message):
      if message['action'] == 'COMPLETE':
        first.messages.append(ConnectionClosed('receipt never reached processor'))

    first = FakeConnection([connected(), command], accept_completions=False,
                           on_send=lose_receipt)
    second = FakeConnection([connected(), {'action': 'SHUTDOWN'}])

    def verify_pending_at_reconnect(message):
      if message['action'] == 'CONNECTION_SUCCESS':
        self.assertFalse(harness.processes[0].joined)
        self.assertFalse(harness.processes[0].args[7].closed)

    second.on_receive = verify_pending_at_reconnect
    with self.assertLogs(level=logging.DEBUG):
      await harness.run(first, second)
    first_complete = [message for message in first.sent
                      if message['action'] == 'COMPLETE']
    replay = [message for message in second.sent if message['action'] == 'COMPLETE']
    self.assertEqual(len(first_complete), 1)
    self.assertEqual(replay, first_complete)
    self.assertEqual(len(harness.worker_calls), 1)
    self.assertTrue(harness.processes[0].joined)
    self.assertIn(accepted(command['path'], command['assignmentId']), second.received)
    self.assertIn('1 videos and 12 frames', harness.duration_messages[-1])

  async def test_terminal_shutdown_before_replayed_receipt_is_buffered(self):
    harness = MainHarness(self, num_processes=1)
    command = assignment('terminal/clip.mp4', 'terminal-assignment')
    first = FakeConnection([connected(), command], accept_completions=False)

    def lose_receipt(message):
      if message['action'] == 'COMPLETE':
        first.messages.append(ConnectionClosed('controller accepted but receipt lost'))

    first.on_send = lose_receipt
    second = FakeConnection([connected(), {'action': 'SHUTDOWN'}])

    def verify_shutdown_does_not_release(message):
      if message['action'] == 'SHUTDOWN':
        self.assertEqual(second.sent[-1]['action'], 'COMPLETE')
        self.assertFalse(harness.processes[0].joined)
        self.assertFalse(harness.processes[0].args[7].closed)

    second.on_receive = verify_shutdown_does_not_release
    with self.assertLogs(level=logging.DEBUG):
      await harness.run(first, second)
    self.assertEqual([message['action'] for message in second.received],
                     ['CONNECTION_SUCCESS', 'SHUTDOWN', 'COMPLETE_ACCEPTED'])
    self.assertEqual([message['action'] for message in second.sent], ['COMPLETE'])
    self.assertTrue(harness.processes[0].joined)
    self.assertIn('1 videos and 12 frames', harness.duration_messages[-1])

  async def test_mismatched_receipts_cannot_release_current_result(self):
    harness = MainHarness(self, num_processes=1)
    command = assignment('owner/clip.mp4', 'owned-assignment')
    mismatched = [accepted(command['path'], 'other-owner-assignment'),
                  accepted('other/clip.mp4', command['assignmentId'])]
    first = FakeConnection([connected(), command], accept_completions=False)

    def inject_wrong_receipts(message):
      if message['action'] == 'COMPLETE':
        first.messages.extend([*mismatched, ConnectionClosed('no valid receipt')])

    def verify_not_released(message):
      if message['action'] == 'COMPLETE_ACCEPTED':
        self.assertFalse(harness.processes[0].joined)
        self.assertFalse(harness.processes[0].args[7].closed)

    first.on_send = inject_wrong_receipts
    first.on_receive = verify_not_released
    second = FakeConnection([connected(), {'action': 'SHUTDOWN'}])
    with self.assertLogs(level=logging.DEBUG):
      await harness.run(first, second)
    self.assertEqual([message for message in first.received
                      if message['action'] == 'COMPLETE_ACCEPTED'], mismatched)
    self.assertEqual([message for message in first.sent if message['action'] == 'COMPLETE'],
                     [message for message in second.sent if message['action'] == 'COMPLETE'])
    self.assertTrue(harness.processes[0].joined)
    self.assertIn('1 videos and 12 frames', harness.duration_messages[-1])

  async def test_receipt_wait_buffers_process_status_and_shutdown_in_order(self):
    harness = MainHarness(self, num_processes=1)
    commands = [assignment('first/clip.mp4', 'buffered-first'),
                assignment('second/clip.mp4', 'buffered-second')]
    connection = FakeConnection([
      connected(), commands[0], {'action': 'STATUS_REQUEST'}, commands[1],
      {'action': 'SHUTDOWN'}])

    def verify_second_assignment_after_first_receipt(message):
      if (message['action'] == 'REQUEST_RECEIVED'
          and message['assignmentId'] == 'buffered-second'):
        self.assertIn(accepted(commands[0]['path'], commands[0]['assignmentId']),
                      connection.received)

    connection.on_send = verify_second_assignment_after_first_receipt
    with self.assertLogs(level=logging.DEBUG) as logs:
      await harness.run(connection)
    self.assertEqual([message['action'] for message in connection.received],
                     ['CONNECTION_SUCCESS', 'PROCESS', 'STATUS_REQUEST', 'PROCESS',
                      'SHUTDOWN', 'COMPLETE_ACCEPTED', 'COMPLETE_ACCEPTED'])
    self.assertEqual([message['assignmentId'] for message in connection.sent
                      if message['action'] == 'REQUEST_RECEIVED'],
                     ['buffered-first', 'buffered-second'])
    self.assertEqual(len(harness.worker_calls), 2)
    self.assertTrue(all(child.joined for child in harness.processes))
    self.assertEqual(sum('control node requested status request' in message
                         for message in logs.output), 1)
    self.assertIn('2 videos and 24 frames', harness.duration_messages[-1])

  async def test_worker_and_result_queues_use_explicit_spawn_context(self):
    harness = MainHarness(self)
    connection = FakeConnection([
      connected(), assignment('nested/clip.mp4', 'spawn-assignment'),
      {'action': 'SHUTDOWN'}])
    with self.assertLogs(level=logging.DEBUG):
      await harness.run(connection)
    self.assertEqual([kind for kind, _ in harness.factory_contexts].count('Process'), 1)
    self.assertGreaterEqual([kind for kind, _ in harness.factory_contexts].count('Queue'), 2)
    self.assertTrue(all(method == 'spawn' for _, method in harness.factory_contexts))

  async def test_partial_completion_drain_preserves_totals_after_reconnect(self):
    harness = MainHarness(self, num_processes=2)
    commands = [assignment('first/clip.mp4', 'partial-first'),
                assignment('second/clip.mp4', 'partial-second')]
    first = FakeConnection([connected(), *commands], fail_complete_on=2)
    second = FakeConnection([connected(), {'action': 'SHUTDOWN'}])
    with self.assertLogs(level=logging.DEBUG) as logs:
      await harness.run(first, second)
    delivered = [message for connection in (first, second)
                 for message in connection.sent if message['action'] == 'COMPLETE']
    self.assertEqual([message['assignmentId'] for message in delivered],
                     ['partial-first', 'partial-second'])
    self.assertEqual(len(harness.worker_calls), 2)
    self.assertTrue(all(child.joined for child in harness.processes))
    self.assertIn('2 videos and 24 frames', harness.duration_messages[-1])
    self.assertIn('Video analysis alone spanned a cumulative 0.50 seconds',
                  '\n'.join(logs.output))

  async def test_invalid_or_duplicate_assignments_do_not_start_another_worker(self):
    invalid = [
      {'action': 'PROCESS', 'path': 'clip.mp4'},
      assignment('', 'empty-path'), assignment(None, 'missing-path'),
      assignment('clip.mp4', ''), assignment('clip.mp4', 7)]
    for command in invalid:
      with self.subTest(command=command):
        harness = MainHarness(self)
        connection = FakeConnection([connected(), command])
        with self.assertLogs(level=logging.DEBUG), \
             self.assertRaisesRegex(ValueError, 'invalid or duplicate'):
          await harness.run(connection)
        self.assertFalse(harness.worker_calls)
        self.assertFalse(any(message['action'] == 'REQUEST_RECEIVED'
                             for message in connection.sent))
    harness = MainHarness(self)
    command = assignment('nested/clip.mp4', 'same-assignment')
    connection = FakeConnection([connected(), command, command])
    with self.assertLogs(level=logging.DEBUG), \
         self.assertRaisesRegex(ValueError, 'invalid or duplicate'):
      await harness.run(connection)
    self.assertEqual(len(harness.worker_calls), 1)
    self.assertEqual(sum(message['action'] == 'REQUEST_RECEIVED'
                         for message in connection.sent), 1)


class FailureCleanupTests(unittest.TestCase):
  def test_exception_and_refused_reconnect_cleanup_before_outer_logger_shutdown(self):
    for refusal in (False, True):
      with self.subTest(refused_reconnect=refusal):
        harness = MainHarness(self, deferred=True)
        command = assignment('live/clip.mp4', 'cleanup-assignment')
        if refusal:
          first = FakeConnection([connected(), command, ConnectionClosed()])
          connections = (first, ConnectionRefusedError('controller unavailable'))
        else:
          first = FakeConnection([connected(), command,
                                  ValueError('synthetic protocol failure')])
          connections = (first,)
        tree = ast.parse((ROOT / 'snva.py').read_text(encoding='utf-8'))
        entrypoint = next(node for node in tree.body if isinstance(node, ast.If)
                          and isinstance(node.test, ast.Compare)
                          and isinstance(node.test.left, ast.Name)
                          and node.test.left.id == '__name__')
        outer_finally = next(node for node in reversed(entrypoint.body)
                             if isinstance(node, ast.Try) and node.finalbody)
        loop = asyncio.new_event_loop()
        self.addCleanup(loop.close)

        def join_main_logger():
          harness.events.append(('outer-logger-join', None))
          harness.module.main_logger_fn(harness.module.log_queue)

        namespace = dict(harness.module.__dict__)
        namespace.update({
          'main': lambda: harness.run(*connections),
          'asyncio': SimpleNamespace(get_event_loop=lambda: loop),
          'logger_thread': SimpleNamespace(join=join_main_logger)})
        with self.assertLogs(level=logging.DEBUG) as logs, \
             mock.patch.object(logging, 'shutdown',
                               side_effect=lambda: harness.events.append(('logging-shutdown', None))):
          code = compile(ast.Module(body=[outer_finally], type_ignores=[]),
                         str(ROOT / 'snva.py'), 'exec')
          if refusal:
            exec(code, namespace)
          else:
            with self.assertRaisesRegex(ValueError, 'synthetic protocol failure'):
              exec(code, namespace)
        self.assertEqual(len(harness.processes), 1)
        child = harness.processes[0]
        self.assertTrue(child.terminated)
        self.assertTrue(child.joined)
        self.assertFalse(child.is_alive())
        self.assertTrue(child.args[7].closed)
        self.assertTrue(child.args[8].closed)
        self.assertTrue(harness.threads[0].joined)
        self.assertFalse(harness.threads[0].is_alive())
        events = [event for event, _ in harness.events]
        self.assertLess(events.index('worker-terminate'), events.index('main-exit'))
        self.assertLess(events.index('forwarder-join'), events.index('main-exit'))
        self.assertLess(events.index('main-exit'), events.index('outer-logger-join'))
        self.assertLess(events.index('outer-logger-join'), events.index('logging-shutdown'))
        self.assertEqual(sum('late worker log during termination' in message
                             for message in logs.output), 1)
        self.assertTrue(harness.module.log_queue.closed)
        self.assertTrue(harness.module.log_queue.feeder_joined)
        self.assertFalse(first.messages)


class TrackingHandler(logging.Handler):
  def __init__(self):
    super().__init__()
    self.close_calls = 0
    self.records = []

  def emit(self, record):
    self.records.append(record)

  def close(self):
    self.close_calls += 1
    super().close()


class WorkerLoggingIntegrationTests(unittest.TestCase):
  def setUp(self):
    self.root = logging.getLogger()
    self.worker = logging.getLogger('utils.processor')
    self.old_disable = logging.root.manager.disable
    self.saved = [(logger, logger.handlers[:], logger.level, logger.propagate,
                   logger.disabled, logger.filters[:])
                  for logger in (self.root, self.worker)]
    for logger, _, _, _, _, _ in self.saved:
      logger.handlers[:] = []
      logger.filters[:] = []
      logger.disabled = False
    logging.disable(logging.NOTSET)
    self.addCleanup(self.restore_logging)

  def restore_logging(self):
    for logger, handlers, level, propagate, disabled, filters in self.saved:
      for handler in logger.handlers[:]:
        handler.close()
      logger.handlers[:] = handlers
      logger.filters[:] = filters
      logger.setLevel(level)
      logger.propagate = propagate
      logger.disabled = disabled
    logging.disable(self.old_disable)

  def configure_logger(self):
    source_path = ROOT / 'utils' / 'processor.py'
    tree = ast.parse(source_path.read_text(encoding='utf-8'))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == 'configure_logger')
    namespace = {'logging': logging, 'QueueHandler': QueueHandler,
                 '__name__': 'utils.processor'}
    exec(compile(ast.Module(body=[function], type_ignores=[]),
                 str(source_path), 'exec'), namespace)
    return namespace['configure_logger']

  def test_actual_worker_config_closes_inherited_handlers_and_queues_once(self):
    inherited_root = TrackingHandler()
    inherited_named = TrackingHandler()
    self.root.addHandler(inherited_root)
    self.worker.addHandler(inherited_named)
    self.worker.propagate = False
    self.worker.setLevel(logging.CRITICAL)
    child_queue = FakeQueue()
    configure = self.configure_logger()

    configure(logging.INFO, child_queue)

    self.assertEqual(inherited_root.close_calls, 1)
    self.assertEqual(inherited_named.close_calls, 1)
    self.assertEqual(len(self.root.handlers), 1)
    self.assertIsInstance(self.root.handlers[0], QueueHandler)
    self.assertIs(self.root.handlers[0].queue, child_queue)
    self.assertEqual(self.root.level, logging.INFO)
    self.assertEqual(self.root.handlers[0].level, logging.INFO)
    self.assertEqual(self.worker.handlers, [])
    self.assertTrue(self.worker.propagate)
    self.assertEqual(self.worker.level, logging.NOTSET)

    logging.info('root worker record')
    self.worker.warning('named worker record')
    logging.debug('filtered root record')
    self.worker.debug('filtered named record')
    records = [child_queue.get_nowait(), child_queue.get_nowait()]
    self.assertEqual([record.getMessage() for record in records],
                     ['root worker record', 'named worker record'])
    self.assertEqual([record.name for record in records], ['root', 'utils.processor'])
    with self.assertRaises(Empty):
      child_queue.get_nowait()
    self.assertFalse(inherited_root.records)
    self.assertFalse(inherited_named.records)

    old_queue_handler = self.root.handlers[0]
    replacement_queue = FakeQueue()
    with mock.patch.object(old_queue_handler, 'close',
                           wraps=old_queue_handler.close) as close:
      configure(logging.DEBUG, replacement_queue)
      close.assert_called_once_with()
    self.assertEqual(len(self.root.handlers), 1)
    logging.debug('replacement queue only')
    self.assertEqual(replacement_queue.get_nowait().getMessage(),
                     'replacement queue only')
    with self.assertRaises(Empty):
      child_queue.get_nowait()
    with self.assertRaises(Empty):
      replacement_queue.get_nowait()


if __name__ == '__main__':
  unittest.main()
