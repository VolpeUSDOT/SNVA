import ast
import inspect
import io
import logging
from logging.handlers import QueueHandler, RotatingFileHandler
import multiprocessing
import os
import pickle
from queue import Queue
import runpy
import sys
import tempfile
import unittest
from unittest import mock

from utils import logger as logging_helper


_sentinel_calls = []


def _payload_sentinel():
  _sentinel_calls.append(True)


class _ExecutablePayload:
  def __reduce__(self):
    return _payload_sentinel, ()


def _queue_only_worker(log_queue):
  root_logger = logging.getLogger()
  for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
    handler.close()
  root_logger.setLevel(logging.DEBUG)
  root_logger.addHandler(QueueHandler(log_queue))
  logging.info('worker queue only')
  logging.shutdown()


class LoggingTests(unittest.TestCase):
  def setUp(self):
    self.root = logging.getLogger()
    self.old_handlers = self.root.handlers[:]
    self.old_filters = self.root.filters[:]
    self.old_level = self.root.level
    self.old_disabled = self.root.disabled
    self.old_disable_level = logging.root.manager.disable
    for handler in self.old_handlers:
      self.root.removeHandler(handler)
    self.root.filters.clear()
    self.root.disabled = False
    logging.disable(logging.NOTSET)
    self.temp_dir = tempfile.TemporaryDirectory()
    self.log_path = os.path.join(self.temp_dir.name, 'snva.log')

  def tearDown(self):
    for handler in self.root.handlers[:]:
      self.root.removeHandler(handler)
      handler.flush()
      handler.close()
    self.root.filters[:] = self.old_filters
    self.root.setLevel(self.old_level)
    self.root.disabled = self.old_disabled
    logging.disable(self.old_disable_level)
    for handler in self.old_handlers:
      self.root.addHandler(handler)
    self.temp_dir.cleanup()

  def configure(self, level='info', mode='silent', max_bytes=2**23):
    return logging_helper.configure_logging(
      self.log_path, '%(levelname)s:%(name)s:%(message)s',
      level, mode, max_bytes)

  def read_log(self, suffix=''):
    for handler in self.root.handlers:
      handler.flush()
    with open(self.log_path + suffix, encoding='utf-8') as log_file:
      return log_file.read()

  def test_main_and_forwarded_queue_records_are_written_once(self):
    handlers = self.configure()
    self.assertEqual(handlers, self.root.handlers)
    self.assertEqual(len(handlers), 1)
    self.assertIsInstance(handlers[0], RotatingFileHandler)
    logging.info('main record')

    child_queue = Queue()
    child_logger = logging.Logger('test_worker', logging.DEBUG)
    child_logger.addHandler(QueueHandler(child_queue))
    child_logger.info('child %s', 'record')
    forwarded_record = child_queue.get_nowait()
    self.root.handle(forwarded_record)

    self.assertEqual(self.read_log(),
                     'INFO:root:main record\nINFO:test_worker:child record\n')

  def test_severity_filters_main_and_forwarded_records(self):
    levels = ((logging.DEBUG, 'debug'), (logging.INFO, 'info'),
              (logging.WARNING, 'warning'), (logging.ERROR, 'error'),
              (logging.CRITICAL, 'critical'))
    for configured in ('debug', 'info', 'error', logging.WARNING):
      with self.subTest(level=configured):
        console = io.StringIO()
        with mock.patch('sys.stderr', console):
          handlers = self.configure(configured, 'verbose')
        threshold = self.root.level
        self.assertTrue(all(handler.level == threshold for handler in handlers))
        child_queue = Queue()
        child_logger = logging.Logger('test_worker', logging.DEBUG)
        child_logger.addHandler(QueueHandler(child_queue))
        for level, name in levels:
          logging.log(level, 'main %s', name)
          child_logger.log(level, 'child %s', name)
          self.root.handle(child_queue.get_nowait())
        expected = ''.join(
          '{}:root:main {}\n{}:test_worker:child {}\n'.format(
            name.upper(), name, name.upper(), name)
          for level, name in levels if level >= threshold)
        self.assertEqual(console.getvalue(), expected)
        self.assertTrue(self.read_log().endswith(expected))
        for handler in handlers:
          self.root.removeHandler(handler)
          handler.close()
        os.remove(self.log_path)

  def test_silent_mode_has_only_file_output(self):
    console = io.StringIO()
    with mock.patch('sys.stderr', console):
      handlers = self.configure(mode='silent')
      logging.warning('file only')
    self.assertEqual(len(handlers), 1)
    self.assertEqual(console.getvalue(), '')
    self.assertEqual(self.read_log(), 'WARNING:root:file only\n')

  def test_verbose_mode_uses_same_format_for_file_and_console(self):
    console = io.StringIO()
    with mock.patch('sys.stderr', console):
      handlers = self.configure(mode='verbose')
      logging.info('both outputs')
    self.assertEqual(len(handlers), 2)
    self.assertIs(type(handlers[1]), logging.StreamHandler)
    self.assertEqual(console.getvalue(), self.read_log())

  def test_invalid_configuration_is_rejected_before_file_open(self):
    cases = ((logging.INFO, 'invalid', 100, ValueError),
             ('invalid', 'silent', 100, ValueError),
             (None, 'silent', 100, TypeError),
             (True, 'silent', 100, TypeError),
             (-1, 'silent', 100, ValueError),
             (logging.INFO, 'silent', -1, ValueError),
             (logging.INFO, 'silent', 1.5, TypeError),
             (logging.INFO, 'silent', True, TypeError))
    existing = logging.StreamHandler(io.StringIO())
    self.root.addHandler(existing)
    level_before = self.root.level
    for level, mode, max_bytes, error in cases:
      with self.subTest(level=level, mode=mode, max_bytes=max_bytes):
        with mock.patch.object(logging_helper, '_RotatingFileHandler') as factory:
          with self.assertRaises(error):
            self.configure(level, mode, max_bytes)
          factory.assert_not_called()
        self.assertEqual(self.root.handlers, [existing])
        self.assertEqual(self.root.level, level_before)
        self.assertFalse(os.path.exists(self.log_path))

  def test_invalid_mode_is_checked_before_other_configuration(self):
    with mock.patch.object(logging_helper, '_RotatingFileHandler') as factory:
      with self.assertRaisesRegex(ValueError, 'logmode'):
        logging_helper.configure_logging(
          None, None, None, 'invalid', None)
      factory.assert_not_called()

  def test_level_names_integer_and_zero_rotation_limit(self):
    for level, expected in (('DEBUG', logging.DEBUG),
                            ('Info', logging.INFO),
                            ('warn', logging.WARNING),
                            (logging.ERROR, logging.ERROR),
                            (25, 25), ('NOTSET', logging.NOTSET)):
      with self.subTest(level=level):
        handlers = self.configure(level, max_bytes=0)
        self.assertEqual(self.root.level, expected)
        self.assertEqual(handlers[0].level, expected)
        self.assertEqual(handlers[0].maxBytes, 0)

  def test_main_and_queue_exceptions_keep_traceback_once(self):
    self.configure()
    child_queue = Queue()
    child_logger = logging.Logger('test_worker', logging.DEBUG)
    child_logger.addHandler(QueueHandler(child_queue))
    try:
      raise ValueError('main failure')
    except ValueError:
      logging.exception('main exception')
    try:
      raise RuntimeError('child failure')
    except RuntimeError:
      child_logger.exception('child exception')
    forwarded_record = child_queue.get_nowait()
    self.assertIsNone(forwarded_record.exc_info)
    self.root.handle(forwarded_record)
    contents = self.read_log()
    self.assertEqual(contents.count('Traceback (most recent call last):'), 2)
    self.assertEqual(contents.count('ValueError: main failure'), 1)
    self.assertEqual(contents.count('RuntimeError: child failure'), 1)
    self.assertIn('ERROR:root:main exception', contents)
    self.assertIn('ERROR:test_worker:child exception', contents)

  def test_utf8_format_and_rotation_settings_are_preserved(self):
    log_format = '%(processName)s:%(process)d:%(levelname)s:%(message)s'
    handlers = logging_helper.configure_logging(
      self.log_path, log_format, logging.INFO, 'silent', 128)
    handler = handlers[0]
    self.assertEqual(handler.backupCount, 2**23)
    self.assertEqual(handler.maxBytes, 128)
    self.assertEqual(handler.encoding, 'utf-8')
    self.assertEqual(handler.formatter._fmt, log_format)
    logging.info('caf\u00e9')
    expected = '{}:{}:INFO:caf\u00e9\n'.format(
      multiprocessing.current_process().name, os.getpid())
    self.assertEqual(self.read_log(), expected)

  def test_rotation_writes_current_and_backup_files(self):
    handler = self.configure(max_bytes=60)[0]
    self.assertEqual(handler.backupCount, 2**23)
    # Bound filesystem work in the test instead of scanning millions of backups.
    handler.backupCount = 2
    for number in range(3):
      logging.info('record %s %s', number, 'x' * 20)
    self.assertIn('record 2', self.read_log())
    self.assertIn('record 1', self.read_log('.1'))
    self.assertIn('record 0', self.read_log('.2'))

  def test_reconfigure_closes_previous_handlers_without_duplicates(self):
    old_handler = self.configure()[0]
    logging.info('before reconfiguration')
    handlers = self.configure('debug', 'silent')
    self.assertIsNone(old_handler.stream)
    self.assertEqual(self.root.handlers, handlers)
    self.assertEqual(len(handlers), 1)
    logging.debug('after reconfiguration')
    self.assertEqual(self.read_log(),
                     'INFO:root:before reconfiguration\n'
                     'DEBUG:root:after reconfiguration\n')

  def test_handler_setup_failure_closes_file_and_keeps_previous_configuration(self):
    previous = self.configure()[0]
    failed_handlers = []

    def open_handler(*args, **kwargs):
      handler = RotatingFileHandler(*args, **kwargs)
      failed_handlers.append(handler)
      return handler

    with mock.patch.object(logging_helper, '_RotatingFileHandler', open_handler):
      with mock.patch.object(logging_helper, '_StreamHandler',
                             side_effect=RuntimeError('console failure')):
        with self.assertRaisesRegex(RuntimeError, 'console failure'):
          self.configure(mode='verbose')
    self.assertIsNone(failed_handlers[0].stream)
    self.assertEqual(self.root.handlers, [previous])
    logging.info('previous still works')
    self.assertEqual(self.read_log(), 'INFO:root:previous still works\n')

  def test_shutdown_flushes_and_closes_file(self):
    handler = self.configure()[0]
    logging.info('last record')
    with mock.patch.object(handler, 'flush', wraps=handler.flush) as flush:
      logging.shutdown([lambda: handler])
      flush.assert_called()
    self.assertIsNone(handler.stream)
    self.root.removeHandler(handler)
    self.assertEqual(self.read_log(), 'INFO:root:last record\n')
    os.remove(self.log_path)
    self.assertFalse(os.path.exists(self.log_path))

  def test_spawn_worker_reset_has_no_direct_file_writes(self):
    self.check_worker_reset('spawn')

  @unittest.skipUnless('fork' in multiprocessing.get_all_start_methods(),
                       'fork is not supported on this platform')
  def test_fork_worker_reset_closes_inherited_file_handler(self):
    self.check_worker_reset('fork')

  def check_worker_reset(self, start_method):
    self.configure()
    context = multiprocessing.get_context(start_method)
    child_queue = context.Queue()
    worker = context.Process(target=_queue_only_worker, args=(child_queue,))
    worker.start()
    try:
      record = child_queue.get(timeout=15)
      worker.join(timeout=15)
      self.assertFalse(worker.is_alive())
      self.assertEqual(worker.exitcode, 0)
      self.assertEqual(self.read_log(), '')
      self.root.handle(record)
      logging.info('parent still writes')
      self.assertEqual(self.read_log(),
                       'INFO:root:worker queue only\n'
                       'INFO:root:parent still writes\n')
    finally:
      if worker.is_alive():
        worker.terminate()
      worker.join(timeout=15)
      child_queue.close()
      child_queue.join_thread()

  def test_no_receiver_deserializer_or_executable_service_remains(self):
    self.assertEqual(logging_helper.__all__, ['configure_logging'])
    public_helpers = [name for name, value in vars(logging_helper).items()
                      if not name.startswith('_') and callable(value)]
    self.assertEqual(public_helpers, ['configure_logging'])
    for removed in ('LogRecordStreamHandler', 'LogRecordSocketReceiver',
                    'unPickle', 'serve_until_stopped'):
      self.assertFalse(hasattr(logging_helper, removed))
    tree = ast.parse(inspect.getsource(logging_helper))
    imported = set()
    for node in ast.walk(tree):
      if isinstance(node, ast.Import):
        imported.update(alias.name.split('.')[0] for alias in node.names)
      elif isinstance(node, ast.ImportFrom):
        imported.add(node.module.split('.')[0])
    self.assertFalse(imported.intersection(
      {'pickle', 'socket', 'socketserver', 'struct', 'subprocess'}))
    with mock.patch('logging.handlers.RotatingFileHandler') as file_handler:
      with mock.patch.object(sys, 'argv', ['logger.py', self.log_path]):
        runpy.run_path(logging_helper.__file__, run_name='__main__')
      file_handler.assert_not_called()
    self.assertFalse(os.path.exists(self.log_path))

  def test_serialized_payload_cannot_invoke_sentinel_through_public_helper(self):
    _sentinel_calls.clear()
    payload = pickle.dumps(_ExecutablePayload(), protocol=2)
    self.assertIn(b'_payload_sentinel', payload)
    self.assertIn(b'\x00', payload)
    defaults = [self.log_path, '%(message)s', logging.INFO, 'silent', 100]
    with mock.patch('pickle.loads', side_effect=AssertionError('deserialized')):
      with mock.patch('pickle.load', side_effect=AssertionError('deserialized')):
        for position in range(len(defaults)):
          with self.subTest(argument=position):
            arguments = defaults[:]
            arguments[position] = payload
            with self.assertRaises((TypeError, ValueError, OSError)):
              logging_helper.configure_logging(*arguments)
            self.assertEqual(_sentinel_calls, [])
        self.configure()
        logging.info('inert payload: %r', payload)
    self.assertEqual(_sentinel_calls, [])
    self.assertIn('inert payload:', self.read_log())


if __name__ == '__main__':
  unittest.main()
