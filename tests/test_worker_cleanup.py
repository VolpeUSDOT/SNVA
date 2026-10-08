"""Exercise forced-abort cleanup with a real abandoned multiprocessing lock."""

import ast
import logging
from multiprocessing import get_context
from pathlib import Path
from queue import Empty
import signal
import threading
import time
import unittest


def hold_queue_lock(queue, ready):
  queue._wlock.acquire()
  ready.set()
  time.sleep(60)


class WorkerCleanupTests(unittest.TestCase):
  def test_forced_abort_does_not_join_abandoned_queue_feeder(self):
    tree = ast.parse((Path(__file__).resolve().parents[1] / 'snva.py').read_text())
    main = next(node for node in tree.body
                if isinstance(node, ast.AsyncFunctionDef) and node.name == 'main')
    cleanup = next(node for node in main.body
                   if isinstance(node, ast.FunctionDef) and node.name == 'close_worker')
    forwarding = next(node for node in tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == 'child_logger_fn')
    context = get_context('spawn')
    queue = context.Queue()
    results = context.Queue()
    parent_logs = context.Queue()
    ready = context.Event()
    stop = threading.Event()
    child = context.Process(target=hold_queue_lock, args=(queue, ready))
    namespace = {'logging': logging, 'Empty': Empty, 'signal': signal,
                 'os': __import__('os'), 'child_process_map': {'attempt': child},
                 'child_log_queue_map': {'attempt': queue},
                 'return_code_queue_map': {'attempt': results},
                 'child_log_stop_map': {'attempt': stop},
                 'assignment_map': {'attempt': {}}, 'completed_result_map': {}}
    exec(compile(ast.Module(body=[cleanup, forwarding], type_ignores=[]),
                 'snva-cleanup', 'exec'), namespace)
    forwarder = threading.Thread(target=namespace['child_logger_fn'],
                                 args=(parent_logs, queue, stop), daemon=True)
    namespace['child_logger_thread_map'] = {'attempt': forwarder}
    child.start()
    forwarder.start()
    try:
      self.assertTrue(ready.wait(5), 'spawned producer did not acquire queue lock')
      started = time.monotonic()
      namespace['close_worker']('attempt', abort=True)
      self.assertLess(time.monotonic() - started, 4)
      self.assertFalse(child.is_alive())
      self.assertTrue(stop.is_set())
      self.assertFalse(namespace['assignment_map'])
    finally:
      stop.set()
      if child.is_alive():
        child.terminate()
      child.join(5)
      # Release the intentionally abandoned lock so test helper threads exit.
      try:
        queue._wlock.release()
      except ValueError:
        pass
      forwarder.join(2)
      for item in (queue, results, parent_logs):
        item.cancel_join_thread()
        item.close()
      child.close()


if __name__ == '__main__':
  unittest.main()
