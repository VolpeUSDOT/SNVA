import logging
import logging.handlers
from logging.handlers import RotatingFileHandler
import pickle
import socketserver
import struct
import sys

class LogRecordStreamHandler(socketserver.StreamRequestHandler):
  """Handler for a streaming logging request.

  This basically logs the record using whatever logging policy is
  configured locally.
  """

  def handle(self):
    """
    Handle multiple requests - each expected to be a 4-byte length,
    followed by the LogRecord in pickle format. Logs the record
    according to whatever policy is configured locally.
    """
    while True:
      chunk = self.connection.recv(4)
      if len(chunk) < 4:
        break
      slen = struct.unpack('>L', chunk)[0]
      chunk = self.connection.recv(slen)
      while len(chunk) < slen:
        chunk = chunk + self.connection.recv(slen - len(chunk))
      obj = self.unPickle(chunk)
      record = logging.makeLogRecord(obj)
      self.handleLogRecord(record)

  def unPickle(self, data):
    return pickle.loads(data)

  def handleLogRecord(self, record):
    # if a name is specified, we use the named logger rather than the one
    # implied by the record.
    if self.server.logname is not None:
      name = self.server.logname
    else:
      name = record.name
    logger = logging.getLogger(name)
    # N.B. EVERY record gets logged. This is because Logger.handle
    # is normally called AFTER logger-level filtering. If you want
    # to do filtering, do it at the client end to save wasting
    # cycles and network bandwidth!
    logger.handle(record)


class LogRecordSocketReceiver(socketserver.ThreadingTCPServer):
  """
  Simple TCP socket-based logging receiver suitable for testing.
  """

  allow_reuse_address = True

  def __init__(self, host='localhost',
               port=logging.handlers.DEFAULT_TCP_LOGGING_PORT,
               handler=LogRecordStreamHandler):
    socketserver.ThreadingTCPServer.__init__(self, (host, port), handler)
    self.abort = 0
    self.timeout = 1
    self.logname = None

  def serve_until_stopped(self):
    import select
    abort = 0
    while not abort:
      rd, wr, ex = select.select([self.socket.fileno()], [], [], self.timeout)
      if rd:
        self.handle_request()
      abort = self.abort


if __name__ == '__main__':
  log_file_path = sys.argv[1]
  log_format = sys.argv[2]

  # Define our log level based on arguments
  if sys.argv[3] == 'error':
    log_level = logging.ERROR
  elif sys.argv[3] == 'debug':
    log_level = logging.DEBUG
  else:
    log_level = logging.INFO

  log_mode = sys.argv[4]

  log_file_max_bytes = int(sys.argv[5])

  log_handlers = [RotatingFileHandler(
    filename=log_file_path, maxBytes=log_file_max_bytes, backupCount=2**23,
    encoding='utf-8')]  # largest backupCount before RotatingFileHandler breaks

  logging.basicConfig(level=log_level, format=log_format, handlers=log_handlers)

  log_server = LogRecordSocketReceiver()

  print('\0', flush=True)  # indicate readiness to snva.py

  log_server.serve_until_stopped()