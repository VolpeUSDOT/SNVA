import logging
from logging import StreamHandler as _StreamHandler
from logging.handlers import RotatingFileHandler as _RotatingFileHandler

__all__ = ['configure_logging']


def configure_logging(log_file_path, log_format, log_level, log_mode,
                      log_file_max_bytes):
  """Configure main-process output; workers must replace root handlers with
  QueueHandler before logging, including closing handlers inherited by fork.

  Drain forwarded worker records before calling logging.shutdown() to flush
  and close the returned handlers. Call this once at startup, before threads.
  """
  if log_mode not in ('verbose', 'silent'):
    raise ValueError('The specified logmode is not in the set '
                     "['verbose', 'silent'].")

  if isinstance(log_level, str):
    log_level = logging.getLevelName(log_level.upper())
    if not isinstance(log_level, int):
      raise ValueError('Unknown logging level: {}'.format(log_level))
  if isinstance(log_level, bool) or not isinstance(log_level, int):
    raise TypeError('log_level must be a logging level name or integer')
  if log_level < 0:
    raise ValueError('log_level must not be negative')
  if (isinstance(log_file_max_bytes, bool)
      or not isinstance(log_file_max_bytes, int)):
    raise TypeError('log_file_max_bytes must be an integer')
  if log_file_max_bytes < 0:
    raise ValueError('log_file_max_bytes must not be negative')

  formatter = logging.Formatter(log_format)
  handlers = []
  try:
    handlers.append(_RotatingFileHandler(
      filename=log_file_path, maxBytes=log_file_max_bytes, backupCount=2**23,
      encoding='utf-8'))
    if log_mode == 'verbose':
      handlers.append(_StreamHandler())
    for handler in handlers:
      handler.setFormatter(formatter)
      # Logger.handle() bypasses logger-level filtering for forwarded records.
      handler.setLevel(log_level)
  except Exception:
    for handler in handlers:
      handler.close()
    raise

  root_logger = logging.getLogger()
  for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
    handler.close()
  root_logger.setLevel(log_level)
  for handler in handlers:
    root_logger.addHandler(handler)
  return handlers
