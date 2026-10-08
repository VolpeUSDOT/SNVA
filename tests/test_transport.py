import contextlib
from concurrent import futures
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import socket
import ssl
import stat
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock
from urllib.parse import parse_qs, urlsplit

from utils import transport

try:
  import grpc
except ImportError:
  grpc = None


ROOT = Path(__file__).resolve().parents[1]


class ControlUrlTests(unittest.TestCase):
  def test_hosts_use_only_wss_and_the_registration_endpoint(self):
    for host, authority in (('localhost:8080', 'localhost:8080'),
                            ('controller.example', 'controller.example'),
                            ('wss://controller.example:443',
                             'controller.example:443'),
                            ('[::1]:8080', '[::1]:8080'),
                            ('::1', '[::1]'),
                            ('wss://[2001:db8::1]:443', '[2001:db8::1]:443')):
      with self.subTest(host=host):
        self.assertEqual(transport.build_control_url(host),
                         'wss://{}/registerProcess'.format(authority))

  def test_plaintext_urls_and_non_authorities_are_rejected(self):
    invalid = (None, '', 'ws://localhost:8080', 'http://localhost',
               'https://localhost', 'grpc://localhost', '//localhost:8080',
               'wss://user:secret@localhost', 'user@localhost:8080',
               'localhost:8080/path', 'wss://localhost/',
               'wss://localhost/registerProcess', 'localhost?x=1',
               'wss://localhost?x=1', 'wss://localhost?', 'localhost#x',
               'wss://localhost#', 'wss://', 'localhost:0',
               'localhost:65536', 'localhost:bad', 'localhost:',
               '[::1]suffix', '[not-an-ip]:443', 'localhost\\path',
               ' localhost', 'localhost\n', 'local\x00host')
    for host in invalid:
      with self.subTest(host=host):
        with self.assertRaises(ValueError):
          transport.build_control_url(host)

  def test_reconnect_identity_and_token_are_urlencoded(self):
    identity = 'process /?&=#+\u2603'
    token = 'secret+/=&?#'
    url = transport.build_control_url('localhost:8080', identity, token)
    parsed = urlsplit(url)
    self.assertEqual(parsed.path, '/registerProcess')
    self.assertEqual(parsed.fragment, '')
    self.assertEqual(parse_qs(parsed.query),
                     {'id': [identity], 'reconnectToken': [token]})
    self.assertIn('reconnectToken=secret%2B%2F%3D%26%3F%23', url)


class TlsUnitTests(unittest.TestCase):
  def test_missing_unreadable_empty_and_malformed_tls_files_fail_closed(self):
    with tempfile.TemporaryDirectory() as directory:
      ca, cert, key = [Path(directory) / name for name in ('ca', 'cert', 'key')]
      for path in (ca, cert, key):
        path.write_bytes(b'invalid PEM')
      helpers = (transport.create_client_context,
                 lambda *paths: transport.create_model_channel(
                   'localhost:8500', *paths))
      for helper in helpers:
        for index in range(3):
          for missing in (None, '', Path(directory) / 'missing'):
            paths = [ca, cert, key]
            paths[index] = missing
            with self.subTest(helper=helper, index=index, missing=missing):
              with self.assertRaises((ValueError, OSError)):
                helper(*paths)
          original = [ca, cert, key][index]
          original.write_bytes(b' \n')
          with self.assertRaises(ValueError):
            helper(ca, cert, key)
          original.write_bytes(b'invalid PEM')
        with self.assertRaises(ssl.SSLError):
          helper(ca, cert, key)

  def test_secure_grpc_credentials_receive_all_pem_material_and_options(self):
    grpc_mock = mock.Mock()
    context = mock.Mock()
    material = (b'root certificates', b'client certificate', b'private key')
    options = [('grpc.max_receive_message_length', 100 * 1024 * 1024)]
    with mock.patch.object(transport, '_load_tls_material',
                           return_value=(context, material)):
      with mock.patch.dict(sys.modules, {'grpc': grpc_mock}):
        channel = transport.create_model_channel(
          'model.example:8500', 'ca.pem', 'cert.pem', 'key.pem', options)
    grpc_mock.ssl_channel_credentials.assert_called_once_with(
      root_certificates=material[0], private_key=material[2],
      certificate_chain=material[1])
    grpc_mock.secure_channel.assert_called_once_with(
      'model.example:8500', grpc_mock.ssl_channel_credentials.return_value,
      options=tuple(options))
    self.assertIs(channel, grpc_mock.secure_channel.return_value)
    grpc_mock.insecure_channel.assert_not_called()

  def test_grpc_hostname_overrides_are_rejected(self):
    for name in ('grpc.ssl_target_name_override', 'grpc.default_authority'):
      with self.subTest(option=name):
        with self.assertRaisesRegex(ValueError, 'hostname overrides'):
          transport.create_model_channel(
            'localhost:8500', None, None, None, [(name, 'trusted.example')])

  def test_python36_context_disables_protocols_older_than_tls12(self):
    class LegacyContext:
      __slots__ = ('check_hostname', 'verify_mode', 'options')

      def __init__(self, protocol):
        self.options = 0

      def load_verify_locations(self, **kwargs):
        pass

      def load_cert_chain(self, **kwargs):
        pass

    with mock.patch.object(transport, '_read_tls_files',
                           return_value=(b'ca', b'cert', b'key')):
      with mock.patch.object(transport.ssl, 'SSLContext', LegacyContext):
        context = transport.create_client_context('ca', 'cert', 'key')
    self.assertTrue(context.check_hostname)
    self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
    for option in (ssl.OP_NO_SSLv3, ssl.OP_NO_TLSv1, ssl.OP_NO_TLSv1_1):
      self.assertEqual(context.options & option, option)


def load_analyzer(filename):
  """Load only the real analyzer source; no TensorFlow/NumPy installation."""
  dependencies = {name: types.ModuleType(name) for name in (
    'numpy', 'skimage', 'skimage.transform', 'tensorflow',
    'tensorflow_serving', 'tensorflow_serving.apis')}
  dependencies['numpy'].ndarray = mock.Mock()
  dependencies['numpy'].uint8 = 'uint8'
  dependencies['numpy'].float32 = 'float32'
  dependencies['skimage'].img_as_float32 = mock.Mock()
  dependencies['skimage.transform'].resize = mock.Mock()
  stub = mock.Mock()
  dependencies['tensorflow_serving.apis'].predict_pb2 = mock.Mock()
  dependencies['tensorflow_serving.apis'].prediction_service_pb2_grpc = \
    types.SimpleNamespace(PredictionServiceStub=stub)
  spec = importlib.util.spec_from_file_location(
    '_transport_test_' + filename, str(ROOT / 'utils' / (filename + '.py')))
  module = importlib.util.module_from_spec(spec)
  with mock.patch.dict(sys.modules, dependencies):
    spec.loader.exec_module(module)
  return module, stub


def analyzer_arguments(filename, host):
  arguments = dict(frame_shape=[1, 1, 3], num_frames=1, num_classes=1,
                   batch_size=1, model_name='model',
                   model_signature_name='serving_default',
                   model_server_host=host, model_input_size=1,
                   should_extract_timestamps=False, timestamp_x=0,
                   timestamp_y=0, timestamp_height=0, timestamp_max_width=0,
                   should_crop=False, crop_x=0, crop_y=0, crop_width=0,
                   crop_height=0, ffmpeg_command=['never-run-ffmpeg'],
                   max_num_threads=1)
  if filename == 'analyzer':
    arguments['processor_mode'] = 'weather'
  return arguments


class AnalyzerTests(unittest.TestCase):
  def test_analyzers_use_shared_helper_before_native_subprocess(self):
    for filename, classname in (('analyzer', 'VideoAnalyzer'),
                                ('signalstateanalyzer', 'SignalVideoAnalyzer')):
      module, stub = load_analyzer(filename)
      analyzer_class = getattr(module, classname)
      for parameter in ('tls_ca', 'tls_cert', 'tls_key'):
        self.assertIsNone(inspect.signature(analyzer_class).parameters[
          parameter].default)
      arguments = analyzer_arguments(filename, 'model.example:8500')
      arguments.update(tls_ca='ca', tls_cert='cert', tls_key='key')
      events = []
      channel = object()
      frame_pipe = types.SimpleNamespace(pid=123, returncode=0)
      with mock.patch.object(module, 'create_model_channel',
                             side_effect=lambda *args, **kwargs:
                             events.append('tls') or channel) as factory:
        with mock.patch.object(module, 'Popen',
                               side_effect=lambda *args, **kwargs:
                               events.append('native') or frame_pipe):
          analyzer_class(**arguments)
      expected_kwargs = {}
      if filename == 'signalstateanalyzer':
        expected_kwargs['options'] = [
          ('grpc.max_message_length', 100 * 1024 * 1024),
          ('grpc.max_receive_message_length', 100 * 1024 * 1024)]
      factory.assert_called_once_with(
        'model.example:8500', 'ca', 'cert', 'key', **expected_kwargs)
      stub.assert_called_once_with(channel)
      self.assertEqual(events, ['tls', 'native'])

  def test_direct_callers_without_tls_fail_before_ffmpeg(self):
    for filename, classname in (('analyzer', 'VideoAnalyzer'),
                                ('signalstateanalyzer', 'SignalVideoAnalyzer')):
      module, _ = load_analyzer(filename)
      analyzer_class = getattr(module, classname)
      with self.subTest(analyzer=classname):
        with mock.patch.object(module, 'Popen') as popen:
          with self.assertRaisesRegex(ValueError, 'tls_ca'):
            analyzer_class(**analyzer_arguments(filename, 'localhost:8500'))
        popen.assert_not_called()


@unittest.skipUnless(shutil.which('openssl'), 'OpenSSL is required for live TLS')
class LiveTransportTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.temporary = tempfile.TemporaryDirectory()
    cls.directory = Path(cls.temporary.name)
    try:
      cls.generate_certificates()
    except BaseException:
      cls.temporary.cleanup()
      raise

  @classmethod
  def tearDownClass(cls):
    cls.temporary.cleanup()

  @classmethod
  def openssl(cls, *arguments):
    subprocess.run(['openssl'] + list(arguments), cwd=str(cls.directory),
                   check=True, stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE, timeout=30)

  @classmethod
  def generate_certificates(cls):
    for name in ('ca', 'untrusted-ca'):
      cls.openssl('req', '-x509', '-newkey', 'rsa:2048', '-nodes',
                  '-keyout', name + '.key', '-out', name + '.pem',
                  '-days', '2', '-subj', '/CN=' + name,
                  '-addext', 'basicConstraints=critical,CA:TRUE')
    (cls.directory / 'index').write_text('')
    (cls.directory / 'serial').write_text('01\n')
    (cls.directory / 'openssl.cnf').write_text(
      '[ca]\ndefault_ca=local\n[local]\n'
      'database=index\nserial=serial\nnew_certs_dir=.\n'
      'certificate=ca.pem\nprivate_key=ca.key\n'
      'default_md=sha256\ndefault_days=2\npolicy=policy\n'
      'unique_subject=no\n[policy]\ncommonName=supplied\n'
      '[server]\nbasicConstraints=critical,CA:FALSE\n'
      'keyUsage=critical,digitalSignature,keyEncipherment\n'
      'extendedKeyUsage=serverAuth\nsubjectAltName=DNS:localhost\n'
      '[client]\nbasicConstraints=critical,CA:FALSE\n'
      'keyUsage=critical,digitalSignature\n'
      'extendedKeyUsage=clientAuth\n')
    for name, extension in (('server', 'server'), ('expired', 'server'),
                            ('client', 'client')):
      cls.openssl('req', '-new', '-newkey', 'rsa:2048', '-nodes',
                  '-keyout', name + '.key', '-out', name + '.csr',
                  '-subj', '/CN=' + name)
      arguments = ['ca', '-batch', '-notext', '-config', 'openssl.cnf',
                   '-in', name + '.csr', '-out', name + '.pem',
                   '-extensions', extension]
      if name == 'expired':
        arguments += ['-startdate', '20000101000000Z',
                      '-enddate', '20000102000000Z']
      cls.openssl(*arguments)

  def paths(self, ca='ca', identity='client'):
    return (str(self.directory / (ca + '.pem')),
            str(self.directory / (identity + '.pem')),
            str(self.directory / (identity + '.key')))

  @contextlib.contextmanager
  def tls_server(self, identity='server'):
    ca, cert, key = self.paths(identity=identity)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(cert, key)
    context.load_verify_locations(cafile=ca)
    context.verify_mode = ssl.CERT_REQUIRED
    errors = []
    with socket.socket() as listener:
      listener.bind(('127.0.0.1', 0))
      listener.listen(1)
      listener.settimeout(5)

      def serve():
        try:
          connection, _ = listener.accept()
          with connection:
            connection.settimeout(5)
            with context.wrap_socket(connection, server_side=True) as secured:
              if secured.recv(4) == b'ping':
                secured.sendall(b'pong')
        except (ssl.SSLError, OSError) as error:
          errors.append(error)

      worker = threading.Thread(target=serve, daemon=True)
      worker.start()
      try:
        yield listener.getsockname()[1], errors
      finally:
        worker.join(timeout=6)
        self.assertFalse(worker.is_alive(), 'TLS server did not finish')

  def exchange(self, context, port, hostname='localhost'):
    with socket.create_connection(('127.0.0.1', port), timeout=5) as connection:
      with context.wrap_socket(connection, server_hostname=hostname) as secured:
        secured.sendall(b'ping')
        return secured.recv(4)

  def test_verified_mutual_tls_success_and_context_policy(self):
    context = transport.create_client_context(*self.paths())
    self.assertTrue(context.check_hostname)
    self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
    self.assertGreaterEqual(context.minimum_version, ssl.TLSVersion.TLSv1_2)
    with self.tls_server() as (port, errors):
      self.assertEqual(self.exchange(context, port), b'pong')
    self.assertEqual(errors, [])

  def test_unknown_ca_wrong_hostname_and_expired_server_fail(self):
    cases = (('untrusted-ca', 'server', 'localhost'),
             ('ca', 'server', 'wrong.example'),
             ('ca', 'expired', 'localhost'))
    for ca, identity, hostname in cases:
      with self.subTest(ca=ca, identity=identity, hostname=hostname):
        context = transport.create_client_context(*self.paths(ca=ca))
        with self.tls_server(identity) as (port, errors):
          with self.assertRaises(ssl.SSLCertVerificationError):
            self.exchange(context, port, hostname)
        self.assertTrue(errors)

  def test_server_refuses_missing_client_certificate(self):
    # Deliberately bypass the helper to prove the server requires client auth.
    context = ssl.create_default_context(cafile=self.paths()[0])
    with self.tls_server() as (port, errors):
      with self.assertRaises(ssl.SSLError):
        self.exchange(context, port)
    self.assertTrue(errors)

  def test_invalid_cert_key_and_mismatch_are_rejected_before_channel(self):
    with tempfile.TemporaryDirectory() as directory:
      invalid = Path(directory) / 'invalid.pem'
      invalid.write_bytes(b'not a PEM file')
      ca, cert, key = self.paths()
      for paths in ((ca, invalid, key), (ca, cert, invalid),
                    (ca, cert, self.paths(identity='server')[2])):
        with self.subTest(paths=paths):
          with self.assertRaises(ssl.SSLError):
            transport.create_client_context(*paths)
          if grpc is not None:
            with mock.patch.object(grpc, 'secure_channel') as secure:
              with self.assertRaises(ssl.SSLError):
                transport.create_model_channel('localhost:8500', *paths)
            secure.assert_not_called()

  def test_encrypted_private_key_fails_without_prompting_for_password(self):
    with tempfile.TemporaryDirectory() as directory:
      encrypted_key = Path(directory) / 'encrypted.key'
      self.openssl('pkey', '-in', self.paths()[2], '-aes256',
                   '-passout', 'pass:temporary-test-password',
                   '-out', str(encrypted_key))
      result = subprocess.run(
        [sys.executable, '-B', '-c',
         'from utils.transport import create_client_context; '
         'import sys; create_client_context(*sys.argv[1:])',
         self.paths()[0], self.paths()[1], str(encrypted_key)],
        cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=5)
      self.assertNotEqual(result.returncode, 0)
      self.assertNotIn(b'Enter PEM pass phrase', result.stderr)
      self.assertIn(b'SSLError', result.stderr)

  @contextlib.contextmanager
  def grpc_server(self, identity='server'):
    ca, cert, key = [Path(path).read_bytes()
                     for path in self.paths(identity=identity)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    handler = grpc.unary_unary_rpc_method_handler(
      lambda request, context: b'pong:' + request,
      request_deserializer=lambda value: value,
      response_serializer=lambda value: value)
    server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(
      'transport.Echo', {'Predict': handler}),))
    credentials = grpc.ssl_server_credentials(
      [(key, cert)], root_certificates=ca, require_client_auth=True)
    port = server.add_secure_port('127.0.0.1:0', credentials)
    self.assertNotEqual(port, 0)
    server.start()
    try:
      yield port
    finally:
      server.stop(0).wait(timeout=5)

  @unittest.skipUnless(grpc is not None, 'grpcio is required for live gRPC')
  def test_both_analyzers_complete_secure_grpc_calls_without_tensorflow(self):
    for filename, classname in (('analyzer', 'VideoAnalyzer'),
                                ('signalstateanalyzer', 'SignalVideoAnalyzer')):
      with self.subTest(analyzer=classname):
        module, stub = load_analyzer(filename)
        channels = []

        def make_stub(channel):
          channels.append(channel)
          return types.SimpleNamespace(Predict=channel.unary_unary(
            '/transport.Echo/Predict', request_serializer=lambda value: value,
            response_deserializer=lambda value: value))

        stub.side_effect = make_stub
        with self.grpc_server() as port:
          arguments = analyzer_arguments(filename, 'localhost:{}'.format(port))
          arguments.update(zip(('tls_ca', 'tls_cert', 'tls_key'), self.paths()))
          with mock.patch.object(module, 'Popen', return_value=
                                 types.SimpleNamespace(pid=123, returncode=0)):
            analyzer = getattr(module, classname)(**arguments)
          try:
            self.assertEqual(analyzer.service_stub.Predict(b'ping', timeout=5),
                             b'pong:ping')
          finally:
            for channel in channels:
              channel.close()

  @unittest.skipUnless(grpc is not None, 'grpcio is required for live gRPC')
  def test_grpc_rejects_unknown_ca_wrong_hostname_and_expired_server(self):
    cases = (('untrusted-ca', 'server', 'localhost'),
             ('ca', 'server', '127.0.0.1'), ('ca', 'expired', 'localhost'))
    for ca, identity, hostname in cases:
      with self.subTest(ca=ca, identity=identity, hostname=hostname):
        with self.grpc_server(identity) as port:
          channel = transport.create_model_channel(
            '{}:{}'.format(hostname, port), *self.paths(ca=ca))
          try:
            rpc = channel.unary_unary('/transport.Echo/Predict')
            with self.assertRaises(grpc.RpcError) as failure:
              rpc(b'ping', timeout=3)
            self.assertEqual(failure.exception.code(),
                             grpc.StatusCode.UNAVAILABLE)
          finally:
            channel.close()

  @unittest.skipUnless(grpc is not None, 'grpcio is required for live gRPC')
  def test_grpc_server_refuses_missing_client_certificate(self):
    with self.grpc_server() as port:
      credentials = grpc.ssl_channel_credentials(
        root_certificates=Path(self.paths()[0]).read_bytes())
      channel = grpc.secure_channel('localhost:{}'.format(port), credentials)
      try:
        with self.assertRaises(grpc.RpcError):
          channel.unary_unary('/transport.Echo/Predict')(b'ping', timeout=3)
      finally:
        channel.close()

  def test_serving_config_cli_embeds_pem_requires_clients_and_uses_mode0600(self):
    with tempfile.TemporaryDirectory() as directory:
      destination = Path(directory) / 'ssl.pbtxt'
      destination.write_text('old configuration')
      destination.chmod(0o644)
      ca, cert, key = self.paths(identity='server')
      subprocess.run(
        [sys.executable, '-B', '-m', 'utils.transport', '--tls-ca', ca,
         '--tls-cert', cert, '--tls-key', key, '--serving-config',
         str(destination)], cwd=str(ROOT), check=True, timeout=10,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      config = destination.read_text()
      fields = dict(line.split(': ', 1) for line in config.splitlines())
      self.assertEqual(fields['client_verify'], 'true')
      for field, path in (('server_key', key), ('server_cert', cert),
                          ('custom_ca', ca)):
        self.assertEqual(json.loads(fields[field]), Path(path).read_text())
      if os.name == 'posix':
        self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o600)
      self.assertEqual(list(Path(directory).iterdir()), [destination])

  def test_invalid_serving_input_does_not_replace_existing_config(self):
    with tempfile.TemporaryDirectory() as directory:
      destination = Path(directory) / 'ssl.pbtxt'
      destination.write_text('keep this')
      with self.assertRaises(ValueError):
        transport.main(['--tls-ca', '', '--tls-cert', self.paths()[1],
                        '--tls-key', self.paths()[2], '--serving-config',
                        str(destination)])
      self.assertEqual(destination.read_text(), 'keep this')
      self.assertEqual(list(Path(directory).iterdir()), [destination])


if __name__ == '__main__':
  unittest.main()
