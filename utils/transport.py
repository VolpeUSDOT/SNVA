"""Verified mutual-TLS transports and TensorFlow Serving SSL configuration."""

import argparse
import ipaddress
import json
import os
import ssl
import tempfile
from urllib.parse import urlencode, urlsplit


def _read_tls_files(tls_ca, tls_cert, tls_key):
  material = []
  for name, path in (('tls_ca', tls_ca), ('tls_cert', tls_cert),
                     ('tls_key', tls_key)):
    if path is None or not os.fspath(path):
      raise ValueError('{} must be a nonempty TLS file path'.format(name))
    with open(path, 'rb') as pem_file:
      pem = pem_file.read()
    if not pem.strip():
      raise ValueError('{} TLS file is empty'.format(name))
    material.append(pem)
  return tuple(material)


def _load_tls_material(tls_ca, tls_cert, tls_key):
  material = _read_tls_files(tls_ca, tls_cert, tls_key)
  context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
  context.check_hostname = True
  context.verify_mode = ssl.CERT_REQUIRED
  if hasattr(context, 'minimum_version'):
    context.minimum_version = ssl.TLSVersion.TLSv1_2
  else:
    # Python 3.6 has no minimum_version API.
    context.options |= (ssl.OP_NO_SSLv2 | ssl.OP_NO_SSLv3 |
                        ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1)
  context.load_verify_locations(cadata=material[0].decode('ascii'))
  context.load_cert_chain(
    certfile=tls_cert, keyfile=tls_key, password=lambda: b'')
  return context, material


def create_client_context(tls_ca, tls_cert, tls_key):
  """Require a trusted server identity and a valid client certificate/key."""
  context, _ = _load_tls_material(tls_ca, tls_cert, tls_key)
  return context


def build_control_url(host, connection_id=None, reconnect_token=None):
  """Build the sole supported WSS endpoint from a host or WSS authority."""
  if not isinstance(host, str) or not host:
    raise ValueError('control host must be nonempty')
  if any(character.isspace() or ord(character) < 32 or ord(character) == 127
         for character in host) or '\\' in host:
    raise ValueError('control host contains invalid characters')
  if '://' in host:
    parsed = urlsplit(host)
    if parsed.scheme.lower() != 'wss':
      raise ValueError('control connections require the wss scheme')
    authority = parsed.netloc
    if parsed.path or parsed.query or parsed.fragment:
      raise ValueError('control host must not include a path, query or fragment')
  else:
    authority = host
  if not authority or any(character in authority for character in '/?#@'):
    raise ValueError('control host must be an authority without userinfo')
  if '?' in host or '#' in host:
    raise ValueError('control host must not include a query or fragment')
  if not authority.startswith('[') and authority.count(':') > 1:
    authority = '[{}]'.format(ipaddress.IPv6Address(authority))
  parsed = urlsplit('//' + authority)
  if not parsed.hostname or parsed.path or authority.endswith(':'):
    raise ValueError('invalid control host')
  if authority.startswith('['):
    ipaddress.IPv6Address(parsed.hostname)
    suffix = authority[authority.index(']') + 1:]
    if suffix and not suffix.startswith(':'):
      raise ValueError('invalid IPv6 control host')
  if parsed.port is not None and not 1 <= parsed.port <= 65535:
    raise ValueError('invalid control port')
  parameters = []
  if connection_id is not None:
    parameters.append(('id', connection_id))
  if reconnect_token is not None:
    parameters.append(('reconnectToken', reconnect_token))
  query = '?' + urlencode(parameters) if parameters else ''
  return 'wss://{}/registerProcess{}'.format(authority, query)


def create_model_channel(host, tls_ca, tls_cert, tls_key, options=None):
  """Create a verified gRPC mTLS channel without hostname overrides."""
  if not isinstance(host, str) or not host.strip():
    raise ValueError('model host must be nonempty')
  if options is not None:
    options = tuple(options)
    if any(name in ('grpc.ssl_target_name_override', 'grpc.default_authority')
           for name, _ in options):
      raise ValueError('model TLS hostname overrides are not permitted')
  _, (root_certificates, certificate_chain, private_key) = _load_tls_material(
    tls_ca, tls_cert, tls_key)
  # Keep the SSLConfig CLI usable without TensorFlow or gRPC installed.
  import grpc
  credentials = grpc.ssl_channel_credentials(
    root_certificates=root_certificates, private_key=private_key,
    certificate_chain=certificate_chain)
  return grpc.secure_channel(host, credentials, options=options)


def _write_serving_config(tls_ca, tls_cert, tls_key, serving_config):
  _, (ca, cert, key) = _load_tls_material(tls_ca, tls_cert, tls_key)
  config = ''.join('{}: {}\n'.format(name, json.dumps(pem.decode('ascii')))
                   for name, pem in (('server_key', key), ('server_cert', cert),
                                     ('custom_ca', ca)))
  config += 'client_verify: true\n'
  destination = os.path.abspath(serving_config)
  descriptor, temporary_path = tempfile.mkstemp(
    prefix='.ssl-config-', dir=os.path.dirname(destination))
  try:
    with os.fdopen(descriptor, 'w', encoding='ascii') as config_file:
      os.chmod(temporary_path, 0o600)
      config_file.write(config)
    # Replace existing files atomically without inheriting permissive modes.
    os.replace(temporary_path, destination)
  finally:
    if os.path.exists(temporary_path):
      os.unlink(temporary_path)


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--tls-ca', required=True)
  parser.add_argument('--tls-cert', required=True)
  parser.add_argument('--tls-key', required=True)
  parser.add_argument('--serving-config', required=True)
  args = parser.parse_args(argv)
  _write_serving_config(
    args.tls_ca, args.tls_cert, args.tls_key, args.serving_config)


if __name__ == '__main__':
  main()
