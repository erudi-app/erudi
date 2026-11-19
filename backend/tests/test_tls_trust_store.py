"""The frozen backend must not run with an empty TLS trust store (#758).

In the packaged macOS app ``_ssl`` links psycopg_binary's ``libcrypto``, whose
OPENSSLDIR is a build path that does not exist on the user's machine, so
``ssl.create_default_context()`` loads no CA at all and every verified HTTPS
connection built on it (ddgs' DuckDuckGo engine) fails with
``CERTIFICATE_VERIFY_FAILED``. ``run.configure_tls_trust_store`` points
``SSL_CERT_FILE`` at the bundled certifi file in that one situation.
"""

import os
import ssl
import sys
from types import SimpleNamespace

import pytest

import run

MISSING = "/tmp/libpq.build/cert.pem"
CERTIFI = "/bundle/certifi/cacert.pem"


def _paths(cafile):
    return lambda: SimpleNamespace(cafile=cafile, openssl_cafile=cafile or MISSING)


def _call(
    environ,
    *,
    frozen=True,
    platform_name="darwin",
    cafile=None,
    existing=(CERTIFI,),
):
    return run.configure_tls_trust_store(
        environ=environ,
        frozen=frozen,
        platform_name=platform_name,
        default_verify_paths=_paths(cafile),
        certifi_where=lambda: CERTIFI,
        path_exists=lambda p: p in existing,
    )


@pytest.mark.unit
def test_frozen_with_missing_default_cafile_sets_certifi():
    env: dict[str, str] = {}
    message = _call(env)
    assert env["SSL_CERT_FILE"] == CERTIFI
    assert message and "certifi" in message and message.isascii()


@pytest.mark.unit
def test_linux_prefers_first_existing_system_bundle():
    env: dict[str, str] = {}
    fedora = "/etc/pki/tls/certs/ca-bundle.crt"
    message = _call(env, platform_name="linux", existing=(fedora, "/etc/ssl/cert.pem", CERTIFI))
    assert env["SSL_CERT_FILE"] == fedora
    assert fedora in message


@pytest.mark.unit
@pytest.mark.parametrize("bundle", run.LINUX_SYSTEM_CA_BUNDLES)
def test_linux_each_system_bundle_is_used_when_alone(bundle):
    env: dict[str, str] = {}
    _call(env, platform_name="linux", existing=(bundle, CERTIFI))
    assert env["SSL_CERT_FILE"] == bundle


@pytest.mark.unit
def test_linux_without_system_bundle_falls_back_to_certifi():
    env: dict[str, str] = {}
    _call(env, platform_name="linux")
    assert env["SSL_CERT_FILE"] == CERTIFI


@pytest.mark.unit
def test_macos_ignores_linux_bundles():
    env: dict[str, str] = {}
    _call(env, platform_name="darwin", existing=(*run.LINUX_SYSTEM_CA_BUNDLES, CERTIFI))
    assert env["SSL_CERT_FILE"] == CERTIFI


@pytest.mark.unit
def test_user_ssl_cert_file_is_left_alone():
    env = {"SSL_CERT_FILE": "/corp/proxy-ca.pem"}
    assert _call(env) is None
    assert env == {"SSL_CERT_FILE": "/corp/proxy-ca.pem"}


@pytest.mark.unit
def test_existing_default_cafile_is_left_alone():
    env: dict[str, str] = {}
    assert _call(env, cafile="/etc/ssl/cert.pem", existing=("/etc/ssl/cert.pem", CERTIFI)) is None
    assert env == {}


@pytest.mark.unit
def test_not_frozen_is_left_alone():
    env: dict[str, str] = {}
    assert _call(env, frozen=False) is None
    assert env == {}


@pytest.mark.unit
def test_windows_is_left_alone():
    # load_default_certs reads the Windows system store there.
    env: dict[str, str] = {}
    assert _call(env, platform_name="win32") is None
    assert env == {}


@pytest.mark.unit
def test_missing_certifi_file_is_not_pointed_at():
    env: dict[str, str] = {}
    assert _call(env, existing=()) is None
    assert env == {}


@pytest.mark.unit
@pytest.mark.skipif(sys.platform == "win32", reason="Windows loads its system store regardless")
def test_default_context_gains_certifi_roots(monkeypatch):
    # The defect: the default cafile is missing, so the context has no CA.
    monkeypatch.setenv("SSL_CERT_FILE", MISSING)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    assert ssl.create_default_context().cert_store_stats()["x509_ca"] == 0

    # The fix: no user variable and a missing default cafile, so certifi's file
    # is used and a fresh default context carries its CAs.
    monkeypatch.delenv("SSL_CERT_FILE")
    message = run.configure_tls_trust_store(
        environ=os.environ,
        frozen=True,
        platform_name="darwin",
        default_verify_paths=_paths(None),
    )
    try:
        assert message is not None
        import certifi

        assert os.environ["SSL_CERT_FILE"] == certifi.where()
        assert ssl.create_default_context().cert_store_stats()["x509_ca"] > 0
    finally:
        os.environ.pop("SSL_CERT_FILE", None)
