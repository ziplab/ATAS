import json
import pytest
from atas.world.geometry_client import GeometryClient
@pytest.mark.parametrize('url', ['http://127.0.0.1:18300', 'http://[::1]:18300', 'https://geometry.example.com'])
def test_local_endpoint_and_remote_https(tmp_path, monkeypatch, url):
    p = tmp_path / 'endpoint.json'
    p.write_text(json.dumps({'url': url}))
    monkeypatch.setenv('GEOMETRY_ENDPOINT_FILE', str(p))
    client = GeometryClient()
    client._refresh_endpoint()
    assert client.base_url == url


@pytest.mark.parametrize('url', ['http://10.0.0.1:18300', 'http://localhost.evil:18300', 'http://127.0.0.1:18300/path', 'http://user:pass@127.0.0.1:18300'])
def test_remote_plaintext_and_malformed_endpoint_rejected(tmp_path, monkeypatch, url):
    p = tmp_path / 'endpoint.json'
    p.write_text(json.dumps({'url': url}))
    monkeypatch.setenv('GEOMETRY_ENDPOINT_FILE', str(p))
    with pytest.raises(ValueError):
        GeometryClient()
