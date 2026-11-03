import json
from pathlib import Path
import pytest
from atas.world.geometry_client import GeometryClient
def test_live_client_reloads_endpoint_and_keeps_job_artifact_route(tmp_path,monkeypatch):
    path=tmp_path/'endpoint.json'
    path.write_text(json.dumps({'url':'https://first.example'}))
    monkeypatch.setenv('GEOMETRY_ENDPOINT_FILE',str(path))
    c=GeometryClient();assert c.base_url=='https://first.example'
    path.write_text(json.dumps({'url':'https://second.example'}))
    calls=[]
    monkeypatch.setattr(c,'_download',lambda url,dest:(calls.append(url),dest.write_bytes(b'validated mp4')))
    c.fetch_memory_input({'job_id':'original-job','memory_input_url':'https://first.example/old'},tmp_path/'out.mp4')
    assert calls==['https://second.example/v1/jobs/original-job/artifacts/memory_input.mp4']


def test_explicit_endpoint_does_not_load_control_file(tmp_path,monkeypatch):
    p=tmp_path/'endpoint.json';p.write_text('broken')
    monkeypatch.setenv('GEOMETRY_ENDPOINT_FILE',str(p))
    assert GeometryClient(base_url='https://explicit.example').base_url=='https://explicit.example'


def test_no_cross_host_token_download(tmp_path,monkeypatch):
    c=GeometryClient(base_url='https://geometry.example')
    monkeypatch.setattr(c,'_download',lambda *a:pytest.fail('untrusted host'))
    assert c.fetch_memory_input({'memory_input_url':'https://other.example/a.mp4'},tmp_path/'out.mp4') is None
