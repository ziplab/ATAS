from pathlib import Path
import pytest
from atas.world.geometry_client import GeometryClient
def test_remote_gpu_path_never_stat_ed_on_client(tmp_path,monkeypatch):
    remote=Path('/var/tmp/geometry-service-v-zhanghaomin/jobs/render_x/memory_input.mp4')
    real=Path.is_file
    def is_file(path):
        if path==remote:raise OSError(24,'Too many open files',str(path))
        return real(path)
    monkeypatch.setattr(Path,'is_file',is_file)
    client=GeometryClient(base_url='https://geometry.example')
    calls=[]
    monkeypatch.setattr(client,'_download',lambda url,dest:(calls.append(url),dest.write_bytes(b'mp4')))
    dest=tmp_path/'out.mp4'
    assert client.fetch_memory_input({'job_id':'render_x','memory_input':str(remote)},dest)==dest
    assert calls==['https://geometry.example/v1/jobs/render_x/artifacts/memory_input.mp4']


def test_http_error_does_not_use_gpu_path_as_local(tmp_path,monkeypatch):
    client=GeometryClient(base_url='https://geometry.example')
    monkeypatch.setattr(Path,'is_file',lambda p:pytest.fail('remote local lookup'))
    def fail(*args):raise RuntimeError('unavailable')
    monkeypatch.setattr(client,'_download',fail)
    assert client.fetch_memory_input({'job_id':'render_x','memory_input':'/remote/video.mp4'},tmp_path/'out') is None


def test_entrant_can_quote_original_source_without_accepting_invented_evidence():
    from types import SimpleNamespace
    from atas.shot.camera_planner import validate_for_shot
    shot=SimpleNamespace(camera_move_name='track_right',entity_ids=['cat','butterfly'],
        prompt='Cat spots Garden Butterfly above flowers.',
        raw={'source_requirements':[{'content':{'video_prompt':'The cat spots a butterfly fluttering above the flowers.'}}]})
    plan={'motion':{'name':'track_right'},'subjects':[
        {'label':'cat','kind':'dynamic','framing_role':'focal'},
        {'label':'flower','kind':'static','subject_type':'object','part':'object','framing_role':'context'}],
        'planned_entrants':[{'label':'butterfly','anchor_label':'flower',
            'screenplay_evidence':'a butterfly fluttering above the flowers',
            'placement':'above the flowers','status':'planned_unobserved'}]}
    assert validate_for_shot(plan,shot,{'camera_relation':'continuous'}) is plan
    plan['planned_entrants'][0]['screenplay_evidence']='an invented flying dragon'
    with pytest.raises(ValueError,match='not copied verbatim'):
        validate_for_shot(plan,shot,{'camera_relation':'continuous'})
