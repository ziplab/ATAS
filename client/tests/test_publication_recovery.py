import json
import pytest
from atas.runner.publication_recovery import publish_with_rebase
from atas.world.geometry_client import GeometryWaitUncertain
def result(jid='old', registered=True):
    return dict(status='succeeded',job_id=jid,cameras_npz='c',l0_dir='l0',l1_npz='l1',l1_point_count=3,
                fuse=dict(registered=registered,publication_status='pending_review',candidate_job_id=jid))

class Client:
    def __init__(self): self.calls=[]; self.waits=[]; self.timeout=False; self.reject=False
    def _request(self, method, route, payload):
        self.calls.append((route, payload['candidate_job_id']))
        if route.endswith('world-rebase'): return {'job_id':'new'}
        if payload['candidate_job_id']=='old':
            raise RuntimeError('geometry HTTP 409: world changed before publication; candidate must be rebuilt')
        return {'publication_status':'committed','novel_points':2}
    def wait(self,jid):
        self.waits.append(jid)
        if self.timeout: raise GeometryWaitUncertain('still running')
        return result(jid, not self.reject)

def test_rebase_reuses_job_and_keeps_publication_gate(tmp_path):
    g=Client(); current,pub=publish_with_rebase(g,tmp_path,result(),{})
    assert current['job_id']=='new' and pub['publication_status']=='committed'
    assert g.calls==[('/v1/world-publication','old'),('/v1/world-rebase','old'),('/v1/world-publication','new')]
    assert g.waits==['new']

def test_timeout_resumes_same_world_job_without_resubmission(tmp_path):
    g=Client();g.timeout=True
    with pytest.raises(GeometryWaitUncertain):publish_with_rebase(g,tmp_path,result(),{})
    g.timeout=False
    publish_with_rebase(g,tmp_path,result(),{})
    assert sum(route.endswith('world-rebase') for route,_ in g.calls)==1
    assert g.waits==['new','new']

def test_bad_rebased_geometry_never_publishes(tmp_path):
    g=Client();g.reject=True
    with pytest.raises(RuntimeError,match='rejected geometry'):publish_with_rebase(g,tmp_path,result(),{})
    assert g.calls[-1][0].endswith('world-rebase')

def test_other_conflict_never_rebases(tmp_path):
    g=Client()
    def reject(*args):raise RuntimeError('geometry HTTP 409: review and reconstruction lineage differ')
    g._request=reject
    with pytest.raises(RuntimeError,match='lineage'):publish_with_rebase(g,tmp_path,result(),{})
    assert not (tmp_path/'world_rebase.json').exists()

def test_rebase_limit_is_durable(tmp_path):
    (tmp_path/'world_rebase.json').write_text(json.dumps({'attempts':[{'status':'completed'}]*2}))
    g=Client()
    with pytest.raises(RuntimeError,match='budget exhausted'):publish_with_rebase(g,tmp_path,result(),{})
    assert len(g.calls)==1
