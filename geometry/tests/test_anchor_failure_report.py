import json
import sys
import numpy as np
import pytest
from atas_geometry.register import anchored_fusion as a
def test_too_few_points_is_explicit_reference_rejection():
    with pytest.raises(a.AnchorFitError):a.fit_sim3(np.zeros((3,3)),np.zeros((3,3)))

def test_anchor_failure_writes_rejected_report_not_partial_world(tmp_path,monkeypatch):
    world=tmp_path/'world.npz';world.write_bytes(b'unchanged world')
    new=tmp_path/'new';ref=tmp_path/'ref';out=tmp_path/'out'
    monkeypatch.setattr(sys,'argv',['anchored_fusion','--reference-l0',str(ref),'--new-l0',str(new),'--l1',str(world),'--output',str(out),'--vista-repo','vista','--model-id','pi3x'])
    monkeypatch.setattr(a,'strict_store',lambda *args:object())
    def reject(*args,**kwargs):raise a.AnchorFitError('not enough paired anchor points')
    monkeypatch.setattr(a,'reconstruct_conditioned',reject)
    a.main()
    d=json.loads((out/'fusion_report.json').read_text())
    assert not d['merged'] and not d['reconstruction']['geometry_accepted']
    assert d['cross_shot_validation']['status']=='not_run'
    assert d['source_l0']==str(new) and d['reference_l0']==str(ref)
    assert world.read_bytes()==b'unchanged world' and not (out/'l1_candidate.npz').exists()
