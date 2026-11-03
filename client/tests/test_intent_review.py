import json
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from atas.review.film_review import decide, review_shot, require_review
def response():
    return {'checks':[{'criterion':name,'status':'pass','severity':'minor','confidence':.99,
        'evidence':'visible at frames 0 and 6'} for name in ['subject','framing','motion','action','event','world_consistency']], 'summary':'intent achieved'}


@pytest.mark.parametrize('status,severity,confidence,expected',[
    ('uncertain','critical',.99,'warning'),('fail','minor',.99,'warning'),
    ('fail','critical',.7,'warning'),('fail','critical',.8,'hold'),('fail','critical',.95,'hold'),('pass','minor',.99,'pass')])
def test_only_clear_critical_contradictions_block(status,severity,confidence,expected):
    r=response();r['checks'][0].update(status=status,severity=severity,confidence=confidence)
    assert decide(r)==expected


def test_incomplete_review_never_silently_passes():
    with pytest.raises(ValueError):decide({'checks':[]})


def test_world_drift_blocks_even_when_all_film_intent_checks_pass():
    r=response();r['checks'][-1].update(status='fail',severity='critical',confidence=.98,
        evidence='Historical green fabric became pink floral in current frame 59')
    assert decide(r)=='hold'
    r['checks'][-1].update(status='uncertain',evidence='Surface occluded in historical frames')
    assert decide(r)=='warning'


def test_old_five_criterion_response_is_not_silently_accepted():
    r=response();r['checks'].pop()
    with pytest.raises(ValueError):decide(r)


def test_review_cache_binds_content_and_original_screenplay(tmp_path,monkeypatch):
    video=tmp_path/'film.mp4'; writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),10,(64,48))
    for i in range(10):writer.write(np.full((48,64,3),i*20,np.uint8))
    writer.release()
    shot=SimpleNamespace(prompt='Pull back to show the bed',shot_size='medium',shot_perspective='eye level',camera_move_name='pull_back',entity_ids=('Rabbit',))
    calls=[]
    def chat(*a,**kw):calls.append(a[0]);return json.dumps(response())
    monkeypatch.setattr('atas.providers.vlm.chat_with_retry',chat)
    a=review_shot(shot,video,tmp_path,stage='film');b=review_shot(shot,video,tmp_path,stage='film')
    assert a==b and len(calls)==1
    shot.prompt='Push in on the face'
    c=review_shot(shot,video,tmp_path,stage='film')
    assert len(calls)==2 and c['cache_key']!=a['cache_key']
    assert not c['exact_camera_or_event_verified']


def test_hold_explains_reason():
    r=response();r['checks'][2].update(status='fail',severity='critical',confidence=.95,evidence='camera pushes in instead of pulling back')
    with pytest.raises(RuntimeError,match='pushes in'):
        require_review({**r,'stage':'film','decision':'hold'})


