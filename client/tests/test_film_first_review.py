import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from atas.review import film_review as mod
def response():
    return {'checks': [dict(criterion=k, status='pass', severity='minor', confidence=.99,
        evidence='Visible at current frames 0 and 6.') for k in (
            'subject', 'framing', 'motion', 'action', 'event', 'world_consistency')]}


@pytest.mark.parametrize('detail', ['No furrowed brow.', 'Tiny eye reflection not visible.', 'Gesture amplitude is slightly small.'])
def test_minor_explicit_details_warn_without_failing_movie(detail):
    r = response()
    r['checks'][3].update(status='fail', evidence=detail)
    assert mod.decide(r, execution_profile='film_first') == 'warning'
    assert r['checks'][3]['status'] == 'fail'


@pytest.mark.parametrize('criterion,basis', [('subject', 'identity'), ('action', 'narrative'),
    ('world_consistency', 'severe_world'), ('motion', 'source_camera')])
def test_core_contradictions_still_hold(criterion, basis):
    r = response()
    next(c for c in r['checks'] if c['criterion'] == criterion).update(status='fail', severity='critical',
        critical_basis=basis, impact='The essential required participant or event is contradicted.')
    assert mod.decide(r, execution_profile='film_first') == 'hold'


def test_film_first_requires_impact_but_does_not_change_strict_contract():
    r = response()
    r['checks'][3].update(status='fail', severity='critical')
    assert mod.decide(r) == 'hold'
    with pytest.raises(ValueError, match='non-empty impact'):
        mod.decide(r, execution_profile='film_first')
    r['checks'][3].update(critical_basis='narrative', impact='Changes the ending.', confidence=.7)
    assert mod.decide(r, execution_profile='film_first') == 'warning'


@pytest.mark.parametrize('criterion', ['subject', 'action', 'event'])
def test_recapture_cannot_soften_observed_event_contradictions(criterion):
    r = response()
    check = next(c for c in r['checks'] if c['criterion'] == criterion)
    check.update(status='fail', severity='minor')
    assert mod.decide(copy.deepcopy(r), execution_profile='film_first') == 'warning'
    assert mod.decide(r, execution_profile='film_first', same_event_recapture=True) == 'hold'
    assert check['severity'] == 'critical'
    check['status'] = 'uncertain'
    assert mod.decide(r, execution_profile='film_first', same_event_recapture=True) == 'warning'


def test_prompt_profile_and_version_are_distinct(monkeypatch):
    profile, version, prompt = mod.review_policy('film')
    assert profile == 'film_first' and version == mod.FILM_FIRST_POLICY_VERSION
    assert 'explicitly mentioned in the screenplay does NOT make a detail critical' in prompt
    assert 'source_camera is critical only when the' in prompt
    assert 'scene_observations' in prompt and 'observed_dynamic_classes' in prompt
    assert 'Never downgrade a genuine geometry/layout contradiction' in prompt
    assert 'unambiguous contradiction of an explicit screenplay requirement' not in prompt
    assert mod.review_policy('preview')[2] == mod.PROMPT


def test_cache_hit_precedes_decode(tmp_path, monkeypatch):
    video = tmp_path / 'result.mp4'; video.write_bytes(b'fixture video')
    shot = SimpleNamespace(prompt='An artist considers the painting.', shot_size='medium',
        shot_perspective='eye level', camera_move_name='locked', entity_ids=(), raw={})
    monkeypatch.delenv('AVWM_EXECUTION_PROFILE', raising=False)
    monkeypatch.setattr(mod, 'frames', lambda *a, **k: ([np.zeros((8,8,3), np.uint8)], [{'frame': 0}]))
    calls = []
    def chat(prompt, **kwargs):
        calls.append(prompt)
        return json.dumps({**response(), 'observed_dynamic_classes': [{'noun': 'person', 'evidence': 'Frame 0.'}]})
    monkeypatch.setattr('atas.providers.vlm.chat_with_retry', chat)
    film = mod.review_shot(shot, video, tmp_path, stage='film')
    assert len(calls) == 1
    assert film['input']['execution_profile'] == 'film_first'
    assert film['observed_dynamic_classes'][0]['noun'] == 'person'
    monkeypatch.setattr(mod, 'frames', lambda *a, **k: pytest.fail('cached review must not decode'))
    assert mod.review_shot(shot, video, tmp_path, stage='film') == film
    assert len(calls) == 1


def test_recapture_without_reference_never_claims_event_pass(tmp_path, monkeypatch):
    video = tmp_path / 'result.mp4'; video.write_bytes(b'fixture video')
    shot = SimpleNamespace(prompt='Replay the jump.', shot_size='medium', shot_perspective='eye level',
        camera_move_name='locked', entity_ids=(), raw={'render_mode': 'recapture'})
    monkeypatch.setenv('AVWM_EXECUTION_PROFILE', 'film_first')
    monkeypatch.setattr(mod, 'frames', lambda *a, **k: ([np.zeros((8,8,3), np.uint8)], [{'frame': 0}]))
    monkeypatch.setattr('atas.providers.vlm.chat_with_retry', lambda *a, **k: json.dumps(response()))
    result = mod.review_shot(shot, video, tmp_path, stage='film')
    assert result['decision'] == 'warning'
    assert next(c for c in result['checks'] if c['criterion'] == 'event')['status'] == 'uncertain'
    assert result['exact_camera_or_event_verified'] is False
