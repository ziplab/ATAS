import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from atas.request import load_plan
from atas.world.geometry_client import render_payload
from atas.providers.wan.generation import build_request
from atas.providers.wan.types import V2VTask, ReferenceRole
@pytest.fixture
def plan(tmp_path):
    path = tmp_path / "story.yaml"
    path.write_text('''shots:
  - id: source
    location_id: room
    event_id: event
    duration: 2
    prompt: A ball rolls across the room.
    camera_move_name: locked
  - id: new
    location_id: room
    event_id: event
    duration: 2
    prompt: The same ball rolling, seen from the side.
    camera_relation: recapture
    source_shot_id: source
    camera_move_name: locked
    camera_side_degrees: 35
''')
    return load_plan(path)






def test_request_is_reobserve_not_continue(plan, tmp_path):
    video, still = tmp_path / "geometry.mp4", tmp_path / "event.png"
    video.touch()
    still.touch()
    request, _, _ = build_request(plan.shots[1], output_dir=tmp_path,
        geometry_video=video, first_frame=still, continuity_still=None,
        same_event_recapture=True, require_files=True)
    assert request.task is V2VTask.REOBSERVE
    assert request.condition.task is V2VTask.REOBSERVE
    assert request.condition.reference_image == still
    assert [r.path for r in request.condition.references if r.media_type == "video"] == [video]
    assert all(r.role is not ReferenceRole.CONTINUITY_MEMORY for r in request.condition.references)
    assert "不要续写新动作" in request.prompt
    assert "已按剧本重画" not in request.prompt
    with pytest.raises(ValueError):
        build_request(plan.shots[1], output_dir=tmp_path, geometry_video=None,
            first_frame=still, continuity_still=None, same_event_recapture=True, require_files=True)


