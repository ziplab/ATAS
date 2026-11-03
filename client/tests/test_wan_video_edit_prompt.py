from __future__ import annotations

from types import SimpleNamespace

from atas.providers.wan.generation import _prompt
def shot():
    return SimpleNamespace(
        style_contract="keep picture-book style",
        provider_prompt=(
            "Close-up rabbit in bedroom with window, moon, and wall painting."
        ),
        prompt="fallback",
        video_edit_prompt=(
            "Little Brown Rabbit close-up, expectant bright eyes and a slight smile. "
            "Only the lamp edge, headboard, pillow, and duvet are visible. "
            "Both hands remain beneath the duvet. "
            "禁止添加视频1未显示的画外背景、开口、夜空天体、墙面装饰或新墙痕。"
        ),
        shot_perspective="Medium close up, eye level shot",
        shot_size="close_up",
        camera_angle="eye-level",
        entity_ids=("Little Brown Rabbit",),
    )


def test_video_edit_uses_visible_frame_prompt_and_excludes_offscreen_tokens() -> None:
    value = _prompt(
        shot(),
        (),
        has_geometry=True,
        geometry_control_mode="video_edit",
    )
    lowered = value.lower()
    assert "little brown rabbit close-up" in lowered
    assert "lamp edge" in lowered
    assert "both hands remain beneath the duvet" in lowered
    assert "window" not in lowered
    assert "moon" not in lowered
    assert "painting" not in lowered
    assert "视频1中不可见的场景物体不要添加" in value


def test_reference_default_keeps_original_provider_prompt() -> None:
    value = _prompt(
        shot(),
        (),
        has_geometry=True,
        geometry_control_mode="reference",
    ).lower()
    assert "window" in value
    assert "moon" in value
    assert "wall painting" in value
    assert "little brown rabbit close-up" not in value
