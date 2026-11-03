"""Resolve shot-scoped character references for cinematic shot planning.

Character identity originates in project assets, while the provider-facing body
and face boards are selected per shot. The director binds only visible tracked
characters and preserves the same A/B/C/D ordering across every board.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CastAssetSelection:
    """Character reference selected during story-to-shot compilation."""

    image: Path | None = None
    face_image: Path | None = None
    # Read-only compatibility for old persisted packages. New non-legacy
    # boards use only image/body_board and face_image/face_board.
    hero_face_image: Path | None = None
    board_id: str = ""
    visible_cast: tuple[str, ...] = ()
    bindings: tuple[str, ...] = ()
    legacy: bool = False


def resolve_cast_assets(
    story: Mapping[str, Any],
    raw_shot: Mapping[str, Any],
    entity_ids: tuple[str, ...],
) -> CastAssetSelection:
    """Resolve a shot cast board without making provider-specific decisions.

    ``visible_cast`` is optional in a shot.  When omitted, entity IDs that are
    declared in ``asset_pack.characters`` are treated as visible characters.
    A shot-scoped board is preferred over individual cards. The project-wide
    board lookup remains only as compatibility for persisted legacy stories.
    """

    pack_value = story.get("asset_pack")
    pack = pack_value if isinstance(pack_value, Mapping) else {}
    characters_value = pack.get("characters")
    characters = characters_value if isinstance(characters_value, Mapping) else {}

    explicit_visible = raw_shot.get("visible_cast")
    if isinstance(explicit_visible, Sequence) and not isinstance(
        explicit_visible, (str, bytes)
    ):
        visible_cast = _unique_strings(explicit_visible)
    else:
        visible_cast = tuple(entity_id for entity_id in entity_ids if entity_id in characters)
    if not visible_cast:
        return CastAssetSelection()

    # Wan should only receive a small, explicitly tracked cast.  For a crowd
    # shot the authored order is the importance order; the remaining people
    # stay provider-generated background extras and never enter the identity
    # reference pack.
    visible_cast = visible_cast[:4]

    per_shot = pack.get("cast_boards")
    board_value = None
    if isinstance(per_shot, Mapping):
        board_value = per_shot.get(str(raw_shot.get("id", "")))
    board_key = "shot"
    if isinstance(per_shot, Mapping):
        board_key = str(raw_shot.get("id", ""))
    if not isinstance(board_value, Mapping):
        board_value = pack.get("cast_board")
        if isinstance(board_value, Mapping) and not bool(board_value.get("legacy")):
            raise ValueError(
                f"{board_key}: global cast_board is not shot-scoped; mark it legacy=true "
                "or provide cast_boards[shot_id] with body_board and face_board"
            )
        if not isinstance(board_value, Mapping) and "main_cast_board" in pack:
            legacy_board = pack.get("main_cast_board")
            if isinstance(legacy_board, Mapping) and bool(legacy_board.get("legacy")):
                board_value = legacy_board
            else:
                raise ValueError(
                    f"{board_key}: main_cast_board is legacy-only; provide a shot-scoped "
                    "cast_boards entry or set legacy=true explicitly"
                )
    board = board_value if isinstance(board_value, Mapping) else {}
    board_image = _path(
        board.get("body_board", board.get("image", board.get("reference_image")))
    )
    face_image = _path(board.get("face_board", board.get("face_image")))
    hero_face_image = _path(board.get("hero_face", board.get("hero_face_image")))
    legacy = bool(board.get("legacy"))
    if board_image is not None and face_image is None and not legacy:
        raise ValueError(
            f"{board_key}: production cast board requires body_board and face_board"
        )
    if board_image is not None and (face_image is not None or legacy):
        board_id = _string(board.get("id", board.get("asset_id", "shot_cast_board")))
        if not legacy and face_image is None:
            raise ValueError(
                f"{board_key}: production cast board requires body_board and face_board"
            )
        return CastAssetSelection(
            image=board_image,
            face_image=face_image,
            hero_face_image=hero_face_image,
            board_id=board_id or ("legacy_cast_board" if legacy else "shot_cast_board"),
            visible_cast=visible_cast,
            bindings=_board_bindings(board, characters, visible_cast),
            legacy=legacy,
        )

    # Existing stories may only have individual character cards.  Keep those
    # usable, but do not label the fallback as a multi-person board.
    for character_id in visible_cast:
        spec_value = characters.get(character_id)
        spec = spec_value if isinstance(spec_value, Mapping) else {}
        image = _path(spec.get("reference_image", spec.get("image")))
        if image is not None and bool(spec.get("legacy")):
            name = _string(spec.get("name", character_id)) or character_id
            return CastAssetSelection(
                image=image,
                visible_cast=visible_cast,
                bindings=(f"{character_id}({name})",),
                legacy=True,
            )
    return CastAssetSelection(visible_cast=visible_cast)


def _board_bindings(
    board: Mapping[str, Any],
    characters: Mapping[str, Any],
    visible_cast: tuple[str, ...],
) -> tuple[str, ...]:
    slots_value = board.get("slots")
    slots = slots_value if isinstance(slots_value, Mapping) else {}
    bindings: list[str] = []
    bound: set[str] = set()
    for slot, value in slots.items():
        character_id = ""
        character_name = ""
        if isinstance(value, Mapping):
            character_id = _string(value.get("character_id", value.get("id", "")))
            character_name = _string(value.get("name", ""))
        else:
            character_id = _string(value)
        if character_id not in visible_cast:
            continue
        spec_value = characters.get(character_id)
        spec = spec_value if isinstance(spec_value, Mapping) else {}
        character_name = character_name or _string(spec.get("name", "")) or character_id
        bindings.append(f"{_string(slot)}={character_id}({character_name})")
        bound.add(character_id)
    for character_id in visible_cast:
        if character_id in bound:
            continue
        spec_value = characters.get(character_id)
        spec = spec_value if isinstance(spec_value, Mapping) else {}
        name = _string(spec.get("name", character_id)) or character_id
        bindings.append(f"{character_id}({name})")
    return tuple(bindings)


def _unique_strings(values: Sequence[Any]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _string(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return tuple(result)


def _path(value: Any) -> Path | None:
    text = _string(value)
    return Path(text).expanduser().resolve() if text else None


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""
