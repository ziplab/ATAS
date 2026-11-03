from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from atas.providers.wan.director import DirectorAgent
from atas.providers.settings import AVWMSettings
from atas.providers.wan.types import (
    MediaReference,
    ReferenceRole,
    RouteDecision,
    ShotSpec,
    V2VCondition,
    V2VRequest,
    V2VTask,
)

from atas.shot.packing import (
    WAN30_REFERENCE_CAP,
    CharacterStill,
    reserved_reference_slots,
    select_character_stills,
)
from atas.request import PlanShot
from atas.world.scene_state import brief as scene_brief, cast_shot, context as scene_context
def _shot_spec(shot: PlanShot, *, first_frame: Path | None) -> ShotSpec:
    planned = DirectorAgent().plan(
        {"schema_version": "avwm-story-v1", "shots": [shot.raw]}
    )[0]
    return replace(
        planned,
        region=replace(planned.region, scene_id=shot.scene_id),
        first_frame_image=first_frame or planned.first_frame_image,
        bootstrap_image=first_frame or planned.bootstrap_image,
        force_route=RouteDecision.GENERATED,
    )


def _prompt(
    shot: PlanShot,
    references: tuple[MediaReference, ...],
    *,
    has_geometry: bool,
    has_first_frame: bool = False,
    skipped_familiar: tuple[str, ...] = (),
    establishing_empty: bool = False,
    geometry_control_mode: str = "reference",
    same_event_recapture: bool = False,
) -> str:
    from atas.shot.prompt_author import submitted_video_prompt
    authored = submitted_video_prompt(shot, references)
    if authored is not None:
        return authored
    authored_prompt = (
        str(getattr(shot, "video_edit_prompt", "") or "").strip()
        if geometry_control_mode == "video_edit"
        else ""
    )
    parts = [
        shot.style_contract,
        scene_brief(shot),
        authored_prompt or shot.provider_prompt or shot.prompt,
    ]
    perspective = shot.shot_perspective.strip()
    if not perspective and (shot.shot_size or shot.camera_angle):
        perspective = f"{shot.shot_size.replace('_', ' ')}, {shot.camera_angle} shot".strip(", ")
    if perspective:
        parts.append(
            f"Camera / shot design: {perspective}. "
            "Match this framing and camera height; do not turn a close-up into a wide, "
            "or a bird's-eye / overhead into eye-level."
        )
    if establishing_empty:
        parts.append(
            "No focal character is specified. This does not empty an established scene. "
            "Preserve previously established characters if their positions enter the view; do not invent strangers."
        )
    if has_geometry:
        if geometry_control_mode == "video_edit":
            parts.append(
                "编辑视频1，仅将点云几何条件转成成片外观，逐帧保持相机运动、构图和遮挡；"
                "不得自行拉远、推近或旋转相机。人物按图片参考填入，图片仅作人物身份、服装和连续性参考，"
                "不得用图片构图替换视频1的镜头。视频1中不可见的场景物体不要添加。"
            )
        else:
            parts.append(
                "用@视频1作为目标机位下的点云几何和遮挡，跟随其相机运动；"
                "不要复制另一段故事片的运镜。"
            )
        if same_event_recapture:
            parts.append("这是同一历史事件的新机位重拍：逐帧遵循@视频1中的动作阶段、时间顺序和遮挡；保持原事件，不要续写新动作。静帧来自原事件指定起始时刻，只约束外观与身份，机位由@视频1决定。")
        elif has_first_frame:
            parts.append(
                "本镜开场静帧已按剧本重画：锁其光线、时辰和人物站位；"
                "相机只跟随@视频1，不要用静帧里的构图去改运镜。"
            )
    bindings: list[str] = []
    for item in references:
        if item.media_type != "image":
            continue
        if item.role is ReferenceRole.FIRST_FRAME:
            bindings.append(
                f"{item.prompt_handle}是原事件起始时刻的外观参考，不约束新机位构图"
                if same_event_recapture
                else f"{item.prompt_handle}是开场静帧：锁身份、服装、光线和场面"
            )
        elif item.role is ReferenceRole.CONTINUITY_MEMORY:
            bindings.append(
                f"{item.prompt_handle}是同一场地上一镜的最后一帧："
                "接服装、光线、场面和人物站位，不要换成别的房间"
            )
        elif item.role is ReferenceRole.CHARACTER:
            people = item.character_mapping or (item.label,)
            if len(people) > 1:
                numbered = "、".join(
                    f"编号{index}是{name}" for index, name in enumerate(people, start=1)
                )
                bindings.append(
                    f"{item.prompt_handle}里{numbered}，按编号锁身份，"
                    "不要把编号和人对调或换成别人"
                )
            else:
                who = people[0]
                bindings.append(f"{item.prompt_handle}只用于{who}的身份，不要换成别人")
    if bindings:
        parts.append("；".join(bindings) + "。")
    appearing = "、".join(shot.entity_ids) if shot.entity_ids else ""
    if appearing:
        parts.append(f"本镜主要拍摄：{'、'.join(scene_context(shot).get('focal_labels',shot.entity_ids))}。"
            "画面可包含同场景仍在场且被机位拍到的其他角色；未列为主体不代表角色离开。不要凭空添加陌生角色。")
    if skipped_familiar:
        names = "、".join(skipped_familiar)
        parts.append(f"{names}已出现在同一场地上一镜末帧中，沿用其身份和服装，不要另换一张脸。")
    return "\n".join(part for part in parts if part.strip())


def _character_references(
    stills: list[CharacterStill],
    *,
    start_ordinal: int,
    skip_path: Path | None,
    require_file: bool,
) -> tuple[list[MediaReference], list[CharacterStill]]:
    skip = skip_path.resolve() if skip_path is not None else None
    items: list[MediaReference] = []
    unused: list[CharacterStill] = []
    ordinal = start_ordinal
    for still in stills:
        if skip is not None and still.path.resolve() == skip:
            unused.append(
                CharacterStill(
                    still.name, still.path, "duplicate_of_scene_still", names=still.people
                )
            )
            continue
        if require_file and not still.path.is_file():
            unused.append(
                CharacterStill(still.name, still.path, "missing_file", names=still.people)
            )
            continue
        items.append(
            MediaReference(
                path=still.path,
                role=ReferenceRole.CHARACTER,
                ordinal=ordinal,
                media_type="image",
                label=f"角色身份参考 ({still.name})",
                asset_role="character_card",
                character_mapping=still.people,
            )
        )
        ordinal += 1
    return items, unused


def build_request(
    shot: PlanShot,
    *,
    output_dir: Path,
    geometry_video: Path | None,
    continuity_still: Path | None,
    first_frame: Path | None,
    previous_entity_ids: tuple[str, ...] = (),
    expect_geometry: bool = False,
    expect_continuity: bool = False,
    expect_first_frame: bool = False,
    require_files: bool = False,
    geometry_control_mode: str = "reference",
    same_event_recapture: bool = False,
) -> tuple[V2VRequest, list[CharacterStill], list[CharacterStill]]:
    if geometry_control_mode not in {"reference", "video_edit"}:
        raise ValueError(f"unknown geometry_control_mode={geometry_control_mode!r}")
    if geometry_control_mode == "video_edit" and geometry_video is None:
        raise ValueError("geometry_control_mode=video_edit requires geometry_video")
    if same_event_recapture and require_files:
        if geometry_video is None or not geometry_video.is_file() or first_frame is None or not first_frame.is_file():
            raise ValueError("recapture requires its geometry video and source-event reference still")
    task = V2VTask.REOBSERVE if same_event_recapture else V2VTask.CONTINUE
    shot=cast_shot(shot)
    spec = _shot_spec(shot, first_frame=first_frame)
    settings = AVWMSettings()
    cfg = settings.backends.wan27
    references: list[MediaReference] = []
    has_geometry = geometry_video is not None or expect_geometry
    has_continuity = continuity_still is not None or expect_continuity
    has_first_frame = first_frame is not None or expect_first_frame
    has_still = has_continuity or has_first_frame
    reserved = reserved_reference_slots(has_geometry=has_geometry, has_still=has_still)
    if has_geometry and geometry_video is not None and (
        not require_files or geometry_video.is_file()
    ):
        references.append(
            MediaReference(
                path=geometry_video,
                role=ReferenceRole.TARGET_GEOMETRY,
                ordinal=1,
                media_type="video",
                label="目标相机点云条件片",
                duration_seconds=shot.duration,
            )
        )
    still = continuity_still if has_continuity else first_frame if has_first_frame else None
    image_ordinal = 1
    if still is not None and (not require_files or still.is_file()):
        role = (
            ReferenceRole.CONTINUITY_MEMORY
            if has_continuity
            else ReferenceRole.FIRST_FRAME
        )
        references.append(
            MediaReference(
                path=still,
                role=role,
                ordinal=image_ordinal,
                media_type="image",
                label="开场静帧" if role is ReferenceRole.FIRST_FRAME else "同一场地上一镜末帧",
            )
        )
        image_ordinal += 1
    elif has_still:
        image_ordinal += 1
    chosen, skipped = select_character_stills(
        shot,
        previous_entity_ids=previous_entity_ids if has_continuity else (),
        has_continuity_still=has_continuity,
        reserved_slots=reserved,
        output_dir=output_dir,
    )
    character_refs, unused = _character_references(
        chosen,
        start_ordinal=image_ordinal,
        skip_path=still,
        require_file=require_files,
    )
    references.extend(character_refs)
    skipped = list(skipped) + unused
    placeholder = output_dir / "geometry_placeholder.mp4"
    report = output_dir / "condition_report.json"
    camera_query = output_dir / "camera_query.json"
    condition = V2VCondition(
        task=task,
        input_video=geometry_video or placeholder,
        camera_query=camera_query,
        report_path=report,
        geometry_source="dense_vista4d_l1_l2" if geometry_video else "bootstrap_image",
        reference_image=still,
        references=tuple(references),
    )
    familiar = tuple(
        item.name for item in skipped if item.reason == "skipped_familiar_last_frame"
    )
    request = V2VRequest(
        shot=spec,
        task=task,
        prompt=_prompt(
            shot,
            tuple(references),
            has_geometry=geometry_video is not None or expect_geometry,
            has_first_frame=has_first_frame,
            skipped_familiar=familiar,
            establishing_empty=not shot.entity_ids and not has_continuity,
            geometry_control_mode=geometry_control_mode,
            same_event_recapture=same_event_recapture,
        ),
        condition=condition,
        fps=cfg.fps,
        width=cfg.width,
        height=cfg.height,
        frame_count=max(2, round(shot.duration * cfg.fps)),
        output_dir=output_dir,
        mock_mode=False,
    )
    packed = [
        item
        for item in request.condition.references
        if item.media_type in {"video", "image"}
    ]
    if len(packed) > WAN30_REFERENCE_CAP:
        raise RuntimeError(
            f"{shot.shot_id} packed {len(packed)} Wan references; cap is {WAN30_REFERENCE_CAP}"
        )
    return request, chosen, skipped


async def generate_shot(
    request: V2VRequest,
    *,
    geometry_control_mode: str = "reference",
) -> Path:
    from atas.providers.wan.router import generate_routed
    artifact, _, _ = await generate_routed(
        request,
        geometry_control_mode=geometry_control_mode,
    )
    return artifact
