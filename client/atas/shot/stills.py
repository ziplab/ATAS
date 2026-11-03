"""Opening stills generated from the AI-authored first-frame prompt."""
from __future__ import annotations

from pathlib import Path


def _authored(shot, output: Path, *, required: tuple[Path, ...] = ()) -> Path:
    from atas.shot.prompt_author import generate_authored_still
    for path in required:
        if not Path(path).is_file() or Path(path).stat().st_size <= 0:
            raise RuntimeError(f"still reference missing for {shot.story_id}/{shot.shot_id}: {path}")
    result = generate_authored_still(shot, output)
    if result is None:
        raise RuntimeError(f"{shot.story_id}/{shot.shot_id}: first-frame prompt was not authored")
    return result


def generate_cold_start_frame(shot, output: Path) -> Path:
    """First observation of a place: a new opening image from identity cards."""
    return _authored(shot, output)


def generate_anchor_still(shot, *, anchor: Path, output: Path) -> Path:
    """New view of a remembered place, anchored on its RGB home frame."""
    return _authored(shot, output, required=(anchor,))


def generate_geometry_aligned_still(shot, *, scene_reference: Path, composition_guide: Path,
                                    output: Path) -> Path:
    """Opening image at the exact composition of the rendered scene memory."""
    return _authored(shot, output, required=(scene_reference, composition_guide))


def write_composition_guide(video: Path, report: dict, output: Path) -> Path:
    """Annotate GPU-projected subject regions on the actual target first frame."""
    import cv2
    import numpy as np
    from PIL import Image,ImageDraw
    capture=cv2.VideoCapture(str(video));ok,frame=capture.read();capture.release()
    if not ok: raise ValueError("cannot decode target-camera first frame")
    regions=report.get("first_frame_subject_regions") or []
    canvas=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(canvas)
    for row in regions:
        try:box=np.asarray(row.get("bbox"),float)
        except (ValueError,TypeError):continue
        if box.shape!=(4,) or not np.isfinite(box).all() or (box<0).any() or (box>1).any():
            continue
        x0,y0,x1,y1=box*[canvas.width,canvas.height,canvas.width,canvas.height]
        if x1<=x0 or y1<=y0: continue
        draw.rectangle((x0,y0,x1,y1),outline=(0,255,0),width=3)
        draw.text((x0,max(0,y0-16)),str(row.get("label","subject"))+" / "+str(row.get("part","subject")),fill=(0,255,0))
    for actor in report.get('scene_actor_regions',[]):
        sample=next((s for s in actor.get('samples',[]) if s.get('frame_index')==0),{})
        box=sample.get('bbox')
        if not box:continue
        try:box=np.asarray(box,float)
        except (ValueError,TypeError):continue
        if box.shape!=(4,) or not np.isfinite(box).all():continue
        x0,y0,x1,y1=np.clip(box,0,1)*[canvas.width,canvas.height,canvas.width,canvas.height]
        if x1<=x0 or y1<=y0:continue
        draw.rectangle((x0,y0,x1,y1),outline=(0,255,255),width=2)
        draw.text((x0,max(0,y0-16)),actor['label']+' / last observed',fill=(0,255,255))
    temporary=output.with_name(output.stem+".partial.png");canvas.save(temporary);temporary.replace(output)
    return output
