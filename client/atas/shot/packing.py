from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from atas.request import PlanShot
# Must stay in sync with atas.providers.wan.client.MAX_R2V_REFERENCES.
WAN30_REFERENCE_CAP = 5
_LABEL_FONT = Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")
_BADGE_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


@dataclass(frozen=True)
class CharacterStill:
    name: str
    path: Path
    reason: str
    names: tuple[str, ...] = ()

    @property
    def people(self) -> tuple[str, ...]:
        return self.names if self.names else (self.name,)


def appearing_characters(shot: PlanShot) -> list[tuple[str, Path]]:
    """YAML Characters Appearing, aligned with this shot's stills."""

    from atas.world.identity import appearance_pairs
    return appearance_pairs(shot)


def reserved_reference_slots(*, has_geometry: bool, has_still: bool) -> int:
    return int(has_geometry) + int(has_still)


def partition_into_slots(items: list, slots: int) -> list[list]:
    """Fit M people into N slots; leftover people share the last sheets."""

    if slots <= 0:
        return []
    if len(items) <= slots:
        return [[item] for item in items]
    base, extra = divmod(len(items), slots)
    groups: list[list] = []
    index = 0
    for slot in range(slots):
        size = base + (1 if slot >= slots - extra else 0)
        groups.append(items[index : index + size])
        index += size
    return groups


def compose_numbered_sheet(
    people: list[tuple[str, Path]],
    output: Path,
    *,
    cell_size: int = 512,
) -> Path:
    """Side-by-side / grid collage with a painted index on each person."""

    count = len(people)
    columns = count if count <= 2 else math.ceil(math.sqrt(count))
    rows = math.ceil(count / columns)
    banner = max(48, cell_size // 8)
    width = columns * cell_size
    height = rows * (cell_size + banner)
    canvas = Image.new("RGB", (width, height), (16, 16, 16))
    draw = ImageDraw.Draw(canvas)
    name_font = _load_font(_LABEL_FONT, max(22, cell_size // 14))
    badge_font = _load_font(_BADGE_FONT, max(36, cell_size // 8))
    for index, (name, path) in enumerate(people):
        row, col = divmod(index, columns)
        x = col * cell_size
        y = row * (cell_size + banner)
        portrait = _fit_cell(_open_rgb(path), cell_size)
        canvas.paste(portrait, (x, y))
        bar_top = y + cell_size
        draw.rectangle((x, bar_top, x + cell_size, bar_top + banner), fill=(20, 20, 20))
        label = f"{index + 1}. {name}"
        draw.text(
            (x + 12, bar_top + max(8, (banner - 28) // 2)),
            label,
            fill=(255, 255, 255),
            font=name_font,
        )
        badge = max(48, cell_size // 8)
        inset = 12
        draw.ellipse(
            (x + inset, y + inset, x + inset + badge, y + inset + badge),
            fill=(220, 40, 40),
            outline=(255, 255, 255),
            width=3,
        )
        digit = str(index + 1)
        bbox = draw.textbbox((0, 0), digit, font=badge_font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            (
                x + inset + (badge - tw) / 2 - bbox[0],
                y + inset + (badge - th) / 2 - bbox[1],
            ),
            digit,
            fill=(255, 255, 255),
            font=badge_font,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def select_character_stills(
    shot: PlanShot,
    *,
    previous_entity_ids: tuple[str, ...],
    has_continuity_still: bool,
    reserved_slots: int,
    output_dir: Path | None = None,
) -> tuple[list[CharacterStill], list[CharacterStill]]:
    """Fill leftover Wan 3.0 slots after packing C (PC video + one still).

    Newcomers always take slots first. Familiar faces fill whatever is left so
    identity stays on the design sheet. Only skip a familiar face when the cap
    is already full — then the continuity last-frame carries them.
    """

    remaining = max(0, WAN30_REFERENCE_CAP - reserved_slots)
    appearing = appearing_characters(shot)
    previous = {name.strip() for name in previous_entity_ids if str(name).strip()}
    newcomers = [(name, path) for name, path in appearing if name not in previous]
    familiar = [(name, path) for name, path in appearing if name in previous]
    skipped: list[CharacterStill] = []
    chosen: list[CharacterStill] = []

    if remaining <= 0:
        skipped.extend(
            CharacterStill(name, path, "dropped_over_wan30_cap", names=(name,))
            for name, path in appearing
        )
        return [], skipped

    if not has_continuity_still:
        leftover = _take_slots(
            [(name, path, "appearing_no_last_frame") for name, path in appearing],
            remaining,
            chosen,
            skipped,
            output_dir,
        )
        return chosen, skipped

    leftover = _take_slots(
        [(name, path, "new_to_last_frame") for name, path in newcomers],
        remaining,
        chosen,
        skipped,
        output_dir,
    )
    leftover = _take_slots(
        [(name, path, "familiar_slot_available") for name, path in familiar],
        leftover,
        chosen,
        skipped,
        output_dir,
        skip_reason="skipped_familiar_last_frame",
    )
    return chosen, skipped


def _take_slots(
    queue: list[tuple[str, Path, str]],
    slots: int,
    chosen: list[CharacterStill],
    skipped: list[CharacterStill],
    output_dir: Path | None,
    *,
    skip_reason: str = "dropped_over_wan30_cap",
) -> int:
    if not queue:
        return slots
    if slots <= 0:
        skipped.extend(
            CharacterStill(name, path, skip_reason, names=(name,))
            for name, path, _reason in queue
        )
        return 0
    start = len(chosen)
    for index, group in enumerate(partition_into_slots(queue, slots), start=1):
        if len(group) == 1:
            name, path, reason = group[0]
            chosen.append(CharacterStill(name, path, reason, names=(name,)))
            continue
        names = tuple(item[0] for item in group)
        label = " ".join(f"{order}.{name}" for order, name in enumerate(names, start=1))
        merged = (output_dir or Path(".")) / f"character_sheet_{start + index:02d}.png"
        sources = [(name, path) for name, path, _reason in group if path.is_file()]
        if len(sources) == len(group):
            compose_numbered_sheet(sources, merged)
        chosen.append(
            CharacterStill(
                label,
                merged,
                "merged_over_wan30_cap",
                names=names,
            )
        )
    return max(0, slots - (len(chosen) - start))


def _load_font(path: Path, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if path.is_file():
        try:
            return ImageFont.truetype(str(path), size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _open_rgb(path: Path) -> Image.Image:
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def _fit_cell(image: Image.Image, size: int) -> Image.Image:
    fitted = Image.new("RGB", (size, size), (32, 32, 32))
    scale = min(size / max(image.width, 1), size / max(image.height, 1))
    resized = image.resize(
        (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    fitted.paste(
        resized,
        ((size - resized.width) // 2, (size - resized.height) // 2),
    )
    return fitted
