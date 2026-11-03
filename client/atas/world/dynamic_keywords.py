"""Resolve SAM3 short nouns on the client; the geometry service only receives explicit keywords."""

from __future__ import annotations

import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from atas.video_io import read_video_frames
VISION_PROMPT = """Look at these video keyframes. They are stills, not a motion meter.
List short English nouns SAM3 should treat as DYNAMIC: people, animals, and props that
are held, carried, worn, or used (a basket in someone's hand, a lamp being swapped).
Do not list walls, floor, sky, large furniture, or set dressing that stays put.
When unsure whether a visible object is held or is set dressing, INCLUDE it.
Missing a moving prop fuses it into the static room forever. Extra nouns only make holes.
Return JSON only: {"nouns": ["rabbit", "basket"]}
Each noun must be 1-2 English words. No character names, no story ids, no sentences."""

FILTER_PROMPT = """You map screenplay onto SAM3 segmentation keywords.

Vision nouns (from still keyframes — they cannot prove motion):
{candidates}

Screenplay / onstage / time-varying context:
{context}

Output short English SAM nouns for moving people, animals, AND handled/carried props
mentioned or implied (basket, lamp, apple, dossier). Map Chinese or character labels
to class nouns (小棕兔→rabbit, 人→person or woman, 卷宗→dossier). Never output the
character's proper name (not "Little Brown Rabbit").
Drop only obvious static set dressing (walls, shelves, table, sky).
When unsure, KEEP the noun. Prefer too many dynamic classes over staining L1.
Never output story ids or sentences.
Return JSON only: {{"nouns": ["rabbit", "basket"]}}
Only return [] if the shot has no onstage being and no handled prop."""


class DynamicKeywordError(RuntimeError):
    """Auto keyword resolution failed; do not submit an empty SAM list."""


def parse_noun_list(text: str) -> list[str]:
    payload = text.strip()
    match = re.search(r"\{.*\}", payload, flags=re.DOTALL)
    if match:
        payload = match.group(0)
        data = json.loads(payload)
        raw = data.get("nouns", data.get("keywords", data.get("dynamic_objects")))
    else:
        start = payload.find("[")
        end = payload.rfind("]")
        if start < 0 or end <= start:
            raise DynamicKeywordError(f"keyword model did not return a JSON list: {text[:200]}")
        raw = json.loads(payload[start : end + 1])
    if not isinstance(raw, list):
        raise DynamicKeywordError("keyword JSON nouns must be a list")
    return [str(item) for item in raw]


def sanitize_sam_nouns(nouns: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in nouns:
        noun = re.sub(r"\s+", " ", str(raw).strip().lower())
        noun = noun.strip(".,;:\"'")
        if not noun or "_" in noun or len(noun.split()) > 2:
            continue
        if any(ch.isdigit() for ch in noun):
            continue
        if noun in {"auto", "none", "n/a", "static"}:
            continue
        if noun not in seen:
            seen.add(noun)
            cleaned.append(noun)
    return cleaned[:8]


def prefer_dynamic_union(vision: list[str], filtered: list[str]) -> list[str]:
    """Keep vision nouns even if the text filter dropped them.

    Keyframes cannot measure motion. Dropping a held basket fuses it into L1.
    Extra SAM classes become grey holes, which later visits can fill.
    """
    return sanitize_sam_nouns([*vision, *filtered])


def resolve_dynamic_keywords(
    video: Path,
    *,
    screenplay_context: str,
    output_dir: Path | None = None,
    frame_count: int = 4,
    allow_empty: bool = False,
) -> list[str]:
    if output_dir is not None:
        existing = output_dir / "dynamic_keywords.json"
        if existing.is_file():
            try:
                payload = json.loads(existing.read_text(encoding="utf-8"))
                reused = sanitize_sam_nouns(list(payload.get("resolved") or []))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                reused = []
            if reused:
                print(f"SAM keywords={reused} (reused from {existing})", flush=True)
                return reused
    frames = _sample_keyframes(video, frame_count)
    vision_raw = _chat(
        VISION_PROMPT,
        images=frames,
        temperature=0.0,
    )
    candidates = sanitize_sam_nouns(parse_noun_list(vision_raw))
    filter_raw = _chat(
        FILTER_PROMPT.format(
            candidates=json.dumps(candidates, ensure_ascii=False),
            context=(screenplay_context or "(none)").strip()[:1200],
        ),
        images=(),
        temperature=0.0,
    )
    filtered = sanitize_sam_nouns(parse_noun_list(filter_raw))
    keywords = prefer_dynamic_union(candidates, filtered)
    if "book" in keywords and "page" in keywords:
        keywords = [item for item in keywords if item != "page"]
    source = "gpt_vision_union_filter_prefer_dynamic"
    report = {
        "schema_version": 1,
        "video": str(video),
        "vision_raw": vision_raw[:4000],
        "candidates": candidates,
        "filter_raw": filter_raw[:4000],
        "filtered": filtered,
        "resolved": keywords,
        "dynamic_source": source,
        "policy": "prefer_dynamic_union",
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "dynamic_keywords.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if not keywords and not allow_empty:
        raise DynamicKeywordError(
            "auto dynamic keywords were empty after GPT vision+filter; "
            "refusing silent all-static SAM. Onstage shots must name a SAM class."
        )
    print(f"SAM keywords={keywords} (resolved on the client)", flush=True)
    return keywords


def _sample_keyframes(video: Path, count: int) -> list[np.ndarray]:
    frames = read_video_frames(video)
    if not frames:
        raise DynamicKeywordError(f"no frames in {video}")
    if len(frames) <= count:
        return [frame[..., :3] for frame in frames]
    indices = np.linspace(0, len(frames) - 1, count, dtype=int)
    return [frames[int(index)][..., :3] for index in indices]


def _jpeg_data_url(frame: np.ndarray) -> str:
    image = Image.fromarray(frame[..., :3].astype(np.uint8))
    image.thumbnail((768, 768))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _chat(prompt: str, *, images: list[np.ndarray], temperature: float) -> str:
    del temperature
    api_key = (
        os.getenv("AVWM_IMAGE_API_KEY", "").strip()
        or os.getenv("ATAS_LLM_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )
    if not api_key:
        raise DynamicKeywordError(
            "set AVWM_IMAGE_API_KEY, ATAS_LLM_API_KEY, or OPENAI_API_KEY for keyword tagging"
        )
    responses_url = os.getenv("ATAS_LLM_URL", "").strip().rstrip("/")
    if responses_url:
        url = responses_url if responses_url.endswith("/responses") else responses_url + "/responses"
        model = (
            os.getenv("AVWM_DYNAMIC_KEYWORD_MODEL", "").strip()
            or os.getenv("ATAS_LLM_MODEL", "").strip()
            or "gpt-4.1-mini"
        )
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for frame in images:
            content.append({"type": "input_image", "image_url": _jpeg_data_url(frame)})
        payload: dict[str, Any] = {
            "model": model,
            "input": [{"role": "user", "content": content}],
            "stream": False,
        }
        tier = os.getenv("ATAS_LLM_SERVICE_TIER", "").strip()
        if tier:
            payload["service_tier"] = tier
        body = _post_json(url, payload, api_key)
        text = body.get("output_text")
        if not isinstance(text, str) or not text.strip():
            parts: list[str] = []
            for item in body.get("output", []) or []:
                if not isinstance(item, dict):
                    continue
                for block in item.get("content", []) or []:
                    if not isinstance(block, dict):
                        continue
                    value = block.get("text") or block.get("output_text")
                    if isinstance(value, dict):
                        value = value.get("text") or value.get("value")
                    if isinstance(value, str):
                        parts.append(value)
            text = "".join(parts)
        if not isinstance(text, str) or not text.strip():
            raise DynamicKeywordError("keyword model returned empty content")
        return text
    base = (
        os.getenv("AVWM_IMAGE_BASE_URL", "").strip()
        or os.getenv("OPENAI_BASE_URL", "").strip()
        or os.getenv("OPENAI_API_BASE", "").strip()
        or "https://api.openai.com/v1"
    ).rstrip("/")
    url = base if base.endswith("/chat/completions") else (
        base + "/chat/completions" if base.endswith("/v1") else base + "/v1/chat/completions"
    )
    model = os.getenv("AVWM_DYNAMIC_KEYWORD_MODEL", "").strip() or "gpt-4.1-mini"
    chat_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for frame in images:
        chat_content.append({"type": "image_url", "image_url": {"url": _jpeg_data_url(frame)}})
    body = _post_json(url, {"model": model, "messages": [{"role": "user", "content": chat_content}]}, api_key)
    try:
        text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as extra:
        raise DynamicKeywordError("keyword model response missing choices[0].message.content") from extra
    if not isinstance(text, str) or not text.strip():
        raise DynamicKeywordError("keyword model returned empty content")
    return text


def _post_json(url: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as extra:
        # Preserve the provider's error category, never credentials or payloads.
        try:
            error = json.loads(extra.read(16384)).get("error", {})
            message = str(error.get("message", "")) if isinstance(error, dict) else ""
        except (ValueError, AttributeError):
            message = ""
        message = message.replace(api_key, "[REDACTED]") if api_key else message
        message = re.sub(r"sk[-_][A-Za-z0-9_-]+", "[REDACTED]", message)[:600]
        failure = DynamicKeywordError(f"model HTTP {extra.code}: {message}")
        failure.http_status = extra.code
        failure.service_unavailable = (extra.code in {429, 500, 502, 503, 504}
            or (extra.code == 404 and "could not find an existing deployment" in message.lower()))
        raise failure from extra
    if not isinstance(parsed, dict):
        raise DynamicKeywordError("keyword model returned a non-object JSON body")
    return parsed
