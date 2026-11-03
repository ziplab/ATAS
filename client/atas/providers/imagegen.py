"""gpt-image generation for identity references and opening stills.

Two transports reach the same image model:

  images_api      POST /images/generations|edits with model gpt-image-2
  responses_tool  POST /responses with the built-in ``image_generation`` tool; reference
                  images are passed as input images (gateways that expose GPT models
                  but no standalone image endpoint)

``ATAS_IMAGE_BACKEND`` selects one; the default ``auto`` tries the images API and
switches to the Responses tool for the rest of the process when the gateway reports
that the image model is unavailable.
"""
from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path
from typing import Sequence

_BACKEND: str | None = None


def _backend() -> str:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = os.getenv("ATAS_IMAGE_BACKEND", "auto").strip().lower()
        if _BACKEND not in {"auto", "images_api", "responses_tool"}:
            raise ValueError("ATAS_IMAGE_BACKEND must be auto, images_api or responses_tool")
    return _BACKEND


def _model_unavailable(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "model_not_found" in text or ("404" in text and "not supported" in text)


def _image_input(path: Path) -> dict:
    from PIL import Image
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((1536, 1536))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    return {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()}


def _responses_tool(prompt: str, output: Path, references: Sequence[Path]) -> None:
    from atas.providers.llm import _api_key, _post, _post_stream, model_name, transient
    url = os.getenv("ATAS_LLM_URL", "").strip().rstrip("/")
    if not url:
        raise RuntimeError("the responses_tool image backend needs ATAS_LLM_URL")
    url = url if url.endswith("/responses") else url + "/responses"
    content = [{"type": "input_text", "text": prompt}] + [_image_input(Path(p)) for p in references]
    tool = {"type": "image_generation", "size": os.getenv("AVWM_IMAGE_SIZE", "1536x1024").replace("*", "x"),
            "quality": os.getenv("AVWM_IMAGE_QUALITY", "high")}
    payload = {"model": model_name(), "input": [{"role": "user", "content": content}], "tools": [tool],
               "tool_choice": {"type": "image_generation"}, "stream": False}
    attempts = max(1, int(os.getenv("AVWM_IMAGE_RETRIES", "4")))
    for attempt in range(attempts):
        try:
            if os.getenv("ATAS_LLM_STREAM", "1") != "0":
                _, body = _post_stream(url, payload, _api_key(), timeout=600)
            else:
                body = _post(url, payload, _api_key(), timeout=600)
            for item in body.get("output", []) or []:
                if item.get("type") == "image_generation_call" and item.get("result"):
                    output.write_bytes(base64.b64decode(item["result"]))
                    return
            raise RuntimeError("image_generation tool returned no image: "
                               + str([i.get("type") for i in body.get("output", [])])[:300])
        except Exception as exc:  # noqa: BLE001 - transient errors retried below
            if attempt == attempts - 1 or not (transient(exc) or "returned no image" in str(exc)):
                raise
            time.sleep(min(60.0, 5.0 * 2 ** attempt))


def _aspect_ok(path: Path, aspect: float, tolerance: float = 0.06) -> bool:
    from PIL import Image
    with Image.open(path) as image:
        width, height = image.size
    return abs(width / height - aspect) / aspect <= tolerance


def _crop_landscape(path: Path, aspect: float) -> bool:
    """Centre-crop a landscape image to ``aspect``; False when it is not landscape."""
    from PIL import Image
    with Image.open(path) as image:
        width, height = image.size
        if width <= height:
            return False
        if width / height > aspect:
            new_w, new_h = round(height * aspect), height
        else:
            new_w, new_h = width, round(width / aspect)
        left, top = (width - new_w) // 2, (height - new_h) // 2
        cropped = image.crop((left, top, left + new_w, top + new_h))
        cropped.load()
    cropped.save(path)
    return True


def generate(prompt: str, output: Path, *, references: Sequence[Path] = (), negative_prompt: str = "",
             aspect: float | None = None) -> Path:
    """Render one image to ``output`` (PNG), conditioned on optional reference images.

    ``aspect`` (width/height) is enforced. A landscape image of another shape (image
    APIs offer only a few sizes, e.g. 3:2) is centre-cropped; a portrait or square
    image is regenerated with an explicit framing instruction.
    """
    if aspect is None:
        return _generate(prompt, output, references=references, negative_prompt=negative_prompt)
    shape = "landscape 16:9" if abs(aspect - 16 / 9) < 0.01 else f"{aspect:.2f}:1 aspect"
    instruction = f"\nOutput a single {shape} wide cinematic frame (width greater than height), not a portrait image."
    for attempt in range(3):
        _generate(prompt + instruction * (attempt + 1), output, references=references, negative_prompt=negative_prompt)
        if _aspect_ok(output, aspect) or _crop_landscape(output, aspect):
            return output
        print(f"gpt-image: wrong aspect for {output.name}, regenerating ({attempt + 1}/3)", flush=True)
    raise RuntimeError(f"image generation kept returning the wrong aspect for {output.name}")


def _generate(prompt: str, output: Path, *, references: Sequence[Path] = (), negative_prompt: str = "") -> Path:
    global _BACKEND
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if negative_prompt:
        prompt += f"\nAvoid: {negative_prompt}"
    backend = _backend()
    if backend in {"auto", "images_api"}:
        from atas.providers.gpt_image import GPTImage2Generator
        from atas.providers.settings import AVWMSettings
        generator = GPTImage2Generator(AVWMSettings())
        if backend == "auto":
            generator.retries = 1   # one probe is enough to learn whether the endpoint exists
        try:
            generator.generate(prompt, output, references=tuple(references))
            return output
        except Exception as exc:  # noqa: BLE001
            if backend == "images_api":
                raise
            if not _model_unavailable(exc):
                # A transient failure on the probe: retry through the images API, unless
                # the retry is what reveals that the model is not served here.
                try:
                    GPTImage2Generator(AVWMSettings()).generate(prompt, output, references=tuple(references))
                    _BACKEND = "images_api"
                    return output
                except Exception as retry_exc:  # noqa: BLE001
                    if not _model_unavailable(retry_exc):
                        raise
            print("gpt-image: images API unavailable on this gateway; using the Responses image tool", flush=True)
            _BACKEND = "responses_tool"
    _responses_tool(prompt, output, references)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("image generation wrote an empty file")
    return output
