"""One LLM/VLM transport for planning, camera grounding, prompt authoring and review.

Both an OpenAI-compatible Responses endpoint (``ATAS_LLM_URL``) and a
chat-completions endpoint are supported. Transient provider failures (rate
limits, 5xx, gateway HTML pages, timeouts, dropped connections) are retried
with backoff; content and schema errors are returned to the caller.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

TRANSIENT_HTTP = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 530}


class LLMError(RuntimeError):
    """A definite model/provider error that retrying the same request will not fix."""


class LLMUnavailable(ConnectionError):
    """The provider stayed unavailable through the whole retry budget."""
    service_unavailable = True


@dataclass(frozen=True)
class Message:
    role: str            # system | user | assistant
    content: str
    images: tuple = ()   # numpy RGB arrays, user messages only


def model_name() -> str:
    return (os.getenv("AVWM_DYNAMIC_KEYWORD_MODEL", "").strip()
            or os.getenv("ATAS_LLM_MODEL", "").strip() or "gpt-4.1-mini")


def _api_key() -> str:
    key = (os.getenv("AVWM_IMAGE_API_KEY", "").strip() or os.getenv("ATAS_LLM_API_KEY", "").strip()
           or os.getenv("OPENAI_API_KEY", "").strip())
    if not key:
        raise LLMError("set ATAS_LLM_API_KEY (or OPENAI_API_KEY) for the language model")
    return key


def _jpeg_data_url(frame: np.ndarray) -> str:
    from PIL import Image
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(frame)[..., :3].astype(np.uint8)).save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _redact(text: str, key: str) -> str:
    text = text.replace(key, "[REDACTED]") if key else text
    return re.sub(r"sk[-_][A-Za-z0-9_-]+", "[REDACTED]", text)


def _post(url: str, payload: dict[str, Any], key: str, timeout: float) -> dict[str, Any]:
    # Some gateways (Cloudflare) reject urllib's default User-Agent with error 1010.
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST",
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                              "User-Agent": os.getenv("ATAS_HTTP_USER_AGENT", "atas-pipeline/1.0")})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = _redact(exc.read(16384).decode("utf-8", errors="replace"), key)[:600]
        error = (LLMUnavailable if exc.code in TRANSIENT_HTTP else LLMError)(f"model HTTP {exc.code}: {detail}")
        error.http_status = exc.code
        raise error from exc
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        # A gateway HTML page or truncated body is an availability problem.
        raise LLMUnavailable(f"model returned a non-JSON body: {raw[:200]!r}") from exc
    if not isinstance(body, dict):
        raise LLMError("model returned a non-object JSON body")
    return body


def _post_stream(url: str, payload: dict[str, Any], key: str, timeout: float) -> tuple[str, dict[str, Any]]:
    """Responses API with server-sent events. Tokens keep the connection busy, so
    long generations are not cut by proxies with an idle/first-byte limit
    (Cloudflare 524 after 100 s)."""
    request = urllib.request.Request(url, data=json.dumps({**payload, "stream": True}).encode("utf-8"), method="POST",
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                              "Accept": "text/event-stream",
                                              "User-Agent": os.getenv("ATAS_HTTP_USER_AGENT", "atas-pipeline/1.0")})
    parts: list[str] = []
    completed: dict[str, Any] | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                kind = event.get("type", "")
                if kind == "response.output_text.delta":
                    parts.append(event.get("delta", ""))
                elif kind == "response.completed":
                    completed = event.get("response") or {}
                elif kind in {"response.failed", "error", "response.error"}:
                    detail = json.dumps(event.get("response", event))[:600]
                    raise LLMUnavailable(f"model stream failed: {_redact(detail, key)}")
    except urllib.error.HTTPError as exc:
        detail = _redact(exc.read(16384).decode("utf-8", errors="replace"), key)[:600]
        error = (LLMUnavailable if exc.code in TRANSIENT_HTTP else LLMError)(f"model HTTP {exc.code}: {detail}")
        error.http_status = exc.code
        raise error from exc
    completed = completed or {}
    text = "".join(parts)
    if not text.strip() and completed:
        text = _responses_text(completed)
    return text, completed


def _responses_payload(messages: Sequence[Message]) -> list[dict[str, Any]]:
    rows = []
    for message in messages:
        kind = "output_text" if message.role == "assistant" else "input_text"
        content = [{"type": kind, "text": message.content}]
        content += [{"type": "input_image", "image_url": _jpeg_data_url(f)} for f in message.images]
        rows.append({"role": message.role, "content": content})
    return rows


def _responses_text(body: dict[str, Any]) -> str:
    text = body.get("output_text")
    if isinstance(text, str) and text.strip():
        return text
    parts = []
    for item in body.get("output", []) or []:
        for block in (item.get("content", []) if isinstance(item, dict) else []) or []:
            value = block.get("text") if isinstance(block, dict) else None
            if isinstance(value, dict):
                value = value.get("text") or value.get("value")
            if isinstance(value, str):
                parts.append(value)
    return "".join(parts)


def _chat_once(messages: Sequence[Message], *, timeout: float) -> str:
    key = _api_key()
    responses = os.getenv("ATAS_LLM_URL", "").strip().rstrip("/")
    if responses:
        url = responses if responses.endswith("/responses") else responses + "/responses"
        payload: dict[str, Any] = {"model": model_name(), "input": _responses_payload(messages), "stream": False}
        tier = os.getenv("ATAS_LLM_SERVICE_TIER", "").strip()
        if tier:
            payload["service_tier"] = tier
        if os.getenv("ATAS_LLM_STREAM", "1") != "0":
            text, _ = _post_stream(url, payload, key, timeout)
        else:
            text = _responses_text(_post(url, payload, key, timeout))
    else:
        base = (os.getenv("AVWM_IMAGE_BASE_URL", "").strip() or os.getenv("OPENAI_BASE_URL", "").strip()
                or "https://api.openai.com/v1").rstrip("/")
        url = base if base.endswith("/chat/completions") else (
            base + "/chat/completions" if base.endswith("/v1") else base + "/v1/chat/completions")
        rows = []
        for message in messages:
            if message.images:
                content: Any = [{"type": "text", "text": message.content}] + [
                    {"type": "image_url", "image_url": {"url": _jpeg_data_url(f)}} for f in message.images]
            else:
                content = message.content
            rows.append({"role": message.role, "content": content})
        body = _post(url, {"model": model_name(), "messages": rows}, key, timeout)
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("model response missing choices[0].message.content") from exc
    if not isinstance(text, str) or not text.strip():
        raise LLMUnavailable("model returned empty content")
    return text


def transient(exc: BaseException) -> bool:
    if isinstance(exc, LLMUnavailable) or getattr(exc, "service_unavailable", False):
        return True
    return isinstance(exc, (TimeoutError, ConnectionError, urllib.error.URLError))


def chat(messages: Sequence[Message] | str, *, images: Sequence[np.ndarray] = (),
         attempts: int | None = None, timeout: float = 240.0) -> str:
    """Send a conversation (or one user prompt with images) and return the reply text."""
    if isinstance(messages, str):
        messages = [Message("user", messages, tuple(images))]
    # About 15 minutes of backoff rides out short gateway outages.
    attempts = attempts or int(os.getenv("ATAS_LLM_ATTEMPTS", "10"))
    for attempt in range(attempts):
        try:
            return _chat_once(messages, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - classified below
            if not transient(exc):
                raise
            if attempt == attempts - 1:
                raise LLMUnavailable(f"model unavailable after {attempts} attempts: {exc}") from exc
            time.sleep(min(120.0, 4.0 * 2 ** attempt))
    raise AssertionError("unreachable")


def strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()
