from __future__ import annotations

import asyncio
import base64
import hashlib
import http.client
import json
import math
import mimetypes
import os
import re
import secrets
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol, cast

import imageio.v3 as iio

from atas.providers.wan.result import V2VProviderResult
from atas.providers.settings import Wan27Config
from atas.providers.wan.types import MediaReference, ReferenceRole, V2VRequest, V2VTask
class Wan27Transport(Protocol):
    def request_json(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any] | None,
        timeout: float,
    ) -> dict[str, Any]: ...

    def download(self, url: str, destination: Path, *, timeout: float) -> None: ...


class Wan27TempMediaStager(Protocol):
    def stage(
        self,
        path: Path,
        *,
        api_key: str,
        base_url: str,
        upload_path: str,
        model: str,
        timeout: float,
    ) -> str: ...


class UrllibWan27Transport:
    def request_json(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any] | None,
        timeout: float,
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:4000]
            raise RuntimeError(f"Wan2.7 HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Wan2.7 network request failed: {exc.reason}") from exc
        if not isinstance(decoded, dict):
            raise RuntimeError("Wan2.7 returned a non-object JSON response")
        return cast(dict[str, Any], decoded)

    def download(self, url: str, destination: Path, *, timeout: float) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "atas/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                with temporary.open("wb") as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            raise RuntimeError(f"Cannot download Wan2.7 result: {exc}") from exc
        temporary.replace(destination)


_WAN_IDENTITY_POLICY_CODES = frozenset({"FaceDetectionSuspect", "IPInfringementSuspect"})


def _wan_identity_policy_code(detail: object) -> str | None:
    if isinstance(detail, dict):
        output = detail.get("output")
        nested = output if isinstance(output, dict) else detail
        code = nested.get("code") if isinstance(nested, dict) else None
        if isinstance(code, str) and code in _WAN_IDENTITY_POLICY_CODES:
            return code
        detail = json.dumps(detail, ensure_ascii=False)
    text = str(detail)
    for code in _WAN_IDENTITY_POLICY_CODES:
        if code in text:
            return code
    return None


def _without_character_stills(
    references: tuple[MediaReference, ...],
) -> tuple[MediaReference, ...]:
    kept = [item for item in references if item.role is not ReferenceRole.CHARACTER]
    return tuple(
        replace(item, ordinal=index)
        for index, item in enumerate(kept, start=1)
    )


def _r2v_reference_priority(role: ReferenceRole) -> int:
    priorities = {
        ReferenceRole.CONTINUITY_MEMORY: 0,
        ReferenceRole.EVENT_MASTER: 0,
        ReferenceRole.TARGET_GEOMETRY: 1,
        ReferenceRole.CHARACTER: 2,
        ReferenceRole.PROP: 3,
        ReferenceRole.SCENE_STYLE: 4,
        ReferenceRole.GLOBAL_STATE: 5,
        ReferenceRole.USER_VIDEO: 5,
        ReferenceRole.GLOBAL_CROP: 6,
    }
    return priorities.get(role, 7)


def _is_transient_wan_error(exc: Exception) -> bool:
    """Recognize proxy/network failures that are safe to retry during polling."""

    if isinstance(
        exc,
        (
            ConnectionError,
            TimeoutError,
            socket.timeout,
            http.client.RemoteDisconnected,
        ),
    ):
        return True
    if isinstance(exc, urllib.error.URLError):
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "http 408",
            "http 425",
            "http 429",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
            "connection reset",
            "remote end closed",
            "read operation timed out",
            "timed out",
        )
    )


class DashScopeTempOSSStager:
    """Upload local media to DashScope temporary OSS and return an oss:// URL."""

    def stage(
        self,
        path: Path,
        *,
        api_key: str,
        base_url: str,
        upload_path: str,
        model: str,
        timeout: float,
    ) -> str:
        policy = self._get_policy(
            api_key=api_key,
            base_url=base_url,
            upload_path=upload_path,
            model=model,
            timeout=timeout,
        )
        file_size = path.stat().st_size
        max_mb = _float_value(policy.get("max_file_size_mb"), 0.0)
        if max_mb > 0 and file_size > int(max_mb * 1024 * 1024):
            raise RuntimeError(
                f"DashScope temporary upload rejected {path.name!r}: "
                f"{file_size} bytes exceeds policy max_file_size_mb={max_mb:g}"
            )
        upload_dir = _required_string(policy, "upload_dir").strip("/")
        key = f"{upload_dir}/{path.name}"
        upload_host = _required_string(policy, "upload_host")
        self._post_multipart(
            url=upload_host,
            fields={
                "OSSAccessKeyId": _required_string(policy, "oss_access_key_id"),
                "Signature": _required_string(policy, "signature"),
                "policy": _required_string(policy, "policy"),
                "x-oss-object-acl": _required_string(policy, "x_oss_object_acl"),
                "x-oss-forbid-overwrite": _required_string(
                    policy, "x_oss_forbid_overwrite"
                ),
                "key": key,
                "success_action_status": "200",
            },
            file_path=path,
            timeout=timeout,
        )
        return f"oss://{key}"

    def _get_policy(
        self,
        *,
        api_key: str,
        base_url: str,
        upload_path: str,
        model: str,
        timeout: float,
    ) -> dict[str, Any]:
        query = urllib.parse.urlencode({"action": "getPolicy", "model": model})
        url = base_url.rstrip("/") + "/" + upload_path.lstrip("/") + f"?{query}"
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            raise RuntimeError(f"DashScope upload policy HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"DashScope upload policy request failed: {exc.reason}") from exc
        if not isinstance(decoded, dict) or not isinstance(decoded.get("data"), dict):
            raise RuntimeError(f"DashScope upload policy returned invalid JSON: {decoded}")
        return cast(dict[str, Any], decoded["data"])

    def _post_multipart(
        self,
        *,
        url: str,
        fields: dict[str, str],
        file_path: Path,
        timeout: float,
    ) -> None:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise RuntimeError(f"DashScope upload_host is not an HTTP URL: {url}")
        boundary = f"----atas-dashscope-{secrets.token_hex(12)}"
        prefix = bytearray()
        for name, value in fields.items():
            prefix.extend(f"--{boundary}\r\n".encode("ascii"))
            prefix.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("ascii"))
            prefix.extend(value.encode("utf-8"))
            prefix.extend(b"\r\n")
        file_name = file_path.name
        mime = mimetypes.guess_type(file_name)[0] or "application/octet-stream"
        prefix.extend(f"--{boundary}\r\n".encode("ascii"))
        prefix.extend(
            (
                f'Content-Disposition: form-data; name="file"; filename="{file_name}"\r\n'
                f"Content-Type: {mime}\r\n\r\n"
            ).encode()
        )
        suffix = f"\r\n--{boundary}--\r\n".encode("ascii")
        content_length = len(prefix) + file_path.stat().st_size + len(suffix)
        target = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        connection_cls = (
            http.client.HTTPSConnection
            if parsed.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_cls(parsed.netloc, timeout=timeout)
        try:
            connection.putrequest("POST", target)
            connection.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
            connection.putheader("Content-Length", str(content_length))
            connection.endheaders()
            connection.send(bytes(prefix))
            with file_path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    connection.send(chunk)
            connection.send(suffix)
            response = connection.getresponse()
            body = response.read().decode("utf-8", errors="replace")[:2000]
            if response.status != 200:
                raise RuntimeError(f"DashScope temporary upload HTTP {response.status}: {body}")
        finally:
            connection.close()


MAX_R2V_REFERENCES = 5
MAX_REFERENCE_VIDEO_OUTPUT_SECONDS = 10


class Wan27Client:
    """Alibaba Model Studio Wan 2.7 async task client.

    The client keeps AVWM's provider boundary on `V2VRequest` while mapping
    production intent to Wan media roles:
      - REOBSERVE -> R2V with target-camera geometry video and the event-master first frame.
      - CONTINUE with geometry -> R2V with L1 target-camera geometry video and the
        previous accepted last frame as reference_image (not a second video).
      - CONTINUE without geometry -> provider-native I2V/T2V fallback.
    """

    def __init__(
        self,
        config: Wan27Config,
        *,
        transport: Wan27Transport | None = None,
        temp_stager: Wan27TempMediaStager | None = None,
    ) -> None:
        self.config = config
        self.transport = transport or UrllibWan27Transport()
        self.temp_stager = temp_stager or DashScopeTempOSSStager()

    async def submit(self, request: V2VRequest) -> V2VProviderResult:
        self._api_key()
        references = self._prepare_continue_first_clip(
            request,
            self._references(request),
            duration=math.ceil(request.shot.duration),
        )
        references = self._fit_reference_video_budget(request, references)
        references = self._fit_reference_count_budget(request, references)
        return await self._submit_resolved(request, references)

    def _fit_reference_count_budget(
        self,
        request: V2VRequest,
        references: tuple[MediaReference, ...],
    ) -> tuple[MediaReference, ...]:
        """Drop the least load-bearing references until the provider cap is met.

        ``build_payload`` asserts the cap, but asserting was all it did: a shot
        packed with six references died outright rather than shipping without a
        third character card. Priority order is geometry, then the continuity
        still that carries identity and end state, then extra character cards.
        """
        countable = [
            reference
            for reference in references
            if reference.media_type in {"video", "image"}
            and not self._reference_is_skipped(request, reference)
        ]
        if len(countable) <= MAX_R2V_REFERENCES:
            return references

        def priority(reference: MediaReference) -> int:
            return {
                ReferenceRole.TARGET_GEOMETRY: 0,
                ReferenceRole.FIRST_FRAME: 1,
                ReferenceRole.EVENT_MASTER: 2,
                ReferenceRole.CONTINUITY_MEMORY: 3,
                ReferenceRole.CHARACTER: 4,
                ReferenceRole.PROP: 5,
                ReferenceRole.SCENE_STYLE: 6,
            }.get(reference.role, 7)

        ranked = sorted(countable, key=lambda item: (priority(item), item.ordinal))
        keep = {id(reference) for reference in ranked[:MAX_R2V_REFERENCES]}
        dropped = [reference for reference in ranked[MAX_R2V_REFERENCES:]]
        for reference in dropped:
            print(
                f"Wan packing: dropped {reference.role.value} reference "
                f"{reference.path.name} to fit the {MAX_R2V_REFERENCES}-item cap",
                flush=True,
            )
        return tuple(
            reference
            for reference in references
            if reference not in countable or id(reference) in keep
        )

    async def _submit_resolved(
        self,
        request: V2VRequest,
        references: tuple[MediaReference, ...],
        *,
        identity_stills_dropped: bool = False,
    ) -> V2VProviderResult:
        stable_urls = {
            reference.path: f"internal-{reference.media_type}://{reference.role.value}/{index}"
            for index, reference in enumerate(references, start=1)
        }
        media = self._media_entries(request, references, media_urls=stable_urls)
        model = self._model_for(request.task, media)
        duration = self._provider_output_duration(
            request,
            model=model,
            media=media,
        )
        signature_payload = self.build_payload(
            request,
            references=references,
            duration=duration,
            media_urls=stable_urls,
        )
        request_id = self._request_id(request, references, signature_payload)
        run_dir = request.output_dir / "wan27" / request_id[:12]
        run_dir.mkdir(parents=True, exist_ok=True)
        artifact = run_dir / "result.mp4"
        state_path = run_dir / "state.json"
        if artifact.exists() and artifact.stat().st_size > 0:
            state = self._read_json(state_path)
            return V2VProviderResult(artifact, self._result_metadata(state, request_id))

        state = self._read_json(state_path)
        task_id = state.get("task_id") if isinstance(state.get("task_id"), str) else None
        if task_id is not None and str(state.get("status", "")).lower() in {
                "failed", "expired", "canceled", "cancelled"}:
            # A task the provider ended without output produced nothing billable;
            # keep its record and allow one new submission instead of re-polling it.
            state_path.rename(run_dir / f"state.{str(state['status']).lower()}.{task_id}.json")
            state, task_id = {}, None
        if task_id is None:
            resumed = self._find_resumable_task(
                request.output_dir,
                model=model,
                duration=duration,
                prompt=request.prompt,
                seed=request.shot.seed,
                exclude=run_dir,
            )
            if resumed is not None:
                run_dir, state_path, state = resumed
                artifact = run_dir / "result.mp4"
                request_id = str(state.get("request_id") or request_id)
                task_id = str(state["task_id"])
        request_metadata = self._request_metadata(
            request,
            references,
            model=model,
            duration=duration,
        )
        self._write_json(run_dir / "request_metadata.json", request_metadata)
        payload: dict[str, Any] | None = None
        if task_id is None:
            try:
                self._enforce_persisted_task_budget(request.output_dir, run_dir)
            except RuntimeError:
                reused = self._latest_succeeded_run(request.output_dir, exclude=run_dir)
                if reused is None:
                    raise
                run_dir, state_path, state = reused
                artifact = run_dir / "result.mp4"
                request_id = str(state.get("request_id") or request_id)
                if artifact.exists() and artifact.stat().st_size > 0:
                    return V2VProviderResult(
                        artifact,
                        self._result_metadata(state, request_id),
                    )
                raise
            staged_urls = self._stage_media_urls(request, references, run_dir)
            payload = self.build_payload(
                request,
                references=references,
                duration=duration,
                media_urls=staged_urls,
            )
            self._preflight(request, references, duration, payload)
            if self.config.store_redacted_requests:
                redacted = self._redact_payload(payload)
                redacted["avwm_request_metadata"] = request_metadata
                self._write_json(run_dir / "request.redacted.json", redacted)

        headers = self._headers(resolve_oss=self._payload_uses_oss(payload))
        if task_id is None:
            created = self.transport.request_json(
                "POST",
                self._url(self.config.create_path),
                headers=headers,
                payload=payload,
                timeout=self.config.request_timeout_seconds,
            )
            task_id = self._task_id(created)
            state = {
                "schema_version": 1,
                "request_id": request_id,
                "task_id": task_id,
                "model": model,
                "status": "submitted",
                "planned_duration_seconds": float(request.shot.duration),
                "provider_output_duration_seconds": duration,
                "reference_contract": request_metadata["references"],
                "created_response": created,
            }
            self._write_json(state_path, state)

        final = await self._poll(task_id, headers, state_path, state)
        status = self._status(final)
        if status not in {"succeeded", "success", "completed"}:
            detail = final.get("error") or final.get("message") or final
            stripped = _without_character_stills(references)
            if (
                not identity_stills_dropped
                and stripped != references
                and _wan_identity_policy_code(detail)
            ):
                return await self._submit_resolved(
                    request,
                    stripped,
                    identity_stills_dropped=True,
                )
            raise RuntimeError(f"Wan2.7 task {task_id} ended with {status!r}: {detail}")
        video_url = self._video_url(final)
        if self.config.download_result:
            self.transport.download(
                video_url,
                artifact,
                timeout=self.config.request_timeout_seconds,
            )
        elif not artifact.exists():
            raise RuntimeError("Wan2.7 download_result=false requires an existing artifact")
        state.update(
            {
                "status": "succeeded",
                "final_response": final,
                "artifact": str(artifact),
                "provider_video_url": video_url,
            }
        )
        self._write_json(state_path, state)
        return V2VProviderResult(artifact, self._result_metadata(state, request_id))

    def _find_resumable_task(
        self,
        shot_output_dir: Path,
        *,
        model: str,
        duration: int,
        prompt: str,
        seed: int,
        exclude: Path,
    ) -> tuple[Path, Path, dict[str, Any]] | None:
        """Reuse a submitted task even when a fresh OSS upload changes its URL.

        Temporary media URLs are deliberately excluded from the semantic shot
        identity.  A provider task already running for this shot is therefore
        safer to poll than to submit a second paid request after a process
        restart or transient upload refresh.
        """

        candidates: list[tuple[float, Path, Path, dict[str, Any]]] = []
        active_statuses = {"submitted", "pending", "queued", "running"}
        for state_path in (shot_output_dir / "wan27").glob("*/state.json"):
            run_dir = state_path.parent
            if run_dir == exclude:
                continue
            state = self._read_json(state_path)
            task_id = state.get("task_id")
            status = str(state.get("status", "")).lower()
            if not isinstance(task_id, str) or status not in active_statuses:
                continue
            state_model = state.get("model")
            if not isinstance(state_model, str):
                redacted = self._read_json(run_dir / "request.redacted.json")
                state_model = redacted.get("model")
            if state_model != model:
                continue
            provider_duration = state.get("provider_output_duration_seconds")
            if provider_duration is not None and int(provider_duration) != duration:
                continue
            redacted = self._read_json(run_dir / "request.redacted.json")
            input_payload = redacted.get("input")
            parameters = redacted.get("parameters")
            if not isinstance(input_payload, dict) or input_payload.get("prompt") != prompt:
                continue
            if not isinstance(parameters, dict) or int(parameters.get("seed", seed)) != seed:
                continue
            persisted_media = input_payload.get("media")
            if isinstance(persisted_media, list):
                media_types = {
                    str(item.get("type", ""))
                    for item in persisted_media
                    if isinstance(item, dict)
                }
                if "first_frame" in media_types and media_types - {"first_frame", "last_frame"}:
                    # Wan 3.0 rejects first_frame with any other media except
                    # last_frame. Do not resume those doomed payloads.
                    continue
            candidates.append((state_path.stat().st_mtime, run_dir, state_path, state))
        if not candidates:
            return None
        _, run_dir, state_path, state = max(candidates, key=lambda item: item[0])
        return run_dir, state_path, state

    def _enforce_persisted_task_budget(self, shot_output_dir: Path, run_dir: Path) -> None:
        task_dirs: list[Path] = []
        for state_path in (shot_output_dir / "wan27").glob("*/state.json"):
            state = self._read_json(state_path)
            status = str(state.get("status", "")).lower()
            if status in {"failed", "expired", "canceled", "cancelled"}:
                continue
            if isinstance(state.get("task_id"), str):
                task_dirs.append(state_path.parent)
        if len(task_dirs) < self.config.max_persisted_tasks_per_shot:
            return
        raise RuntimeError(
            "Blocked before paid Wan submission: persistent per-shot task budget "
            f"{len(task_dirs)}/{self.config.max_persisted_tasks_per_shot} is exhausted. "
            f"Prospective request cache: {run_dir}. Reuse or edit an existing take first."
        )

    def _latest_succeeded_run(
        self,
        shot_output_dir: Path,
        *,
        exclude: Path,
    ) -> tuple[Path, Path, dict[str, Any]] | None:
        ranked: list[tuple[int, float, Path, Path, dict[str, Any]]] = []
        for state_path in (shot_output_dir / "wan27").glob("*/state.json"):
            run_dir = state_path.parent
            if run_dir == exclude:
                continue
            artifact = run_dir / "result.mp4"
            if not artifact.is_file() or artifact.stat().st_size <= 0:
                continue
            state = self._read_json(state_path)
            if str(state.get("status", "")).lower() != "succeeded":
                continue
            contract = state.get("reference_contract")
            packing_c = 0
            if isinstance(contract, list):
                packing_c = int(
                    any(
                        isinstance(item, dict) and item.get("role") == "first_frame"
                        for item in contract
                    )
                )
            ranked.append(
                (packing_c, state_path.stat().st_mtime, run_dir, state_path, state)
            )
        if not ranked:
            return None
        _score, _mtime, run_dir, state_path, state = max(ranked, key=lambda item: item[:2])
        return run_dir, state_path, state

    def build_payload(
        self,
        request: V2VRequest,
        references: tuple[MediaReference, ...] | None = None,
        duration: int | None = None,
        *,
        media_urls: dict[Path, str] | None = None,
    ) -> dict[str, Any]:
        refs = references or self._references(request)
        output_duration = duration or math.ceil(request.shot.duration)
        media = self._media_entries(request, refs, media_urls=media_urls)
        raw_input: dict[str, Any] = {
            "prompt": request.prompt,
            "negative_prompt": self.config.negative_prompt,
        }
        if media:
            raw_input["media"] = media
        parameters: dict[str, Any] = {
            "duration": output_duration,
            "resolution": self.config.resolution,
            "ratio": self.config.ratio,
            "prompt_extend": self.config.prompt_extend,
            "watermark": self.config.watermark,
            "seed": request.shot.seed,
        }
        if self.config.generate_audio is not None:
            parameters["audio"] = bool(self.config.generate_audio)
        return {
            "model": self._model_for(request.task, media),
            "input": raw_input,
            "parameters": parameters,
        }

    def _media_entries(
        self,
        request: V2VRequest,
        references: tuple[MediaReference, ...],
        *,
        media_urls: dict[Path, str] | None,
    ) -> list[dict[str, str]]:
        if request.task is V2VTask.CONTINUE:
            return self._continue_media_entries(references, media_urls=media_urls)
        return self._reobserve_media_entries(references, media_urls=media_urls)

    def _continue_media_entries(
        self,
        references: tuple[MediaReference, ...],
        *,
        media_urls: dict[Path, str] | None,
    ) -> list[dict[str, str]]:
        if self._uses_point_memory_r2v(references) or self._uses_soft_reference_r2v(
            references
        ):
            return self._r2v_media_entries(references, media_urls=media_urls)
        first_clip = self._first_reference(
            references,
            roles={
                ReferenceRole.CONTINUITY_MEMORY,
                ReferenceRole.EVENT_MASTER,
                ReferenceRole.GLOBAL_STATE,
                ReferenceRole.USER_VIDEO,
            },
            media_type="video",
        )
        if first_clip is not None:
            return [
                {
                    "url": self._media_url(first_clip, media_urls=media_urls),
                    "type": "first_clip",
                }
            ]
        first_frame = self._first_reference(
            references,
            roles={ReferenceRole.FIRST_FRAME, ReferenceRole.GLOBAL_STATE},
            media_type="image",
        )
        if first_frame is None:
            return []
        return [
            {
                "url": self._media_url(first_frame, media_urls=media_urls),
                "type": "first_frame",
            }
        ]

    def _reobserve_media_entries(
        self,
        references: tuple[MediaReference, ...],
        *,
        media_urls: dict[Path, str] | None,
    ) -> list[dict[str, str]]:
        # Re-observation keeps the target geometry, while still allowing the
        # shot-specific identity/prop/style cards selected by the condition
        # builder.  The provider budget and role priority are enforced by the
        # common R2V packer below.
        packed = tuple(
            reference
            for reference in references
            if reference.role
            in {
                ReferenceRole.TARGET_GEOMETRY,
                ReferenceRole.FIRST_FRAME,
                ReferenceRole.CHARACTER,
                ReferenceRole.PROP,
                ReferenceRole.SCENE_STYLE,
                ReferenceRole.EVENT_MASTER,
                ReferenceRole.CONTINUITY_MEMORY,
                ReferenceRole.GLOBAL_STATE,
                ReferenceRole.USER_VIDEO,
            }
        )
        return self._r2v_media_entries(
            packed or references,
            media_urls=media_urls,
        )

    def _r2v_media_entries(
        self,
        references: tuple[MediaReference, ...],
        *,
        media_urls: dict[Path, str] | None,
    ) -> list[dict[str, str]]:
        usable = [
            reference
            for reference in references
            if reference.media_type in {"video", "image"}
            and not (
                reference.role is ReferenceRole.TARGET_GEOMETRY
                and not self.config.include_geometry_reference
            )
        ]
        first_frame = next(
            (
                reference
                for reference in usable
                if reference.role is ReferenceRole.FIRST_FRAME
                and reference.media_type == "image"
            ),
            None,
        )
        # Wan's R2V media list uses `first_frame` as a special input only for
        # image-only requests.  When a video reference is present, the
        # authored first frame is still required evidence, but it must be
        # encoded as an ordinary `reference_image` so it can coexist with the
        # dense target-camera video.  Keep the semantic FIRST_FRAME role on
        # the AVM reference for reports and prompt construction; only the
        # provider-facing media type is normalized here.
        has_reference_video = any(reference.media_type == "video" for reference in usable)
        # Wan 3.0 I2V: first_frame may only pair with last_frame. Character /
        # prop / style images belong on R2V as reference_image. If a shot also
        # explicitly requests identity/prop/style cards, pack the authored
        # first frame as an ordinary reference image so those cards are not
        # silently discarded.
        if first_frame is not None and not has_reference_video:
            has_soft_cards = any(
                reference.role
                in {ReferenceRole.CHARACTER, ReferenceRole.PROP, ReferenceRole.SCENE_STYLE}
                for reference in usable
            )
            if not has_soft_cards:
                return [
                    {
                        "url": self._media_url(first_frame, media_urls=media_urls),
                        "type": "first_frame",
                    }
                ]
        ordinary = [
            (index, reference)
            for index, reference in enumerate(usable)
            if reference is not first_frame
        ]
        ordinary_limit = min(
            5,
            max(0, self.config.max_reference_media - int(first_frame is not None)),
        )
        selected_indices = {
            index
            for index, _reference in sorted(
                ordinary,
                key=lambda item: (_r2v_reference_priority(item[1].role), item[0]),
            )[:ordinary_limit]
        }
        selected = [
            reference
            for index, reference in enumerate(usable)
            if index in selected_indices or reference is first_frame
        ]
        entries: list[dict[str, str]] = []
        for reference in selected:
            if reference.media_type == "video":
                entries.append(
                    {
                        "url": self._media_url(reference, media_urls=media_urls),
                        "type": "reference_video",
                    }
                )
            elif reference.media_type == "image":
                role = (
                    "reference_image"
                    if reference.role is ReferenceRole.FIRST_FRAME
                    and (has_reference_video or any(
                        item.role
                        in {
                            ReferenceRole.CHARACTER,
                            ReferenceRole.PROP,
                            ReferenceRole.SCENE_STYLE,
                        }
                        for item in usable
                    ))
                    else (
                        "first_frame"
                        if reference.role is ReferenceRole.FIRST_FRAME
                        else "reference_image"
                    )
                )
                entries.append(
                    {"url": self._media_url(reference, media_urls=media_urls), "type": role}
                )
        return entries

    def _uses_point_memory_r2v(
        self,
        references: tuple[MediaReference, ...],
    ) -> bool:
        return self.config.include_geometry_reference and any(
            reference.role is ReferenceRole.TARGET_GEOMETRY
            for reference in references
        )

    @staticmethod
    def _uses_soft_reference_r2v(
        references: tuple[MediaReference, ...],
    ) -> bool:
        return any(
            reference.role
            in {ReferenceRole.CHARACTER, ReferenceRole.PROP, ReferenceRole.SCENE_STYLE}
            for reference in references
        )

    def _first_reference(
        self,
        references: tuple[MediaReference, ...],
        *,
        roles: set[ReferenceRole],
        media_type: str,
    ) -> MediaReference | None:
        for reference in references:
            if reference.media_type == media_type and reference.role in roles:
                return reference
        return None

    def _preflight(
        self,
        request: V2VRequest,
        references: tuple[MediaReference, ...],
        duration: int,
        payload: dict[str, Any],
    ) -> None:
        self._api_key()
        if not self.config.min_duration_seconds <= duration <= self.config.max_duration_seconds:
            raise RuntimeError(
                f"Wan2.7 output duration must be {self.config.min_duration_seconds}-"
                f"{self.config.max_duration_seconds} whole seconds; got {request.shot.duration}"
            )
        raw_input = payload.get("input")
        media = raw_input.get("media") if isinstance(raw_input, dict) else None
        if not isinstance(media, list) or not media:
            if request.task is V2VTask.CONTINUE and payload.get("model") == self.config.model_t2v:
                return
            raise RuntimeError("Wan2.7 request has no usable media references")
        if len(media) > self.config.max_reference_media:
            raise RuntimeError(
                f"Wan2.7 accepts at most {self.config.max_reference_media} media references; "
                f"got {len(media)}"
            )
        media_types = {str(item.get("type", "")) for item in media if isinstance(item, dict)}
        reference_count = sum(
            1
            for item in media
            if isinstance(item, dict)
            and str(item.get("type", "")) in {"reference_image", "reference_video"}
        )
        first_frame_count = sum(
            1
            for item in media
            if isinstance(item, dict) and str(item.get("type", "")) == "first_frame"
        )
        if first_frame_count > 1:
            raise RuntimeError("Wan2.7 accepts at most one first_frame media item")
        extra_with_first_frame = media_types - {"first_frame", "last_frame"}
        if first_frame_count and extra_with_first_frame:
            raise RuntimeError(
                "Wan first_frame cannot be combined with other media types except "
                f"last_frame; got {sorted(extra_with_first_frame)}"
            )
        if reference_count > MAX_R2V_REFERENCES:
            raise RuntimeError(
                f"Wan2.7 R2V accepts at most {MAX_R2V_REFERENCES} "
                f"reference_image/reference_video items; got {reference_count}. "
                "_fit_reference_count_budget should have trimmed this before submission"
            )
        if request.task is V2VTask.REOBSERVE and reference_count < 1:
            raise RuntimeError(
                "Wan2.7 REOBSERVE requires at least one reference_image or reference_video"
            )
        model = str(payload.get("model", ""))
        if (
            model.startswith("wan2.7")
            and "reference_video" in media_types
            and duration > 10
        ):
            raise RuntimeError(
                "Wan2.7 R2V with reference_video supports 2-10 second output duration"
            )
        if (
            request.task is V2VTask.CONTINUE
            and payload.get("model") != self.config.model_r2v
            and not ("first_clip" in media_types or "first_frame" in media_types)
        ):
            raise RuntimeError("Wan2.7 CONTINUE requires first_clip or first_frame")
        for reference in references:
            if self._reference_is_skipped(request, reference):
                continue
            if reference.uri:
                continue
            if self.config.stage_local_media and self._is_local_media_reference(reference):
                continue
            if reference.media_type == "video":
                raise RuntimeError(
                    "Wan2.7 video references require provider/internal URLs. "
                    "Enable DashScope temporary staging or pass accepted Wan result URLs forward."
                )
            if not reference.path.exists() or reference.path.stat().st_size <= 0:
                raise FileNotFoundError(
                    f"Missing or empty {reference.prompt_handle} {reference.label}: "
                    f"{reference.path}"
                )

    def _provider_output_duration(
        self,
        request: V2VRequest,
        *,
        model: str,
        media: list[dict[str, str]],
    ) -> int:
        """Compile a legal Wan duration without changing story-time semantics.

        Wan 2.7 R2V has a stricter ten-second limit when a video is supplied as
        a reference.  That legacy limit must not be applied to newer models:
        Wan 3.0 is configured independently and can use the normal 15-second
        provider budget.  The backend propagates the provider duration to
        ``VideoClip.duration`` so memory and editing reflect the actual take.
        """

        requested = math.ceil(request.shot.duration)
        media_types = {str(item.get("type", "")) for item in media}
        if (
            model.startswith("wan2.7")
            and "reference_video" in media_types
        ):
            return min(requested, MAX_REFERENCE_VIDEO_OUTPUT_SECONDS)
        return min(requested, int(self.config.max_duration_seconds))

    async def _poll(
        self,
        task_id: str,
        headers: dict[str, str],
        state_path: Path,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        deadline = time.monotonic() + self.config.task_timeout_seconds
        while True:
            response = await self._poll_request_with_retry(
                task_id,
                headers,
            )
            status = self._status(response)
            state.update({"status": status, "last_response": response})
            self._write_json(state_path, state)
            if status in {
                "succeeded",
                "success",
                "completed",
                "failed",
                "cancelled",
                "canceled",
                "expired",
            }:
                return response
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Wan2.7 task {task_id} did not finish within "
                    f"{self.config.task_timeout_seconds:.0f}s; rerun to resume polling"
                )
            await asyncio.sleep(max(0.25, self.config.poll_interval_seconds))

    async def _poll_request_with_retry(
        self,
        task_id: str,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        """Retry safe task-status reads without repeating a paid submission."""

        attempts = max(1, int(self.config.transport_retries))
        backoff = max(0.0, float(self.config.transport_retry_backoff_seconds))
        url = self._url(self.config.task_path_template.format(task_id=task_id))
        for attempt in range(1, attempts + 1):
            try:
                return self.transport.request_json(
                    "GET",
                    url,
                    headers=headers,
                    payload=None,
                    timeout=self.config.request_timeout_seconds,
                )
            except Exception as exc:
                if attempt >= attempts or not _is_transient_wan_error(exc):
                    raise
                delay = min(backoff * (2 ** (attempt - 1)), 30.0)
                if delay > 0:
                    await asyncio.sleep(delay)
        raise RuntimeError("unreachable Wan2.7 polling retry state")

    def _references(self, request: V2VRequest) -> tuple[MediaReference, ...]:
        if request.condition.references:
            return request.condition.references
        if request.condition.geometry_source in {"bootstrap_image", "neutral_holes"}:
            # Cold-start conditions deliberately contain no observed geometry.
            # Do not turn their placeholder input video into an R2V reference.
            return ()
        return (
            MediaReference(
                path=request.condition.input_video,
                role=ReferenceRole.TARGET_GEOMETRY,
                ordinal=1,
                media_type="video",
                label="target-camera geometry proxy",
                duration_seconds=request.shot.duration,
            ),
        )

    def _fit_reference_video_budget(
        self,
        request: V2VRequest,
        references: tuple[MediaReference, ...],
    ) -> tuple[MediaReference, ...]:
        """Pack local video references under a provider-specific duration cap.

        Wan3.0 limits the combined duration of reference videos. Production
        packing C sends a single point-cloud video; this compressor remains
        for ablation B (two videos) or a single clip that exceeds the cap.
        """

        budget = self.config.max_reference_video_seconds
        if budget is None:
            return references
        videos = [
            reference
            for reference in references
            if reference.media_type == "video" and reference.path.is_file()
        ]
        if not videos:
            return references
        durations = {
            reference: max(0.1, self._duration(reference, request))
            for reference in videos
        }
        total = sum(durations.values())
        if total <= budget + 0.05:
            return references

        # A single accumulated CONTINUE clip can exceed the provider limit as
        # well. Keep it under the same cap instead of letting the API reject
        # the request before generation starts.
        if len(videos) == 1:
            allocations = {videos[0]: min(durations[videos[0]], budget)}
        else:
            minimum = min(2.0, budget / len(videos))
            remaining = max(0.0, budget - minimum * len(videos))
            weights = {
                reference: 1.5
                if reference.role is ReferenceRole.TARGET_GEOMETRY
                else 1.0
                for reference in videos
            }
            weight_total = sum(weights.values()) or 1.0
            allocations = {
                reference: min(
                    durations[reference],
                    minimum + remaining * weights[reference] / weight_total,
                )
                for reference in videos
            }
        allocated_total = sum(allocations.values())
        if allocated_total > budget:
            scale = budget / allocated_total
            allocations = {
                reference: max(0.5, duration * scale)
                for reference, duration in allocations.items()
            }

        packed: dict[Path, MediaReference] = {}
        for reference in videos:
            target_duration = allocations[reference]
            fingerprint = hashlib.sha256(
                (
                    f"{self._content_digest(reference.path)}:"
                    f"wan-reference-budget:{target_duration:.6f}"
                ).encode("ascii")
            ).hexdigest()[:12]
            output_dir = request.output_dir / "wan27_inputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            packed_path = output_dir / (
                f"reference_{reference.role.value}_{fingerprint}_"
                f"{target_duration:.3f}s.mp4"
            )
            if not packed_path.is_file() or packed_path.stat().st_size <= 0:
                speed = target_duration / durations[reference]
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-y",
                        "-i",
                        str(reference.path),
                        "-vf",
                        f"setpts={speed:.8f}*PTS",
                        "-an",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "fast",
                        "-crf",
                        "18",
                        "-pix_fmt",
                        "yuv420p",
                        str(packed_path),
                    ],
                    check=True,
                )
            packed[reference.path] = replace(
                reference,
                path=packed_path,
                duration_seconds=target_duration,
            )

        return tuple(packed.get(reference.path, reference) for reference in references)

    def _prepare_continue_first_clip(
        self,
        request: V2VRequest,
        references: tuple[MediaReference, ...],
        *,
        duration: int,
    ) -> tuple[MediaReference, ...]:
        if request.task is not V2VTask.CONTINUE or self._uses_point_memory_r2v(references):
            return references
        first_clip = self._first_reference(
            references,
            roles={
                ReferenceRole.CONTINUITY_MEMORY,
                ReferenceRole.EVENT_MASTER,
                ReferenceRole.GLOBAL_STATE,
                ReferenceRole.USER_VIDEO,
            },
            media_type="video",
        )
        if first_clip is None:
            return references
        if not first_clip.path.is_file():
            raise RuntimeError(
                "Wan2.7 continuation needs a local accepted clip so AVWM can extract its "
                f"tail as first_clip: {first_clip.path}"
            )

        fps = max(1, round(request.fps))
        max_context_frames = max(1, duration * fps - 1)
        requested_context_frames = max(
            1,
            round(self.config.continuation_first_clip_seconds * fps),
        )
        frame_count = min(max_context_frames, requested_context_frames)
        context_seconds = frame_count / fps
        if self._duration(first_clip, request) <= context_seconds + 1.0 / fps:
            return references
        fingerprint = hashlib.sha256(
            f"{self._content_digest(first_clip.path)}:"
            f"tail:{frame_count}:{fps}".encode()
        ).hexdigest()[:12]
        trim_dir = request.output_dir / "wan27_inputs"
        trim_dir.mkdir(parents=True, exist_ok=True)
        trimmed = trim_dir / f"first_clip_tail_{fingerprint}_{frame_count}f.mp4"
        if not trimmed.is_file() or trimmed.stat().st_size <= 0:
            command = [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-sseof",
                f"-{context_seconds:.6f}",
                "-i",
                str(first_clip.path),
                "-map",
                "0:v:0",
                "-frames:v",
                str(frame_count),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-r",
                str(fps),
                "-movflags",
                "+faststart",
                str(trimmed),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            if completed.returncode != 0 or not trimmed.is_file() or trimmed.stat().st_size <= 0:
                detail = completed.stderr.strip()[-2000:]
                raise RuntimeError(f"Failed to trim Wan2.7 first_clip: {detail}")

        replacement = replace(
            first_clip,
            path=trimmed,
            duration_seconds=context_seconds,
            uri=None,
        )
        return tuple(
            replacement if reference is first_clip else reference for reference in references
        )

    def _stage_media_urls(
        self,
        request: V2VRequest,
        references: tuple[MediaReference, ...],
        run_dir: Path,
    ) -> dict[Path, str]:
        if not self.config.stage_local_media or not references:
            return {}
        cache_path = run_dir / "staged_media.json"
        cache = self._read_json(cache_path)
        raw_items = cache.get("items")
        items: dict[str, Any] = raw_items if isinstance(raw_items, dict) else {}
        staged: dict[Path, str] = {}
        changed = False
        now = time.time()
        api_key = self._api_key()
        model = (
            self.config.model_r2v
            if request.task is V2VTask.REOBSERVE
            or self._uses_point_memory_r2v(references)
            or self._uses_soft_reference_r2v(references)
            else self.config.model_i2v
        )
        for reference in references:
            if self._reference_is_skipped(request, reference):
                continue
            if not self._is_local_media_reference(reference):
                continue
            if not reference.path.exists() or reference.path.stat().st_size <= 0:
                raise FileNotFoundError(
                    f"Missing or empty {reference.prompt_handle} {reference.label}: "
                    f"{reference.path}"
                )
            cache_key = str(reference.path.resolve())
            cached = items.get(cache_key)
            if self._cached_stage_is_valid(cached, reference.path, model, now=now):
                staged[reference.path] = str(cached["uri"])
                continue
            uri = self.temp_stager.stage(
                reference.path,
                api_key=api_key,
                base_url=self.config.base_url,
                upload_path=self.config.temp_upload_path,
                model=model,
                timeout=self.config.request_timeout_seconds,
            )
            stat = reference.path.stat()
            items[cache_key] = {
                "uri": uri,
                "model": model,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "expires_at": now + self.config.temp_oss_expiration_seconds,
                "media_type": reference.media_type,
                "role": reference.role.value,
            }
            staged[reference.path] = uri
            changed = True
        if changed:
            self._write_json(
                cache_path,
                {
                    "schema_version": 1,
                    "policy": "DashScope temporary oss:// URL; expires automatically",
                    "items": items,
                },
            )
        return staged

    def _reference_is_skipped(self, request: V2VRequest, reference: MediaReference) -> bool:
        del request
        return (
            reference.role is ReferenceRole.TARGET_GEOMETRY
            and not self.config.include_geometry_reference
        )

    def _is_local_media_reference(self, reference: MediaReference) -> bool:
        if reference.uri:
            return False
        raw_path = str(reference.path)
        if raw_path.startswith(("http://", "https://", "oss://", "data:")):
            return False
        return reference.media_type in {"image", "video"}

    def _cached_stage_is_valid(
        self,
        cached: object,
        path: Path,
        model: str,
        *,
        now: float,
    ) -> bool:
        if not isinstance(cached, dict):
            return False
        uri = cached.get("uri")
        if not isinstance(uri, str) or not uri.startswith("oss://"):
            return False
        if cached.get("model") != model:
            return False
        expires_at = _float_value(cached.get("expires_at"), 0.0)
        if expires_at <= now + self.config.temp_oss_refresh_margin_seconds:
            return False
        try:
            stat = path.stat()
        except OSError:
            return False
        return cached.get("size") == stat.st_size and cached.get("mtime_ns") == stat.st_mtime_ns

    def _media_url(
        self,
        reference: MediaReference,
        *,
        media_urls: dict[Path, str] | None,
    ) -> str:
        if media_urls is not None and reference.path in media_urls:
            return media_urls[reference.path]
        if reference.uri:
            return reference.uri
        raw_path = str(reference.path)
        if raw_path.startswith(("http://", "https://", "oss://")):
            return raw_path
        if reference.media_type == "image":
            return self._data_url(reference.path)
        raise RuntimeError(
            f"Wan2.7 video reference {reference.label!r} has no provider/internal URL"
        )

    def _duration(self, reference: MediaReference, request: V2VRequest) -> float:
        if reference.duration_seconds is not None:
            return max(0.0, reference.duration_seconds)
        try:
            props = iio.improps(reference.path)
            meta = iio.immeta(reference.path)
            frames = int(props.shape[0])
            fps = float(meta.get("fps", request.fps)) if isinstance(meta, dict) else request.fps
            return frames / max(1.0, fps)
        except Exception:
            return request.shot.duration

    def _model_for(self, task: V2VTask, media: list[dict[str, str]]) -> str:
        media_types = {item.get("type") for item in media}
        if task is V2VTask.REOBSERVE or media_types & {
            "reference_image",
            "reference_video",
        }:
            return self.config.model_r2v
        return self.config.model_i2v if media else self.config.model_t2v

    def _data_url(self, path: Path) -> str:
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _headers(self, *, resolve_oss: bool = False) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        }
        if resolve_oss:
            headers["X-DashScope-OssResourceResolve"] = "enable"
        return headers

    def _payload_uses_oss(self, value: Any) -> bool:
        if isinstance(value, dict):
            return any(self._payload_uses_oss(item) for item in value.values())
        if isinstance(value, list):
            return any(self._payload_uses_oss(item) for item in value)
        return isinstance(value, str) and value.startswith("oss://")

    def _api_key(self) -> str:
        if self.config.api_key is not None:
            value = self.config.api_key.get_secret_value().strip()
            if value:
                return value
        value = os.getenv(self.config.api_key_env, "").strip()
        if value:
            return value
        raise RuntimeError(
            f"Wan2.7 API key is missing. Set {self.config.api_key_env} in your environment "
            "or AVWM_BACKENDS__WAN27__API_KEY in .env."
        )

    def _request_id(
        self,
        request: V2VRequest,
        references: tuple[MediaReference, ...],
        payload: dict[str, Any],
    ) -> str:
        digest = hashlib.sha256()
        digest.update(request.shot.shot_id.encode())
        digest.update(request.prompt.encode())
        digest.update(json.dumps(self._redact_payload(payload), sort_keys=True).encode())
        for reference in references:
            if self._reference_is_skipped(request, reference):
                continue
            digest.update(reference.role.value.encode())
            digest.update(reference.media_type.encode())
            if reference.path.exists() and reference.media_type == "image":
                digest.update(reference.path.read_bytes())
            elif reference.uri:
                digest.update(reference.uri.encode())
            elif reference.path.exists():
                digest.update(self._content_digest(reference.path).encode())
            else:
                digest.update(str(reference.path).encode())
        return digest.hexdigest()

    @staticmethod
    def _content_digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _task_id(self, response: dict[str, Any]) -> str:
        output = response.get("output")
        candidates: list[object] = [response.get("task_id"), response.get("id")]
        if isinstance(output, dict):
            candidates.extend([output.get("task_id"), output.get("id")])
        for value in candidates:
            if isinstance(value, str) and value:
                return value
        raise RuntimeError(f"Wan2.7 create response has no task id: {response}")

    def _status(self, response: dict[str, Any]) -> str:
        output = response.get("output")
        candidates: list[object] = [response.get("task_status"), response.get("status")]
        if isinstance(output, dict):
            candidates.extend([output.get("task_status"), output.get("status")])
        for value in candidates:
            if isinstance(value, str) and value:
                lowered = value.lower()
                return "succeeded" if lowered == "succeeded" else lowered
        return ""

    def _video_url(self, response: dict[str, Any]) -> str:
        containers: list[object] = [response, response.get("output"), response.get("result")]
        output = response.get("output")
        if isinstance(output, dict):
            containers.append(output.get("results"))
        for container in containers:
            value = self._video_url_from_container(container)
            if value is not None:
                return value
        raise RuntimeError("Wan2.7 succeeded but returned no video_url")

    def _video_url_from_container(self, container: object) -> str | None:
        if isinstance(container, dict):
            for key in ("video_url", "url", "output_url"):
                value = container.get(key)
                if isinstance(value, str) and value.startswith(("http://", "https://")):
                    return value
            for value in container.values():
                nested = self._video_url_from_container(value)
                if nested is not None:
                    return nested
        if isinstance(container, list):
            for item in container:
                nested = self._video_url_from_container(item)
                if nested is not None:
                    return nested
        return None

    def _url(self, path: str) -> str:
        return self.config.base_url.rstrip("/") + "/" + path.lstrip("/")

    def _redact_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return cast(dict[str, Any], self._redact_value(payload))

    def _redact_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            redacted: dict[str, Any] = {}
            for key, item in value.items():
                if key == "url" and isinstance(item, str):
                    redacted[key] = self._redacted_url(item)
                else:
                    redacted[key] = self._redact_value(item)
            return redacted
        if isinstance(value, list):
            return [self._redact_value(item) for item in value]
        return value

    def _redacted_url(self, url: str) -> str:
        if url.startswith("data:"):
            return f"<embedded-media:{len(url):,}-chars>"
        return "<provider-or-internal-media-url>"

    @staticmethod
    def _request_metadata(
        request: V2VRequest,
        references: tuple[MediaReference, ...],
        *,
        model: str,
        duration: int,
    ) -> dict[str, Any]:
        """Persist AVM's complete reference contract beside a provider request.

        The metadata is deliberately local and is not added to the DashScope
        payload, whose schema accepts only media URLs.  It gives acceptance and
        stale-local audits stable absolute paths and character board semantics.
        """
        return {
            "schema_version": 1,
            "shot_id": request.shot.shot_id,
            "task": request.task.value,
            "model": model,
            "duration_seconds": duration,
            "cast_board_id": request.shot.cast_board_id or None,
            "cast_board_legacy": request.shot.cast_board_legacy,
            "visible_cast": list(request.shot.visible_cast),
            "character_mapping": list(request.shot.cast_board_bindings),
            "references": [
                {
                    "path": str(reference.path.expanduser().resolve()),
                    "role": reference.role.value,
                    "asset_role": reference.asset_role or None,
                    "ordinal": reference.ordinal,
                    "media_type": reference.media_type,
                    "label": reference.label,
                    "character_mapping": list(reference.character_mapping),
                }
                for reference in references
            ],
        }

    def _read_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _result_metadata(self, state: dict[str, Any], request_id: str) -> dict[str, object]:
        expected_width, expected_height = _expected_output_size(
            self.config.resolution,
            self.config.ratio,
        )
        return {
            "provider": "alibaba-model-studio",
            "provider_request_hash": request_id,
            "provider_task_id": str(state.get("task_id", "")),
            "provider_video_url": (
                str(state.get("provider_video_url"))
                if isinstance(state.get("provider_video_url"), str)
                else None
            ),
            "provider_video_url_ttl_seconds": 86400,
            "provider_url_policy": "temporary provider/internal URL; do not public-relay",
            "model_r2v": self.config.model_r2v,
            "model_i2v": self.config.model_i2v,
            "resolution": self.config.resolution,
            "ratio": self.config.ratio,
            "expected_width": expected_width,
            "expected_height": expected_height,
            "planned_duration_seconds": state.get("planned_duration_seconds"),
            "provider_output_duration_seconds": state.get(
                "provider_output_duration_seconds"
            ),
            "reference_contract": state.get("reference_contract"),
        }


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"DashScope upload policy missing {key!r}: {payload}")
    return value


def _expected_output_size(resolution: str, ratio: str) -> tuple[int | None, int | None]:
    match = re.search(r"(\d+)", resolution)
    if match is None or ":" not in ratio:
        return None, None
    base = int(match.group(1))
    left, right = ratio.split(":", 1)
    try:
        aspect = float(left) / float(right)
    except (ValueError, ZeroDivisionError):
        return None, None
    if aspect >= 1.0:
        height = base
        width = round(base * aspect)
    else:
        width = base
        height = round(base / aspect)
    return _nearest_even(width), _nearest_even(height)


def _nearest_even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


def _float_value(value: object, default: float) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default
