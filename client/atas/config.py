"""Configuration: ``client/.env`` (git-ignored; ``ATAS_ENV_FILE`` overrides) plus the process environment."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = Path(os.environ.get("ATAS_ENV_FILE", str(ROOT / ".env")))


def load_dotenv(path: Path = ENV_PATH) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def apply_generation_defaults() -> None:
    os.environ.setdefault("AVWM_IMAGE__MODEL", "gpt-image-2")
    os.environ.setdefault("AVWM_IMAGE_STREAM", "0")
    os.environ.setdefault("AVWM_IMAGE_RETRIES", "4")
    os.environ.setdefault("AVWM_BACKENDS__WAN27__ENABLED", "true")
    for kind in ("R2V", "I2V", "T2V"):
        os.environ.setdefault(f"AVWM_BACKENDS__WAN27__MODEL_{kind}", "wan3.0-video")
    os.environ.setdefault("AVWM_BACKENDS__WAN27__RESOLUTION", "1080P")
    os.environ.setdefault("AVWM_BACKENDS__WAN27__WIDTH", "1920")
    os.environ.setdefault("AVWM_BACKENDS__WAN27__HEIGHT", "1080")
    os.environ.setdefault("AVWM_BACKENDS__WAN27__STAGE_LOCAL_MEDIA", "true")
    os.environ.setdefault("AVWM_BACKENDS__WAN27__INCLUDE_GEOMETRY_REFERENCE", "true")
    os.environ.setdefault("GEOMETRY_ENABLED", "true")
    responses = os.environ.get("ATAS_LLM_URL", "").strip().rstrip("/")
    if responses:
        base = responses[: -len("/responses")] if responses.endswith("/responses") else responses
        os.environ.setdefault("AVWM_IMAGE_BASE_URL", base)


def configure() -> Path:
    load_dotenv()
    apply_generation_defaults()
    return ROOT
