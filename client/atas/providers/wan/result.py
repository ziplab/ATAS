from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class V2VProviderResult:
    """Minimal result returned by a provider-specific API client."""

    artifact_path: Path
    metadata: dict[str, object] = field(default_factory=dict)
