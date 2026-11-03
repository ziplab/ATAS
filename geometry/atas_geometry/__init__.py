"""ATAS geometry service: reconstruction, registration, world versions and memory rendering."""
from __future__ import annotations

from pathlib import Path

GEOMETRY_ROOT = Path(__file__).resolve().parents[1]      # geometry/: put on PYTHONPATH of subprocesses
BACKBONES_DIR = GEOMETRY_ROOT / "backbones"   # bundled Pi3X reconstruction and SAM3 segmentation code
