"""Character identity references for a screenplay that ships without images.

The recursive compiler's skeleton names the recurring characters with one portrait
prompt each (``portrait_plan``); gpt-image renders them into a character manifest
(label -> image path), exactly like a user-supplied ``--characters`` file.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from atas.runner.recon_ledger import write_json_atomic
def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def render_portraits(rows: list[dict], style: str, folder: Path) -> dict[str, str]:
    """Render identity references for planner portrait entries ({id, prompt}). Idempotent."""
    from atas.providers.imagegen import generate as render_image
    from PIL import Image
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    assets = {}
    for row in rows:
        label = str(row["id"])
        target = folder / f"{label}.png"
        receipt = target.with_suffix(".json")
        prompt = (f"{row.get('prompt') or row.get('name') or label}. Single character identity reference, "
                  f"full body, plain background, no text. Visual style: {style}.")
        key = _digest({"prompt": prompt})
        if not (target.is_file() and receipt.is_file() and json.loads(receipt.read_text()).get("key") == key):
            render_image(prompt, target)
            with Image.open(target) as image:
                image.verify()
            write_json_atomic(receipt, {"key": key, "label": label, "prompt": prompt})
        assets[label] = str(target.resolve())
    return assets
