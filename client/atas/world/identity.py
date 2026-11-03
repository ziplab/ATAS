"""Stable screenplay identities and explicit appearance mapping."""
from __future__ import annotations
import hashlib
from pathlib import Path


def normalized_label(label):
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKC",str(label)).casefold() if c.isalnum())


def appearance_pairs(shot):
    labels=list(shot.entity_ids)
    mapping=getattr(shot,"raw",{}).get("character_reference_map",{})
    result=[]
    for path in shot.character_reference_images:
        explicit=[name for name,p in mapping.items() if Path(p)==path]
        matches=explicit or [name for name in labels if normalized_label(name)==normalized_label(path.stem)]
        if not matches and len(labels)==len(shot.character_reference_images)==1: matches=labels
        if len(matches)!=1 or matches[0] not in labels:
            raise ValueError(f"ambiguous character reference {path.name}; supply character_reference_map")
        result.append((matches[0],path))
    if len({name for name,_ in result})!=len(result): raise ValueError("duplicate character appearance mapping")
    return result


def registry_for_shot(shot):
    references=dict(appearance_pairs(shot)); records=[]
    for label in shot.entity_ids:
        path=references.get(label)
        digest=hashlib.sha256(path.read_bytes()).hexdigest() if path and path.is_file() else None
        eid="entity_"+hashlib.sha256(f"{shot.story_id}:{normalized_label(label)}".encode()).hexdigest()[:16]
        records.append({"entity_id":eid,"label":label,"reference_sha256":digest,
                        "reference_path":str(path) if path else None})
    return records
