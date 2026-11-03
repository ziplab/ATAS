"""SAM3 nouns: use the plan when it already lists them; otherwise resolve them on the client."""

from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path

from atas.world.dynamic_keywords import DynamicKeywordError, resolve_dynamic_keywords
from atas.request import COMPILER_VERSION, PlanShot
def screenplay_context(shot: PlanShot) -> str:
    onstage = ", ".join(shot.entity_ids) if shot.entity_ids else "(none)"
    return (
        f"Onstage character labels (map to class nouns, do not send these names to SAM): {onstage}\n"
        f"Shot prompt: {shot.prompt.strip()[:900]}"
    )


def uses_observed_dynamics(shot: PlanShot) -> bool:
    return shot.raw.get('planning_provenance', {}).get('kind') in {
        'ai-story-planner-v1', COMPILER_VERSION,
    }


def planned_sam_nouns(shot: PlanShot) -> tuple[str, ...]:
    # Whole-film AI vocabulary describes the world, NOT observed dynamic masks.
    # A pond/bed appearing in a story must never automatically be removed from L1.
    if uses_observed_dynamics(shot):
        return ()
    raw = shot.raw.get("sam_keywords")
    if not isinstance(raw, list):
        return ()
    nouns = tuple(str(item).strip().lower() for item in raw if str(item).strip())
    return nouns


def sam_nouns_for_recon(shot: PlanShot, video: Path, output_dir: Path) -> tuple[str, ...]:
    """After Wan, before GPU recon. Empty onstage may be []. Onstage + empty nouns fails."""
    # A film audit already inspected these exact video bytes, current
    # actors and handled props. Reuse its grounded classes rather than run a
    # second vision call followed by a text filter that unions everything back.
    from atas.review.film_review import file_hash, validate_dynamic_classes
    from atas.runner.recon_ledger import write_json_atomic
    review_path=output_dir/'film_intent_review.json'
    if review_path.is_file():
        review=json.loads(review_path.read_text())
        authorized=review.get('decision') in {'pass','warning'}
        if (review.get('stage')=='film' and authorized
                and 'observed_dynamic_classes' in review and review.get('video_sha256')==file_hash(video)):
            rows=validate_dynamic_classes(review['observed_dynamic_classes'])
            nouns=tuple(row['noun'] for row in rows)
            if nouns or not shot.entity_ids:
                write_json_atomic(output_dir/'dynamic_keywords.json',{
                    'schema_version':1,'video':str(video),'video_sha256':review['video_sha256'],
                    'resolved':list(nouns),'source':'film_review_observed_classes_v1',
                    'dynamic_source':'film_review_observed_classes_v1',
                    'review_cache_key':review.get('cache_key'),'evidence':rows})
                return nouns
    planned = planned_sam_nouns(shot)
    if planned:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "dynamic_keywords.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "video": str(video),
                    "resolved": list(planned),
                    "dynamic_source": "plan_sam_keywords",
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"SAM keywords={list(planned)} (from plan {shot.shot_id})",
            flush=True,
        )
        return planned
    last: BaseException | None = None
    ai_plan = uses_observed_dynamics(shot)
    resolve_dir = output_dir
    if ai_plan:
        from atas.review.film_review import file_hash
        key = hashlib.sha256(json.dumps({'video':file_hash(video),
            'context':screenplay_context(shot),'policy':'observed-dynamics-v1'},sort_keys=True).encode()).hexdigest()
        resolve_dir = output_dir/'keyword_observations'/key
    for attempt in range(4):
        try:
            nouns = resolve_dynamic_keywords(
                video,
                screenplay_context=screenplay_context(shot),
                output_dir=resolve_dir,
                allow_empty=not shot.entity_ids,
            )
            if ai_plan:
                from atas.runner.recon_ledger import write_json_atomic
                report=json.loads((resolve_dir/'dynamic_keywords.json').read_text())
                write_json_atomic(output_dir/'dynamic_keywords.json',{**report,
                    'cache_key':key,'ignored_whole_story_keywords':shot.raw.get('sam_keywords',[]),
                    'source':'observed-dynamics-v1','evidence_dir':str(resolve_dir)})
            return tuple(nouns)
        except (TimeoutError, OSError, DynamicKeywordError) as exc:
            last = exc
            time.sleep(min(30, 2 ** attempt))
    assert last is not None
    raise last
