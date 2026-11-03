# Geometry service HTTP API

The client (`client/atas/world/geometry_client.py`) is the only caller. All `/v1` routes accept
`Authorization: Bearer $GEOMETRY_API_TOKEN` when a token is configured. Identifiers
(`run_id`, `story_id`, `scene_id`, `shot_id`, `event_id`) are `[A-Za-z0-9_-]+`.

## Health

- `GET /healthz` → `{"status": "ok"}` while the process serves requests.
- `GET /readyz` → `200` with `"ready": true` when the Vista4D code, its Python, the Pi3X and SAM3
  weights, a writable hot root and `nvidia-smi` are all present; `503` and the failing check otherwise.

## Jobs

`POST /v1/jobs` creates a job and returns `{"job_id": ...}`. Identical requests return the same job
(a request fingerprint); `retry_failed: true` re-queues a failed one. Poll `GET /v1/jobs/{job_id}`
until `status` is `succeeded` or `failed`. While a job waits for a card its `stage` is
`waiting_gpu`; admission is by free GPU memory.

### Reconstruction (multipart form)

Fields: `video` (the reviewed clip), `run_id`, `story_id`, `shot_id`, `scene_id`, `event_id`,
`duration`, `world_time_start`, `dynamic_keywords` (JSON list of SAM3 class nouns observed in the
clip), `entity_registry`, `review_lineage`, `publication_mode=reviewed_candidate_v1`.

The service reconstructs the clip (Pi3X + SAM3, dense per-frame RGB-D and dynamic masks), then
either builds the scene's initial world (first observation) or registers the clip to the committed
scene world with anchored fusion. The result's `fuse` block reports `registered` (the paper's `r`)
and, when registered, a `candidate_job_id` with `publication_status: pending_review`. Nothing is
published by reconstruction alone.

### Render (JSON, `type: "render"`, `camera_contract: "semantic-camera-v1"`)

Renders the committed scene memory along a planned camera: `geometry_layers: "l1"` for ordinary
shots, `"l1+l2"` with `render_mode: "recapture"`, `source_shot_id`, `source_event_id`,
`source_time_start/end` for a historical event (static L1 and the source clip's time-local dynamic
points share one depth buffer). `camera_plan` comes from the visual planner after
`/v1/camera-context`; `camera_relation` is `continuous`, `cut_recompose` or `recapture`.
Fetch the result with `GET /v1/jobs/{job_id}/artifacts/memory_input.mp4`.

## Camera context

`POST /v1/camera-context` returns the observation the visual planner grounds subjects in (source
shot, frame index, crop and sampling metadata, current world version) and a `context_id`. A render
whose plan carries a stale `context_id` is refused.

## World publication

- `POST /v1/world-publication` publishes a registered candidate as the next world version. It is the
  state-update check (`a`): the film review must certify the exact reconstructed bytes, the review
  and reconstruction lineage must agree, the candidate must belong to the requested scene, and its
  parent version must still be current. A stale parent returns `409`.
- `POST /v1/world-rebase` re-registers such a candidate against the newer world.
