# Design notes: the world state

This document describes how the geometry service builds, validates and renders the world state
`W_t = (S_t, E_t)`. The client-side loop is described in the top-level README.

## Layers

| Layer | Content | Lifetime |
|---|---|---|
| **L0** — observation | One reconstructed clip: every frame's RGB, depth (EXR), camera, SAM3 dynamic masks and per-instance tracks, plus `observation.json` (scene, event, duration, fps, source frame count, video SHA-256, raw→scene similarity transform) | Immutable once committed |
| **L1** — scene memory `S_t` | Static points of one physical scene in that scene's coordinate frame (dynamic regions, sky and unstable depth edges excluded) | Append-only, versioned per scene |
| **L2** — event memory `E_t` | The time-indexed dynamic evidence of a committed L0, unprojected on demand with its registration transform | Bound to its event and source interval; never fused into L1 |

Coordinates are reconstruction units of the scene's first observation, not certified metres.

## Reconstruction

Each reviewed clip is reconstructed with Pi3X and segmented with SAM3 (class nouns observed by the
film review). Defaults are the low-VRAM configuration: windowed Pi3X inference (48-frame windows,
24 in low-VRAM mode, 6-frame overlap, ~160k-pixel inference limit), SAM3 frames and outputs
offloaded to CPU. Whole-clip inference (`GEOMETRY_PI3_VO_CHUNK=0`) is an option for large cards and
falls back to windows above `GEOMETRY_PI3_MAX_SINGLE_PASS_FRAMES` (default 160).

## The first observation of a scene

The first admitted clip of a place defines the scene frame, so every later registration inherits its
errors. Its internal consistency is checked by lifting matched pixels of frame pairs (13 frames
sampled across the clip) into fixed coordinates and measuring their disagreement against a tolerance
`t` of 2% of median depth.

- **Consistent** — at least three supported pairs that together span ≥ 80% of the clip, none beyond
  `2t`: the clip is published as reconstructed.
- **Drift** — two pairs beyond `2t`, or one beyond `8t`: the clip is re-reconstructed with its first
  window as a fixed anchor.
- **Insufficient evidence** — too few pairs, or pairs that leave part of the clip unchecked. Absence
  of evidence is not consistency: the clip is re-reconstructed against its first window as well, which
  subjects it to the same held-out and seam validation as any later registration.

A re-reconstructed clip is accepted when that validation passes and every original pair keeps its
evidence and ends within `2t`. Otherwise the clip is delivered as film only (`r = 0`) and the next clip
of the place becomes the candidate first observation.

## Registering later observations (anchored fusion)

A later clip of a known scene is reconstructed *conditioned on history* rather than independently
and aligned afterwards:

- up to six frames of trusted earlier observations of the scene are injected into every Pi3X window;
  four carry static depth, pose and intrinsics, two carry RGB only and are held out for validation;
- historical priors exclude dilated dynamic masks, sky and unstable depth edges; SAM never runs
  across a history/new boundary;
- a residual similarity transform is fitted from repeated historical-pixel correspondences and
  applied once to both camera translations and depth;
- acceptance (`r = 1`) requires the held-out history frames, chunk-overlap static points and
  independent new/history SIFT correspondences (≥ 30, verified in world coordinates without
  refitting) to agree. Up to two trusted references are tried; a rejected registration never
  touches the world.

New static points are appended only where they are novel (not near existing points or behind an
observed historical surface), are not contradicted by historical free space seen from at least two
views, and are observed in at least two of three temporal bins. Existing points are kept
byte-for-byte; old geometry and colour are never averaged or warped.

## Publication and versions (`a`)

Reconstruction produces a *candidate* version. `/v1/world-publication` commits it only when the
film review certifies the exact reconstructed bytes, review and reconstruction lineage agree, the
candidate belongs to the requested scene, and its parent is still the current version (a stale
parent is rebased by re-registering against the newer world). Each scene keeps
`world_state.json` (current version, parent, L1, admitted observations) and an immutable
`.world_versions/<version>/commit.json`; the pointer is replaced atomically after the manifest.

## Rendering the memory (query)

The visual planner grounds the shot's focal subjects in an earlier observation (frame-local boxes
and seed points, never a persistent identity). The service lifts them with seed-local depth and
solves the camera: the selected part's projected extent sets distance, the subject is placed at the
requested screen position, continuous shots keep the previous endpoint pose at frame zero, cuts
choose a new start. Supported motions include locked, push/pull, orbit/arcs, truck, crane, pan,
tilt, handheld and tracking moves. Candidate paths are checked for clearance against sampled
geometry; unsafe ones are rejected, not converted to a locked shot. Coverage of the rendered frame
is a soft ranking signal, never a veto.

Static points are splatted with a footprint of their voxel size projected to the image (at most
5×5 pixels at 1080p); unobserved pixels stay grey. For a recapture, the source clip's dynamic points
at each requested source time (using its recorded fps and centre-slice offset) are composited with
L1 through one depth buffer. The number of points kept resident on the GPU follows the card's free
memory; larger scenes stream in batches.