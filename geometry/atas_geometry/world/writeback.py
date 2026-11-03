"""Validated production writeback with immutable observation/world revisions."""
from __future__ import annotations
import json
import os
import shutil
import subprocess
from pathlib import Path
import numpy as np
from atas_geometry.service.durable import atomic_json, file_hash
from atas_geometry.world.world_state import load_world, current_l1, checked_path, commit_world
from atas_geometry.register.registration_recovery import run_bounded, accepted as registration_accepted, diagnose as registration_diagnose, MAX_ATTEMPTS
def _count(path):
    with np.load(path) as data: return len(data["xyz"])


def _snapshot_l0(source, target):
    # Immutable files can share storage; metadata is replaced atomically later.
    # Resume an interrupted directory traversal without accepting a partial L0.
    def link_missing(src,dst):
        if not Path(dst).exists(): os.link(src,dst)
        return dst
    shutil.copytree(source,target,copy_function=link_missing,symlinks=False,dirs_exist_ok=True)


def _metadata(l0,payload):
    from atas_geometry.world.recapture import record_observation
    identity={"scale":1.,"rotation":np.eye(3).tolist(),"translation":[0.,0.,0.]}
    record_observation(l0,payload,{"merged":True,"transform":identity})
    p=l0/"observation.json"; meta=json.loads(p.read_text())
    meta.update(coordinate_space="scene_world",raw_to_scene=identity,registered=True,
                entity_registry=payload.get("entity_registry",[]),
                source_review_lineage=payload.get("review_lineage"))
    if payload.get("reconstruction_contract"):
        from atas_geometry.reconstruct.recon_revision import contract_digest
        meta.update(source_reconstruction_contract=payload["reconstruction_contract"],
                    source_reconstruction_contract_digest=contract_digest(payload))
    atomic_json(p,meta)


def trusted_references(scene, home):
    state=load_world(scene)
    if not state: raise ValueError("registration requires a committed scene world")
    trusted=[checked_path(scene,r["l0"]) for r in state["observations"].values()]
    candidates=trusted[-3:]
    home=Path(home).resolve()
    if home in trusted: candidates.insert(0,home)
    return list(dict.fromkeys(candidates))


def rank_references(scene,source,home):
    from atas_geometry.register.anchored_fusion import strict_store, select_history
    candidates=trusted_references(scene,home)
    new=strict_store(source); ranked=[]
    for path in dict.fromkeys(candidates):
        if path.resolve()==Path(source).resolve(): continue
        try:
            ref=strict_store(path)
            _,scores=select_history(ref,new)
            ranked.append((sum(s["static_matches"] for s in scores),str(path)))
        except (ValueError,FileNotFoundError): continue
    if not ranked: raise ValueError("no trusted historical observation for this scene")
    selected=[Path(path) for score,path in sorted(ranked,reverse=True) if score>=30]
    if not selected: raise ValueError("no historical overlap; cannot register a disconnected new observation")
    return selected[:MAX_ATTEMPTS]


def writeback_world(l0,scene,*,payload,job_id,settings,gpu_index,home_l0,home_fuse=None):
    scene=Path(scene); state=load_world(scene)
    report={}   # registration report; the first observation of a scene has none
    shot=payload["shot_id"]; digest=payload.get("video_sha256") or file_hash(payload["video_path"])
    if state and shot in state["observations"]:
        row=state["observations"][shot]
        if row["source_sha256"]!=digest: raise ValueError("accepted shot id has different video content")
        accepted=checked_path(scene,row["l0"])
        return dict(merged=False,registered=True,registration_status="registered",already_committed=True,
                    l1_point_count=_count(current_l1(scene)),world_version=state["version"],
                    accepted_l0=str(accepted),l1_npz=str(current_l1(scene)),novel_points=0,
                    fusion_version="anchored_world_v1")
    version=job_id
    folder=scene/".world_versions"/version; folder.mkdir(parents=True,exist_ok=True)
    expected=(state or {}).get("version")
    original=Path(home_fuse["candidate_l1"]) if home_fuse and home_fuse.get("candidate_l1") else current_l1(scene)
    before=_count(original)
    if home_fuse:
        observation=folder/"l0"
        _snapshot_l0(l0,observation)
        candidate=folder/"l1.npz"
        if not candidate.exists(): os.link(original,candidate)
        added=before
    else:
        trusted=trusted_references(scene,home_l0)
        recovery_path=folder/"registration_recovery.json"
        intent=folder/"intent.json"
        if recovery_path.exists():
            references=[Path(p) for p in json.loads(recovery_path.read_text())["contract"]["references"]]
        else:
            references=rank_references(scene,l0,home_l0)
            if intent.exists():
                previous=json.loads(intent.read_text())
                if (previous.get("source_sha256"),previous.get("parent_version")) != (digest,expected):
                    raise ValueError("registration inputs changed; explicit revision required")
                # Preserve the original attempt's reference and failure record.
                first=Path(previous["reference"]).resolve()
                references=list(dict.fromkeys([first]+references))[:MAX_ATTEMPTS]
            elif any(folder.glob("attempt_*")):
                raise ValueError("registration attempts lack input receipt; reconcile before retry")
        if any(ref.resolve() not in trusted for ref in references):
            raise ValueError("registration reference is not a committed observation in this scene")
        stamp={"source_sha256":digest,"source_l0":str(Path(l0).resolve()),
               "parent_version":expected,"l1_sha256":file_hash(original)}
        if not intent.exists():
            atomic_json(intent,{"source_sha256":digest,"parent_version":expected,"reference":str(references[0])})
        def execute(reference,attempt):
            from atas_geometry import GEOMETRY_ROOT as repo
            command=[str(settings.vista4d_python),"-m","atas_geometry.register.anchored_fusion",
                "--reference-l0",str(reference),"--new-l0",str(l0),"--l1",str(original),
                "--output",str(attempt),"--vista-repo",str(settings.vista4d_repo),"--model-id",settings.pi3_model_id]
            from atas_geometry.service.resident_models import resident_command
            command=resident_command(command,repo=repo,vista_repo=settings.vista4d_repo,
                                     log_path=folder/"reconstruction.log")
            from atas_geometry.service.device_mapping import gpu_selector
            env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=gpu_selector(gpu_index),CUDA_DEVICE_ORDER="PCI_BUS_ID",
                PYTHONUNBUFFERED="1",OPENBLAS_NUM_THREADS="4",PYTHONPATH=str(repo))
            with (folder/"reconstruction.log").open("a") as log:
                offset = log.tell()
                try:
                    subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                except subprocess.CalledProcessError:
                    log.flush()
                    from atas_geometry.service.resource_errors import log_resource_error
                    resource_error = log_resource_error(folder/"reconstruction.log", physical_gpu_index=gpu_index,
                        stage="anchored_reconstruction", start_offset=offset)
                    if resource_error:
                        raise resource_error
                    raise
        report,recovery=run_bounded(folder,stamp=stamp,references=references,execute=execute,
            recover_interrupted=bool(payload.get("recover_interrupted_registration")))
        if not registration_accepted(report):
            return dict(merged=False,registered=False,registration_status="rejected",l1_point_count=before,
                gpu_reserved_peak_mib=report.get("gpu_reserved_peak_mib"),
                world_version=expected,reason=report.get("reason","geometry_validation_failed"),
                fusion_version="anchored_world_v1",report_path=str(folder),
                registration_diagnosis=registration_diagnose(report),registration_recovery=recovery)
        observation=Path(report["conditioned_l0"])
        candidate=Path(report["candidate"]) if report.get("candidate") else original
        added=int(report.get("new_points",0))
        # Copy all per-instance masks as immutable hardlinks. The source RGB and
        # union dynamic masks already remain accessible through stable symlinks.
        for name in ("instance_tracks.json","dynamic_mask_by_instance","entity_masks.json","dynamic_mask_by_entity"):
            src=l0/name; dst=observation/name
            if src.exists() and not dst.exists():
                if src.is_dir(): shutil.copytree(src,dst,copy_function=os.link)
                else: os.link(src,dst)
    _metadata(observation,payload)
    if payload.get("publication_mode") == "reviewed_candidate_v1":
        from atas_geometry.register.trajectory_diagnostic import diagnose
        diagnostic=diagnose(settings.hot_root,payload,observation) if settings else {"status":"not_applicable"}
        atomic_json(folder/"trajectory_diagnostic.json",diagnostic)
        manifest={"schema_version":1,"job_id":job_id,"shot_id":shot,"scene_id":payload["scene_id"],
            "run_id":payload["run_id"],"story_id":payload["story_id"],
            "source_sha256":digest,"parent_version":expected,"geometry_accepted":True,
            "l0":str(observation.resolve().relative_to(scene.resolve())),
            "l1":str(candidate.resolve().relative_to(scene.resolve())),"novel_points":added,
            "home":bool(home_fuse),"lineage":payload.get("review_lineage",{})}
        atomic_json(folder/"candidate.json",manifest)
        return dict(merged=False,registered=True,registration_status="candidate",publication_status="pending_review",
            gpu_reserved_peak_mib=report.get("gpu_reserved_peak_mib"),
            candidate_job_id=job_id,novel_points=added,l1_point_count=_count(candidate),
            world_version=expected,accepted_l0=str(observation),l1_npz=str(candidate),
            trajectory_diagnostic=diagnostic,
            fusion_version="anchored_world_v1")
    committed,published=commit_world(scene,version=version,l1=candidate,observation=observation,
        shot_id=shot,source_sha256=digest,expected_version=expected)
    return dict(merged=bool(added),registered=True,registration_status="registered",home=bool(home_fuse),
        novel_points=added,l1_point_count=_count(candidate),world_version=committed["version"],
        accepted_l0=str(observation),l1_npz=str(candidate),commit_published=published,
        transform={"scale":1.,"rotation":np.eye(3).tolist(),"translation":[0.,0.,0.]},
        fusion_version="anchored_world_v1")
