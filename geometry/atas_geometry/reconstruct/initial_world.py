"""Repair evidenced temporal drift before a new scene has a published world.

No existing L1 or accepted observation is edited. Reuses dense source RGB/masks,
and re-estimates depth/poses together against the first unchained VO window.
Image correspondences used for verification never fit a correction transform.
"""
from __future__ import annotations
import copy
import itertools
import json
from pathlib import Path
import time
import numpy as np

POLICY = 'initial-world-fixed-anchor-v3'


def temporal_evidence(source):
    import cv2
    from atas_geometry.register.matching import _lift_uv_robust
    from atas_geometry.register.anchored_fusion import static_depth
    from atas_geometry.register.matching import evidence_frames
    ids = evidence_frames(source.count)
    detector = cv2.SIFT_create(nfeatures=5000)
    matcher = cv2.BFMatcher()
    features = {}
    depths = []
    for i in ids:
        rgb = source.frames[i]
        depth = static_depth(source, i)
        depths.append(depth[depth > 0])
        mask = (depth > 0).astype(np.uint8)*255
        features[i] = detector.detectAndCompute(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),mask)
    positive = np.concatenate(depths)
    if not len(positive):
        return [], None
    tolerance = .02*float(np.median(positive))
    pairs = []
    for a,b in itertools.combinations(ids,2):
        ka,da=features[a]; kb,db=features[b]
        if da is None or db is None or min(len(da),len(db))<2: continue
        forward=[p[0] for p in matcher.knnMatch(da,db,k=2) if len(p)==2 and p[0].distance<.72*p[1].distance]
        reverse={p[0].queryIdx:p[0].trainIdx for p in matcher.knnMatch(db,da,k=2) if len(p)==2 and p[0].distance<.72*p[1].distance}
        matches=[m for m in forward if reverse.get(m.trainIdx)==m.queryIdx]
        if len(matches)<100: continue
        uv=np.array([ka[m.queryIdx].pt for m in matches]); vv=np.array([kb[m.trainIdx].pt for m in matches])
        cv2.setRNGSeed(0)
        matrix,inliers=cv2.findFundamentalMat(uv,vv,cv2.USAC_MAGSAC,1.5,.999,10000)
        if matrix is None or inliers is None: continue
        s,sv,_=_lift_uv_robust(source,a,uv); t,tv,_=_lift_uv_robust(source,b,vv)
        valid=sv&tv&inliers.ravel().astype(bool)
        if valid.sum()<100: continue
        error=np.linalg.norm(s[valid]-t[valid],axis=1)
        pairs.append(dict(frames=[a,b],uv=uv[valid],vv=vv[valid],error=error,
                          count=len(error),median=float(np.median(error))))
    return pairs,tolerance


GROSS = 4      # one pair beyond GROSS x (2 x tolerance) is drift on its own
MIN_PAIRS = 3
MIN_COVERAGE = .8


def coverage(pairs, count):
    """Fraction of the clip's time span bridged by supported frame pairs."""
    if count <= 1:
        return 1.
    covered = np.zeros(count - 1, bool)
    for p in pairs:
        if p['count'] >= 100:
            a, b = sorted(p['frames'])
            covered[a:b] = True
    return float(covered.mean())


def assess(pairs, tolerance, count):
    """Classify a first observation from fixed-coordinate pair disagreement.

    drift: two supported pairs beyond 2x tolerance, or one beyond GROSS times that.
    insufficient: too few supported pairs, or pairs that leave part of the clip
    unchecked -- absence of evidence is not evidence of consistency.
    """
    supported = [p for p in pairs if p['count'] >= 100]
    drifted = [p for p in supported if tolerance is not None and p['median'] > 2*tolerance]
    gross = [p for p in drifted if p['median'] > GROSS*2*tolerance]
    span = coverage(supported, count)
    return dict(supported_pairs=len(supported), drifted_pairs=len(drifted), gross_pairs=len(gross),
                coverage=span, drift=len(drifted) >= 2 or bool(gross),
                sufficient=tolerance is not None and len(supported) >= MIN_PAIRS and span >= MIN_COVERAGE)


def verify_candidate(candidate,pairs,tolerance):
    """Re-measure the original correspondences in the repaired world, no refit.

    Every pair keeps its evidence and ends within the drift line (2x tolerance):
    drifted pairs must be resolved, consistent pairs must stay consistent.
    A small change of an already-consistent pair is not a regression.
    """
    from atas_geometry.register.matching import _lift_uv_robust
    checks=[]
    for p in pairs:
        a,b=p['frames']
        s,sv,_=_lift_uv_robust(candidate,a,p['uv']); t,tv,_=_lift_uv_robust(candidate,b,p['vv'])
        valid=sv&tv
        error=np.linalg.norm(s[valid]-t[valid],axis=1)
        count=len(error)
        median=float(np.median(error)) if count else None
        fraction=float(np.mean(error<=2*tolerance)) if count else 0.
        checks.append(dict(frames=p['frames'],count=count,original_count=p['count'],
            before_median=p['median'],median=median,within_2t=fraction,
            passed=bool(count>=100 and count>=.8*p['count'] and median<=2*tolerance and fraction>=.65)))
    return checks, all(c['passed'] for c in checks)


def prepare_initial_world(source_path,folder,*,vista_repo,model_id,first_window=48,pixel_limit=255000):
    from atas_geometry.register.anchored_fusion import strict_store, reconstruct_conditioned, sha256, write_json
    tick=time.perf_counter()
    source_path=Path(source_path).resolve(); folder=Path(folder).resolve()
    source=strict_store(source_path)
    stamp=dict(policy=POLICY,source=str(source_path.resolve()),video=sha256(source_path/'video.mp4'),
        cameras=sha256(source_path/'cameras.npz'),frames=source.count,first_window=first_window,
        pixel_limit=pixel_limit,model_id=str(model_id),
        dense_manifest=sha256(source_path/'dense_complete.json') if (source_path/'dense_complete.json').is_file() else None)
    receipt=folder/'initial_world.json'
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if old['input']!=stamp: raise ValueError('initial-world repair input changed')
        if old['status']=='repaired':
            result=strict_store(old['l0'])
            if result.count!=source.count: raise ValueError('initial-world candidate incomplete')
        return old
    folder.mkdir(parents=True,exist_ok=True)
    intent=folder/'input.json'
    if intent.exists() and json.loads(intent.read_text())!=stamp:
        raise ValueError('interrupted initial-world repair input changed')
    write_json(intent,stamp)
    pairs,tolerance=temporal_evidence(source)
    assessment=assess(pairs,tolerance,source.count)
    out=dict(input=stamp,tolerance=tolerance,source=str(source_path),l0=str(source_path),
        before=[{k:p[k] for k in ('frames','count','median')} for p in pairs],assessment=assessment,
        published=False)
    if assessment['sufficient'] and not assessment['drift']:
        out.update(status='unchanged',reason='consistent_with_sufficient_evidence')
    elif first_window<=0 or source.count<=first_window:
        # Too short for a fixed-anchor re-estimate; never publish unverified drift.
        out.update(status='rejected' if assessment['drift'] else 'unchanged',
                   reason='drift_in_single_window_clip' if assessment['drift'] else 'single_window_clip')
    else:
        # Drift, or too little evidence to call the clip consistent: re-estimate it
        # against its own first window as a fixed anchor. The anchored reconstruction
        # applies the same held-out/seam validation as any later registration, and
        # every available pair is re-measured on top.
        reference=copy.copy(source);reference.count=min(first_window,source.count)
        target=folder/'conditioned_l0'
        # Interrupted candidate is retained, never treated as a completed cache.
        report_path=target/'anchored_reconstruction.json'
        if report_path.exists():
            report=json.loads(report_path.read_text())
        else:
            if target.exists(): target.rename(folder/f'conditioned_interrupted_{time.time_ns()}')
            report=reconstruct_conditioned(reference,source,target,model_id=model_id,
                vista_repo=Path(vista_repo),pixel_limit=pixel_limit,local_repair=True)
        checks,passed=verify_candidate(strict_store(target),pairs,tolerance)
        good=bool(report['geometry_accepted'] and report['frame_count']==source.count and passed)
        out.update(status='repaired' if good else 'rejected',after=checks,
            reason=('drift_resolved' if assessment['drift'] else 'verified_by_anchored_reconstruction') if good
                   else ('drift_unresolved' if assessment['drift'] else 'consistency_unverifiable'),
            internal_geometry_accepted=report['geometry_accepted'],candidate_l0=str(target))
        if good:
            # Keep instance masks/time metadata, not just their dynamic union.
            for name in ('instance_tracks.json','dynamic_mask_by_instance','dynamic_mask_by_entity',
                         'dense_complete.json','reconstruction_timings.json'):
                if (source_path/name).exists() and not (target/name).exists():
                    (target/name).symlink_to((source_path/name).resolve())
            out['l0']=str(target)
    out['seconds']=time.perf_counter()-tick
    write_json(receipt,out)
    return out


def run_initial_world(source,folder,*,settings,gpu_index):
    """Service process delegates GPU work to the existing Vista runtime."""
    import os,subprocess
    from atas_geometry.reconstruct.vista4d import effective_pi3_settings
    pixel_limit, first_window = effective_pi3_settings(
        settings.pi3_pixel_limit, settings.pi3_vo_chunk_size, settings.low_vram)
    legacy_receipt = Path(source) / 'dense_complete.json'
    if legacy_receipt.is_file():
        contract = json.loads(legacy_receipt.read_text()).get('reconstruction_contract', {})
        if all(key in contract for key in ('pi3_pixel_limit', 'pi3_vo_chunk_size', 'low_vram')):
            pixel_limit, first_window = effective_pi3_settings(
                int(contract['pi3_pixel_limit']), int(contract['pi3_vo_chunk_size']),
                str(contract['low_vram']).lower() == 'true')
    # Prefer the actual source settings when recorded, not today's settings.
    actual_path = Path(source) / 'reconstruction_effective.json'
    if actual_path.is_file():
        actual = json.loads(actual_path.read_text())
        pixel_limit = int(actual['pi3_pixel_limit'])
        first_window = int(actual['pi3_vo_chunk_size'])
    from atas_geometry import GEOMETRY_ROOT as root
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    command=[str(settings.vista4d_python),'-m','atas_geometry.reconstruct.initial_world',str(source),str(folder),
        '--vista-repo',str(settings.vista4d_repo),'--model-id',settings.pi3_model_id,
        '--first-window',str(first_window),'--pixel-limit',str(pixel_limit)]
    from atas_geometry.service.device_mapping import gpu_selector
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu_selector(gpu_index),CUDA_DEVICE_ORDER='PCI_BUS_ID',
        OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1',
        PYTHONPATH=str(root))
    with (folder/'initial_world.log').open('a') as log:
        offset=log.tell()
        try: subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        except subprocess.CalledProcessError:
            log.flush()
            from atas_geometry.service.resource_errors import log_resource_error
            error=log_resource_error(folder/'initial_world.log',physical_gpu_index=gpu_index,
                                     stage='initial_world_alignment',start_offset=offset)
            if error: raise error
            raise
    result=json.loads((folder/'initial_world.json').read_text())
    if result['status']=='rejected':
        raise ValueError(f'initial world rejected ({result.get("reason")}); raw L0 and candidate retained at {folder}')
    return Path(result['l0']),result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('folder')
    p.add_argument('--vista-repo',required=True);p.add_argument('--model-id',required=True)
    p.add_argument('--first-window',type=int,default=48);p.add_argument('--pixel-limit',type=int,default=255000)
    args=p.parse_args();print(json.dumps(prepare_initial_world(args.source,args.folder,
        vista_repo=args.vista_repo,model_id=args.model_id,first_window=args.first_window,pixel_limit=args.pixel_limit)))
