"""Bounded GPU reconstruction of rejected intervals using trusted neighbours.

No interpolation, threshold relaxation, temporal subsampling or publication here.
All replacement RGB-D frames come from Pi3X and pass independent held-out checks.
"""
from pathlib import Path
import os
import time
import numpy as np


def trusted_frame_indices(owner, reports):
    accepted = {r['chunk'] for r in reports if r.get('accepted_geometry') is True}
    return [i for i, chunk in enumerate(owner) if chunk in accepted]


def support_split(targets, trusted, attempt=0, width=4):
    """Each available side contributes separate conditioning and test frames."""
    lo, hi = min(targets), max(targets)
    before = sorted(i for i in trusted if i < lo)
    after = sorted(i for i in trusted if i > hi)
    offset = attempt * width
    left = before[max(0, len(before)-offset-width):len(before)-offset] if offset < len(before) else []
    right = after[offset:offset+width]
    # A lone support cannot be both a prior and an independent seam check.
    sides = [side for side in (left, right) if len(side) >= 2]
    prior = sorted(i for side in sides for i in side[::2])
    # Even when the second attempt uses farther priors, retain independent
    # checks at the immediate joins; do not move the test away from the seam.
    test = sorted(set(i for side in sides+[before[-width:],after[:width]]
                      if len(side)>=2 for i in side[1::2])-set(prior))
    return prior, test


def target_frames(record, owner):
    return [i for i in range(record['start'], record['end'])
            if owner[i] in (-1, record['chunk'])]


def context_frames(targets, priors, tests, trusted, attempt, limit=24):
    """Add nearby RGB-only views on retry, without changing write ownership.

    Short final chunks lack the motion context of normal reconstruction windows.
    Keep the model input bounded; every prior/test always remains present.
    """
    required=set(targets+priors+tests)
    if attempt==0:return sorted(required)
    candidates=sorted(set(trusted)-required,
                      key=lambda i:(min(abs(i-min(targets)),abs(i-max(targets))),i))
    return sorted(required|set(candidates[:max(0,limit-len(required))]))


def repair_checks_pass(row, tolerance, expected_tests):
    seams=row.get('seam_checks',[])
    return (row.get('heldout_count',0)>=100 and row.get('heldout_median') is not None
            and row['heldout_median']<=tolerance and row.get('heldout_within_2t',0)>=.65
            and sorted(s['frame'] for s in seams)==sorted(expected_tests)
            and bool(expected_tests)
            and all(s.get('count',0)>100 and s.get('median') is not None and s['median']<=2*tolerance for s in seams)
            and row.get('pinhole_residual_pixels',float('inf'))<=2.)


def balanced_anchor_fit(sources, targets, history_count):
    """Equal weight to historical and temporal priors; trim inside each view.

Pooled robust trimming may remove the entire smaller temporal family. Sampling
here is ONLY for fitting one Sim(3), never for storing source/dynamic pixels.
"""
    from atas_geometry.register.anchored_fusion import fit_sim3, apply_sim3, AnchorFitError
    pairs=[]
    for source,target in zip(sources,targets):
        valid=np.isfinite(source).all(-1)&np.isfinite(target).all(-1)
        pairs.append((source[valid],target[valid]))
    sources=[s for s,t in pairs];targets=[t for s,t in pairs]
    families=[list(zip(sources[:history_count],targets[:history_count])),
              list(zip(sources[history_count:],targets[history_count:]))]
    groups=[]
    for family in families:
        family=[(s,t) for s,t in family if len(s)>=6]
        if not family:raise AnchorFitError('missing historical or temporal fit support')
        budget=max(6,2048//len(family))
        for source,target in family:
            ids=np.linspace(0,len(source)-1,min(budget,len(source))).astype(int)
            groups.append((source[ids],target[ids]))
    masks=[np.ones(len(s),bool) for s,t in groups]
    for _ in range(8):
        tr=fit_sim3(np.concatenate([s[m] for (s,t),m in zip(groups,masks)]),
                    np.concatenate([t[m] for (s,t),m in zip(groups,masks)]))
        updated=[]
        for source,target in groups:
            error=np.linalg.norm(apply_sim3(source,tr)-target,axis=1)
            updated.append(error<=max(float(np.quantile(error,.65)),1e-7))
        if all(np.array_equal(a,b) for a,b in zip(masks,updated)):break
        masks=updated
    return fit_sim3(np.concatenate([s[m] for (s,t),m in zip(groups,masks)]),
                    np.concatenate([t[m] for (s,t),m in zip(groups,masks)]))


def repair_conditioned_chunks(*, reference, new, output, reports, owner, poses_out,
                              k_out, model, preprocess, anchor_rgb, anchor_ids,
                              history_depth, history_points, history_k, heldout,
                              tolerance, pixel_limit, max_attempts=2):
    import cv2
    import torch
    import torch.nn.functional as F
    from atas_geometry.register.anchored_fusion import (
        static_depth, lift_depth, robust_sim3, apply_sim3, fit_intrinsics,
        write_exr, write_json, AnchorFitError)
    from atas_geometry.memory.fuse import _install_depth_reader, _read_depth_any
    _install_depth_reader()
    output = Path(output)
    h, w = new.frames[0].shape[:2]
    ph, pw = history_depth.shape[1:]
    nk = len(anchor_ids)
    scale_k = np.array([pw/w, ph/h, pw/w, ph/h])
    ledger = []

    def stored_frame(frame):
        # Read the actual retained geometry, not a rejected overlapping prediction.
        depth = _read_depth_any(output/'depths'/f'{frame:05d}.exr')
        if depth.ndim == 3: depth = depth[..., 0]
        depth = cv2.resize(depth, (pw,ph), interpolation=cv2.INTER_NEAREST)
        static = cv2.resize((static_depth(new,frame)>0).astype(np.uint8),(pw,ph),interpolation=cv2.INTER_NEAREST).astype(bool)
        confidence = np.load(output/'confidence'/f'{frame:05d}.npy')
        valid = static & np.isfinite(depth) & (depth>0) & (confidence>.2)
        k = k_out[frame]*scale_k
        return dict(depth=depth,valid=valid,pose=poses_out[frame].copy(),k=k,
                    points=lift_depth(depth,poses_out[frame],k))

    for record in reports:
        if record.get('accepted_geometry'): continue
        targets = target_frames(record,owner)
        if not targets: continue
        previous_evidence = set()
        for attempt_index in range(max_attempts):
            trusted = trusted_frame_indices(owner,reports)
            priors, tests = support_split(targets,trusted,attempt_index)
            if not priors or not tests: continue
            signature = (tuple(priors),tuple(tests))
            if signature in previous_evidence: continue
            previous_evidence.add(signature)
            supports = {i:stored_frame(i) for i in priors+tests}
            ids = context_frames(targets,priors,tests,trusted,attempt_index)
            slots = {frame:nk+i for i,frame in enumerate(ids)}
            folder = output/'local_repairs'/f"chunk_{record['chunk']:03d}_{attempt_index:02d}"
            folder.mkdir(parents=True,exist_ok=False)
            tick=time.perf_counter()
            row=dict(chunk=record['chunk'],attempt=attempt_index,target_frames=targets,
                     prior_frames=priors,test_frames=tests,accepted=False,
                     rgb_context_frames=ids,
                     evidence_basis='accepted_frame_owner_only',thresholds_changed=False)
            # Full original history; held-out history and seam frames are RGB-only.
            rgb,_=preprocess(np.stack([new.frames[i] for i in ids]),pixel_limit)
            images=torch.cat((anchor_rgb,rgb),0)[None].to('cuda',dtype=torch.bfloat16)
            n=nk+len(ids)
            depth_in=np.zeros((n,ph,pw),np.float32)
            pose_in=np.repeat(np.eye(4,dtype=np.float32)[None],n,axis=0)
            k_in=np.repeat(np.eye(3,dtype=np.float32)[None],n,axis=0)
            masks=np.zeros((1,n),bool)
            for j in np.flatnonzero(~heldout):
                depth_in[j]=history_depth[j];pose_in[j]=reference.cam_c2w[anchor_ids[j]]
                fx,fy,cx,cy=history_k[j];k_in[j]=[[fx,0,cx],[0,fy,cy],[0,0,1]];masks[0,j]=True
            for frame in priors:
                s=supports[frame];j=slots[frame]
                depth_in[j]=np.where(s['valid'],s['depth'],0);pose_in[j]=s['pose']
                fx,fy,cx,cy=s['k'];k_in[j]=[[fx,0,cx],[0,fy,cy],[0,0,1]];masks[0,j]=True
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                pred=model(images,depths=torch.from_numpy(depth_in[None]).to('cuda'),
                    poses=torch.from_numpy(pose_in[None]).to('cuda'),intrinsics=torch.from_numpy(k_in[None]).to('cuda'),
                    mask_add_depth=torch.from_numpy(masks.copy()).to('cuda'),
                    mask_add_pose=torch.from_numpy(masks.copy()).to('cuda'),
                    mask_add_ray=torch.from_numpy(masks.copy()).to('cuda'),with_prior=True)
            local=pred['local_points'][0].float().cpu().numpy()
            world=pred['points'][0].float().cpu().numpy()
            poses=pred['camera_poses'][0].float().cpu().numpy()
            conf=pred['conf'][0].sigmoid().float().cpu().numpy()[...,0]
            del pred,images
            fit_src=[];fit_dst=[];test_src=[];test_dst=[]
            for j in range(nk):
                valid=(history_depth[j]>0)&np.isfinite(world[j]).all(-1)&(conf[j]>.1)
                sample=np.flatnonzero(valid)[::8]
                (test_src if heldout[j] else fit_src).append(world[j].reshape(-1,3)[sample])
                (test_dst if heldout[j] else fit_dst).append(history_points[j].reshape(-1,3)[sample])
            history_fit_count=len(fit_src)
            for frame in priors:
                s=supports[frame];j=slots[frame]
                sample=np.flatnonzero(s['valid']&np.isfinite(world[j]).all(-1)&(conf[j]>.1))[::8]
                fit_src.append(world[j].reshape(-1,3)[sample]);fit_dst.append(s['points'].reshape(-1,3)[sample])
            try:
                tr=balanced_anchor_fit(fit_src,fit_dst,history_fit_count)
                # Diagnostic comparison must not prevent the actual repair fit.
                pooled=None
                try:
                    src=np.concatenate(fit_src);dst=np.concatenate(fit_dst)
                    finite=np.isfinite(src).all(-1)&np.isfinite(dst).all(-1)
                    pooled=robust_sim3(src[finite],dst[finite])
                except (ValueError,np.linalg.LinAlgError):
                    pass
                row['fit_groups']=[dict(family='history' if i<history_fit_count else 'temporal',count=len(s),
                    pooled_median=float(np.median(np.linalg.norm(apply_sim3(s,pooled)-t,axis=1))) if len(s) and pooled is not None else None,
                    balanced_median=float(np.median(np.linalg.norm(apply_sim3(s,tr)-t,axis=1))) if len(s) else None)
                    for i,(s,t) in enumerate(zip(fit_src,fit_dst))]
                err=np.linalg.norm(apply_sim3(np.concatenate(test_src),tr)-np.concatenate(test_dst),axis=1)
                scale,rotation,translation=tr
                poses[:,:3,3]=apply_sim3(poses[:,:3,3],tr)
                poses[:,:3,:3]=rotation[None]@poses[:,:3,:3]
                seam=[];low={};rays=[]
                for frame in ids:
                    j=slots[frame];k,ray=fit_intrinsics(local[j]);rays.append(ray)
                    depth=local[j,...,2]*scale
                    low[frame]=(depth,k,poses[j],conf[j])
                    if frame in tests:
                        points=lift_depth(depth,poses[j],k);s=supports[frame]
                        good=s['valid']&(conf[j]>.2)&np.isfinite(points).all(-1)
                        residual=float(np.median(np.linalg.norm(points[good]-s['points'][good],axis=-1))) if good.sum()>100 else None
                        seam.append(dict(frame=frame,count=int(good.sum()),median=residual))
                row.update(heldout_count=len(err),heldout_median=float(np.median(err)) if len(err) else None,
                    heldout_within_2t=float(np.mean(err<=2*tolerance)) if len(err) else 0,
                    seam_checks=seam,pinhole_residual_pixels=float(max(rays)))
                passed=repair_checks_pass(row,tolerance,tests)
            except (ValueError,np.linalg.LinAlgError) as exc:
                row['error']=str(exc);passed=False
            if passed:
                # Stage complete target frames before replacing anything. Good
                # frame owners are immutable. The derived L0 is not published yet.
                for frame in targets:
                    depth,k,pose,confidence=low[frame]
                    full=F.interpolate(torch.from_numpy(depth)[None,None],(h,w),mode='bilinear',align_corners=False)[0,0].numpy()
                    write_exr(folder/f'{frame:05d}.exr',full)
                    np.save(folder/f'{frame:05d}.npy',confidence.astype(np.float16))
                for frame in targets:
                    old_depth=output/'depths'/f'{frame:05d}.exr'
                    if old_depth.exists():os.link(old_depth,folder/f'{frame:05d}.before.exr')
                    os.replace(folder/f'{frame:05d}.exr',old_depth)
                    os.replace(folder/f'{frame:05d}.npy',output/'confidence'/f'{frame:05d}.npy')
                    _,k,pose,_=low[frame];poses_out[frame]=pose;k_out[frame]=k/scale_k;owner[frame]=record['chunk']
                baseline=dict(record)
                record.update(accepted_geometry=True,local_repair=True,baseline=baseline,
                    heldout_median=row['heldout_median'],heldout_within_2t=row['heldout_within_2t'],
                    seam_median=max(s['median'] for s in seam),pinhole_residual_pixels=row['pinhole_residual_pixels'],
                    repair_report=str(folder/'report.json'))
                row['accepted']=True
            row['seconds']=time.perf_counter()-tick
            write_json(folder/'report.json',row);ledger.append(row)
            write_json(output/'local_repair_progress.json',dict(attempts=ledger))
            print('LOCAL_REPAIR '+__import__('json').dumps(row),flush=True)
            if passed:break
    return dict(policy='trusted_temporal_neighbours_v1',attempts=ledger,
                repaired_chunks=[r['chunk'] for r in reports if r.get('local_repair')],
                unresolved_chunks=[r['chunk'] for r in reports if not r.get('accepted_geometry')],
                thresholds_changed=False,max_attempts_per_chunk=max_attempts)
