"""Reconstruction-based camera comparison; never a film/writeback acceptance gate."""
import json
from pathlib import Path
import numpy as np
from atas_geometry.world.recapture import identifier
def compare_cameras(planned,actual):
    planned=np.asarray(planned,float);actual=np.asarray(actual,float)
    if planned.shape!=actual.shape or planned.ndim!=3 or planned.shape[1:]!=(4,4) or len(planned)<2:
        raise ValueError('incompatible camera samples')
    if not np.isfinite(planned).all() or not np.isfinite(actual).all():raise ValueError('nonfinite camera samples')
    a=planned[-1,:3,3]-planned[0,:3,3];b=actual[-1,:3,3]-actual[0,:3,3]
    travel_a=float(np.linalg.norm(a));travel_b=float(np.linalg.norm(b))
    rotations=np.einsum('nji,njk->nik',planned[:,:3,:3],actual[:,:3,:3])
    angle=np.degrees(np.arccos(np.clip((np.trace(rotations,axis1=1,axis2=2)-1)/2,-1,1)))
    return {'planned_net_travel':travel_a,'actual_net_travel':travel_b,
        'net_travel_ratio':travel_b/travel_a if travel_a>1e-8 else None,
        'net_direction_cosine':float(np.dot(a,b)/(travel_a*travel_b)) if min(travel_a,travel_b)>1e-8 else None,
        'median_position_error':float(np.median(np.linalg.norm(actual[:,:3,3]-planned[:,:3,3],axis=1))),
        'median_orientation_error_degrees':float(np.median(angle))}


def diagnose(root,payload,l0):
    base={'acceptance_gate':False,'coordinate_units':'uncalibrated_scene_units',
        'basis':'already_scene_registered_reconstruction_no_camera_refit','not_ground_truth':True}
    jid=payload.get('review_lineage',{}).get('target_render_job_id')
    if not jid:return {**base,'status':'not_applicable','reason':'no_target_render'}
    try:
        jid=identifier(jid,'target_render_job_id');folder=Path(root)/'jobs'/jid
        job=json.loads((folder/'job.json').read_text()); scope=job['payload']
        if any(scope.get(k)!=payload.get(k) for k in ('run_id','story_id','scene_id','shot_id')):
            return {**base,'status':'unavailable','reason':'render_from_different_scope'}
        meta=json.loads((Path(l0)/'observation.json').read_text())
        if meta.get('coordinate_space')!='scene_world':raise ValueError('observation not in scene coordinates')
        with np.load(folder/'target_camera.npz') as data:
            planned=np.array(data['cam_c2w']);times=np.array(data['world_times'],float)
        with np.load(Path(l0)/'cameras.npz') as data: actual=np.array(data['cam_c2w'])
        times=times-times[0]
        offset=(int(meta['original_frame_count'])-len(actual))//2
        indices=np.rint(times*float(meta['source_fps'])).astype(int)-offset
        if np.any(indices<0) or np.any(indices>=len(actual)):raise ValueError('target times outside reconstructed film')
        metrics=compare_cameras(planned,actual[indices])
        return {**base,'status':'measured','target_render_job_id':jid,'sample_count':len(indices),**metrics}
    except (ValueError,KeyError,OSError) as exc:
        return {**base,'status':'unavailable','reason':str(exc)}
