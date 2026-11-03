"""Rebase an immutable candidate using the existing dense reconstruction cache."""
from atas_geometry.world.reviewed_world import candidate_source
from atas_geometry.world.world_state import load_world
from atas_geometry.service.durable import file_hash
def submit_rebase(runtime, payload):
    scene, candidate, _ = candidate_source(runtime.settings.hot_root, payload)
    with runtime.scene_lock(candidate['run_id'], candidate['story_id'], candidate['scene_id']):
        parent = (load_world(scene) or {}).get('version')
        if parent == candidate['parent_version']:
            raise ValueError('candidate parent has not changed; retry publication, not reconstruction')
        original = runtime.store.get(candidate['job_id'])
        if not original or original['kind'] != 'recon':
            raise ValueError('candidate reconstruction job is unavailable')
        source = dict(original['payload'])
        if any(source.get(k) != candidate[k] for k in ('run_id','story_id','scene_id','shot_id')):
            raise ValueError('candidate job scope differs')
        if source.get('video_sha256') != candidate['source_sha256'] or file_hash(source['video_path']) != candidate['source_sha256']:
            raise ValueError('candidate source video changed')
        # The new fingerprint creates a fresh derived candidate, not a new film
        # or raw observation. revision_l0/dense_manifest reuse validated L0.
        source.update(rebase_source_candidate=candidate['job_id'],
                      rebase_parent_version=parent, retry_failed=True)
        job = runtime.submit('recon', source)
        return {'job_id': job['job_id'], 'source_candidate': candidate['job_id'],
                'target_parent_version': parent}
