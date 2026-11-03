import json
import numpy as np
import pytest
from atas_geometry.service.durable import atomic_json
from atas_geometry.world.reviewed_world import publish_candidate, candidate_source
from atas_geometry.world.world_state import load_world, current_l1
from atas_geometry.service.paths import l0_dir
def fixture(root,registry=None):
    p=dict(run_id='run',story_id='story',scene_id='room',shot_id='shot',candidate_job_id='recon_test')
    scene=root/'run/story/scenes/room'; folder=scene/'.world_versions/recon_test'; obs=folder/'l0'
    obs.mkdir(parents=True)
    np.savez(folder/'l1.npz',xyz=np.ones((3,3)))
    atomic_json(obs/'observation.json',{'registered':True,'scene_id':'room','entity_registry':registry or []})
    data={**p,'job_id':'recon_test','geometry_accepted':True,'l0':'.world_versions/recon_test/l0',
        'l1':'.world_versions/recon_test/l1.npz','source_sha256':'video','parent_version':None,
        'home':True,'lineage':{'target_render_job_id':None},'novel_points':3}
    atomic_json(folder/'candidate.json',data)
    p['film_review']={'schema_version':1,'stage':'film','decision':'warning','cache_key':'review',
        'video_sha256':'video','lineage':data['lineage']}
    return p,scene,obs


def test_candidate_invisible_until_commit_and_replay_idempotent(tmp_path):
    p,scene,obs=fixture(tmp_path)
    assert load_world(scene) is None and not current_l1(scene).exists()
    assert l0_dir(tmp_path,'run','story','shot')!=obs
    a=publish_candidate(tmp_path,p); b=publish_candidate(tmp_path,p)
    assert a['commit_published'] and not b['commit_published']
    assert len(load_world(scene)['observations'])==1
    assert l0_dir(tmp_path,'run','story','shot')==obs


@pytest.mark.parametrize('field,value',[('decision','hold'),('video_sha256','wrong'),('stage','preview'),('lineage',{})])
def test_review_failure_cannot_publish(tmp_path,field,value):
    p,scene,_=fixture(tmp_path);p['film_review'][field]=value
    with pytest.raises(ValueError): publish_candidate(tmp_path,p)
    assert load_world(scene) is None


def test_missing_identity_does_not_block_geometrically_accepted_publication(tmp_path):
    p,scene,_=fixture(tmp_path,[{'entity_id':'rabbit'}])
    assert publish_candidate(tmp_path,p)['commit_published']
    assert load_world(scene) is not None


def test_stale_candidate_does_not_replace_world(tmp_path):
    p,scene,_=fixture(tmp_path)
    atomic_json(scene/'world_state.json',{'version':'newer','observations':{},'l1':'other'})
    with pytest.raises(RuntimeError,match='world changed'):publish_candidate(tmp_path,p)
    assert load_world(scene)['version']=='newer'


@pytest.mark.parametrize('invalid',[None,'scene','hash','geometry'])
def test_best_available_hold_is_not_pass_but_can_publish_accepted_geometry(tmp_path,invalid):
    p,scene,obs=fixture(tmp_path,[{'entity_id':'rabbit'}])
    p['film_review']['decision']='hold'
    r=p['film_review']
    p['film_selection']=dict(schema='best_available_v1',cache_key='selection',
        budget_exhausted=True,audit_passed=False,same_scene=True,video_sha256='video',
        original_decision='hold',review_cache_key='review',actual_event='hand stayed down',
        candidate_count=5,repair_budget=4,selected_index=0,
        candidates=[dict(sha256='video',review=r) for _ in range(5)])
    if invalid=='scene':p['film_selection']['same_scene']=False
    if invalid=='hash':p['film_selection']['video_sha256']='tampered'
    if invalid=='geometry':
        path=scene/'.world_versions/recon_test/candidate.json'
        data=json.loads(path.read_text());data['geometry_accepted']=False;atomic_json(path,data)
    if invalid:
        with pytest.raises(ValueError):publish_candidate(tmp_path,p)
        assert load_world(scene) is None
    else:
        assert publish_candidate(tmp_path,p)['commit_published']
        metadata=json.loads((obs/'observation.json').read_text())
        assert metadata['film_audit_decision']=='hold'
        assert metadata['actual_event']=='hand stayed down'
        assert not metadata['scripted_event_verified']


def test_candidate_cannot_cross_scope(tmp_path):
    p,_,_=fixture(tmp_path);p['shot_id']='other'
    with pytest.raises(ValueError,match='scope'): candidate_source(tmp_path,p)


def test_home_preparation_never_publishes(tmp_path,monkeypatch):
    from atas_geometry.world.writeback import writeback_world
    import atas_geometry.world.writeback as module
    scene=tmp_path/'run/story/scenes/room'; source=tmp_path/'raw';source.mkdir()
    atomic_json(source/'observation.json',{'registered':False})
    candidate=scene/'.world_versions/recon_test/home_static.npz';candidate.parent.mkdir(parents=True)
    np.savez(candidate,xyz=np.ones((3,3)))
    monkeypatch.setattr(module,'_metadata',lambda l0,payload:atomic_json(l0/'observation.json',{'registered':True}))
    p=dict(run_id='run',story_id='story',shot_id='shot',scene_id='room',video_sha256='video',publication_mode='reviewed_candidate_v1')
    result=writeback_world(source,scene,payload=p,job_id='recon_test',settings=None,gpu_index=5,home_l0=source,home_fuse={'candidate_l1':str(candidate)})
    assert result['publication_status']=='pending_review'
    assert not (scene/'l1.npz').exists() and load_world(scene) is None
    assert not json.loads((source/'observation.json').read_text())['registered']


def test_interrupted_commit_can_retry(tmp_path,monkeypatch):
    import atas_geometry.world.world_state as module
    p,scene,_=fixture(tmp_path); original=module.atomic_json
    def crash(path,data):
        if path.name=='world_state.json': raise OSError('interrupted')
        original(path,data)
    monkeypatch.setattr(module,'atomic_json',crash)
    with pytest.raises(OSError):publish_candidate(tmp_path,p)
    assert load_world(scene) is None
    monkeypatch.setattr(module,'atomic_json',original)
    assert publish_candidate(tmp_path,p)['commit_published']


def test_camera_diagnostic_keeps_scale_deviation_without_refitting():
    from atas_geometry.register.trajectory_diagnostic import compare_cameras
    target=np.repeat(np.eye(4)[None],3,axis=0);target[:,2,3]=[0,.5,1.]
    actual=target.copy();actual[:,2,3]*=2
    result=compare_cameras(target,actual)
    assert result['net_travel_ratio']==2 and result['net_direction_cosine']==1
    assert result['median_position_error']==.5


def test_locked_camera_has_no_infinite_ratio():
    from atas_geometry.register.trajectory_diagnostic import compare_cameras
    target=np.repeat(np.eye(4)[None],3,axis=0)
    result=compare_cameras(target,target)
    assert result['net_travel_ratio'] is None and result['median_position_error']==0
