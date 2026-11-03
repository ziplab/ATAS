import sys
from pathlib import Path
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[1]/'backbones'))
import numpy as np,pytest,torch
from atas_geometry.memory.gpu_voxel_index import IncrementalKeys, TemporalSupport, unique_first
from utils.recon_and_seg.pi3_geometry import native_rgbd

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_incremental_membership_exact_with_duplicates_and_high_keys(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    base=[0,0,5,2**62+7];index=IncrementalKeys(base,device=device);truth=set(base)
    rng=np.random.default_rng(7)
    for _ in range(20):
        batch=rng.integers(0,100,30)
        assert np.array_equal(index.contains(batch),[int(x) in truth for x in batch])
        index.add(batch);truth.update(map(int,batch))
    query=np.array([-1,0,5,99,101,2**62+7],dtype=np.int64)
    assert np.array_equal(index.contains(query),[int(x) in truth for x in query])
    assert len(index.base)==3

def test_unique_representative_matches_numpy():
    values=np.array([7,3,7,1,3,8,8],np.int64)
    expected=np.unique(values,return_index=True)[1]
    actual,_=unique_first(values,torch.device('cpu'))
    assert np.array_equal(expected,actual)

def test_temporal_votes_are_distinct_bins_not_repeat_counts():
    support=TemporalSupport(device='cpu');support.add([1,1,2],0);support.add([1],0)
    support.add([2,3],1);support.add([3,4],2)
    assert support.mask(np.array([1,2,3,4]),2,chunk_size=2).tolist()==[False,True,True,False]

def test_native_rgbd_roundtrip_keeps_depth_and_off_center_intrinsics():
    h,w=20,30;y,x=np.mgrid[:h,:w];z=2+.01*x+.02*y
    local=np.stack(((x+.5-13.2)*z/35.,(y+.5-9.1)*z/37.,z),-1)[None].astype(np.float32)
    d,k,errors=native_rgbd(local)
    assert np.array_equal(d,local[...,2])
    assert np.allclose(k[0],[35,37,13.2,9.1],atol=1e-5) and max(errors)<1e-4

def test_native_rgbd_rejects_degenerate_rays():
    with pytest.raises(ValueError):native_rgbd(np.ones((1,5,5,3)))

def test_overlap_priors_preserve_world_gauge_and_rigid_rotation():
    from pi3.pipe.pi3x_vo_v2 import rigid_overlap_priors
    pose=torch.eye(4).repeat(1,2,1,1);pose[..., :3,:3]*=3.;pose[..., :3,3]=torch.tensor([1.,2.,5.])
    depth=torch.ones(1,2,5,6)*2
    rigid,scaled=rigid_overlap_priors(pose,depth)
    assert torch.equal(rigid[..., :3,:3],torch.eye(3).repeat(1,2,1,1))
    assert torch.equal(rigid[..., :3,3],pose[..., :3,3])
    assert torch.equal(scaled,depth*3) and torch.equal(depth,torch.ones_like(depth)*2)
    assert torch.equal(pose[..., :3,:3],torch.eye(3).repeat(1,2,1,1)*3)

@pytest.mark.parametrize('inject',[None,[]])
def test_vo_injects_overlap_geometry_by_default_but_allows_explicit_control(inject):
    from pi3.pipe.pi3x_vo_v2 import Pi3XVO
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1));self.calls=[]
        def forward(self,images,**kwargs):
            self.calls.append(kwargs)
            b,t,_,h,w=images.shape;y,x=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
            local=torch.stack((x.float()/10,y.float()/10,torch.ones_like(x)*2.),-1).repeat(b,t,1,1,1)
            return dict(local_points=local,points=local,camera_poses=torch.eye(4).repeat(b,t,1,1),
                conf=torch.ones(b,t,h,w,1)*3,rays=local/local[...,2:])
    model=Model();out=Pi3XVO(model)(torch.zeros(1,7,3,4,4),chunk_size=4,overlap=2,
        inject_condition=inject,device='cpu')
    assert out['points'].shape[1]==7 and len(model.calls)==3
    assert not model.calls[0]['with_prior']
    for call in model.calls[1:]:
        assert call['with_prior']==(inject is None)
        if inject is None:
            for key in ('mask_add_pose','mask_add_depth','mask_add_ray'):
                assert call[key].sum()==2
            assert torch.allclose(torch.det(call['poses'][...,:3,:3]),torch.ones_like(call['mask_add_pose'],dtype=torch.float32),atol=1e-4)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_stream_dispatch_budget_changes_processing_not_world(monkeypatch):
    from atas_geometry.memory.fuse import _memory_from_voxels
    from atas_geometry.render.render import render_l1_gpu
    from atas_geometry.memory.observation import CameraCalibration
    xyz=np.array([[0,0,2],[.5,0,3]],np.float32)
    mem=_memory_from_voxels(xyz,np.full((2,3),220,np.uint8),voxel_size=.1,origin=np.zeros(3),shot_id='test')
    cam=CameraCalibration('test',np.eye(4),np.array([[15,0,16],[0,15,12],[0,0,1.]]),(32,24))
    monkeypatch.setenv('GEOMETRY_RENDER_RESIDENT_POINTS','1')
    monkeypatch.setenv('GEOMETRY_RENDER_POINT_BATCH','1')
    _,stats=render_l1_gpu(mem,[cam],width=32,height=24,device=torch.device('cuda:0'))
    assert stats['streamed'] and stats['point_count']==2 and stats['max_gpu_point_batch']==1
    assert np.array_equal(xyz,mem.xyz)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('splat',['pixel','bounded_square'])
def test_streamed_one_zbuffer_matches_resident_across_chunks(splat):
    from atas_geometry.memory.fuse import _memory_from_voxels
    from atas_geometry.render.render import render_l1_gpu
    from atas_geometry.render.streamed_render import render_streamed
    from atas_geometry.memory.observation import CameraCalibration
    xyz=np.array([[0,0,5],[0,0,2],[.5,0,3],[-.5,0,4]],np.float32)
    rgb=np.array([[255,0,0],[0,255,0],[0,0,255],[255,255,0]],np.uint8)
    mem=_memory_from_voxels(xyz,rgb,voxel_size=.3,origin=np.zeros(3),shot_id='test')
    cam=CameraCalibration('test',np.eye(4),np.array([[15,0,16],[0,15,12],[0,0,1.]]),(32,24))
    args=dict(width=32,height=24,device=torch.device('cuda:0'),splat_mode=splat,point_footprint=.3,max_point_radius=2)
    expected,stats=render_l1_gpu(mem,[cam],**args)
    actual,stream=render_streamed(mem,[cam],point_batch_size=1,**args)
    assert np.array_equal(expected[0],actual[0])
    assert stream['max_gpu_point_batch']==1 and stream['points_discarded_for_budget']==0
    assert stats['mean_coverage']==stream['mean_coverage']

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_streamed_dynamic_occlusion_and_source_support():
    from atas_geometry.memory.fuse import _memory_from_voxels
    from atas_geometry.render.render import render_l1_gpu
    from atas_geometry.render.streamed_render import render_streamed
    from atas_geometry.memory.observation import CameraCalibration
    mem=_memory_from_voxels(np.array([[0,0,3]],np.float32),np.array([[255,0,0]],np.uint8),voxel_size=.1,origin=np.zeros(3),shot_id='static')
    cam=CameraCalibration('test',np.eye(4),np.array([[15,0,16],[0,15,12],[0,0,1.]]),(32,24))
    dyn=(np.array([[0,0,2],[.4,0,4]],np.float32),np.array([[0,255,0],[0,0,255]],np.uint8),np.array([.3,.3]),np.array([1,1]))
    args=dict(width=32,height=24,device=torch.device('cuda:0'),splat_mode='bounded_square',point_footprint=.1,max_point_radius=2,dynamic_splat_mode='source_pixel',max_dynamic_radius=2)
    expected,_=render_l1_gpu(mem,[cam],dynamic_frames=iter([dyn]),**args)
    actual,stats=render_streamed(mem,[cam],dynamic_frames=iter([dyn]),point_batch_size=1,**args)
    assert np.array_equal(expected[0],actual[0])
    assert stats['dynamic_visible_pixels_per_frame'][0]>0
