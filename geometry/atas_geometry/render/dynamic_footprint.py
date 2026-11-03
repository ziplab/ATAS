"""Time-local RGB-D sample footprints; never borrow the static voxel size."""
import cv2
import numpy as np


def dynamic_render_options(payload):
    mode=str(payload.get("dynamic_splat_mode","source_pixel"))
    if mode not in {"pixel","source_pixel"}:
        raise ValueError("dynamic_splat_mode must be pixel or source_pixel")
    cap=payload.get("max_dynamic_radius",2)
    if isinstance(cap,bool) or not isinstance(cap,int) or not 0<=cap<=2:
        raise ValueError("max_dynamic_radius must be an integer in 0..2")
    return {"dynamic_splat_mode":mode,"max_dynamic_radius":cap}


def source_pixel_support(depth, dynamic, fx, fy):
    """World-space pixel edge and conservative raster caps on the source grid.

    The support is depth / geometric-mean focal length, in the SAME scene scale
    as XYZ. Silhouettes cap expansion at one output pixel; depth discontinuities
    retain point centers only. No points are deleted and no time is accumulated.
    """
    depth=np.asarray(depth,dtype=np.float32);dynamic=np.asarray(dynamic,dtype=bool)
    if depth.shape!=dynamic.shape or depth.ndim!=2:
        raise ValueError("depth/mask shape mismatch")
    if not np.isfinite([fx,fy]).all() or min(fx,fy)<=0:
        raise ValueError("invalid source focal length")
    valid=np.isfinite(depth)&(depth>1e-5)
    if np.any(dynamic&~valid): raise ValueError("dynamic points require valid depth")
    interior=cv2.erode(dynamic.astype(np.uint8),np.ones((3,3),np.uint8),
        borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
    caps=np.where(interior,2,1).astype(np.uint8)
    padded=np.pad(depth,1,mode="edge");safe=np.ones_like(dynamic)
    for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
        neighbor=padded[1+dy:1+dy+depth.shape[0],1+dx:1+dx+depth.shape[1]]
        safe &= np.isfinite(neighbor)&(neighbor>1e-5)&(np.abs(neighbor-depth)<=.05*depth)
    caps[~safe]=0
    return (depth[dynamic]/np.sqrt(float(fx)*float(fy))).astype(np.float32),caps[dynamic]


def dynamic_frame(recon,index,observation=None):
    """Preserve exactly the dense unprojection ordering and every dynamic point."""
    from atas_geometry.memory.vista4d_layers import _mask
    observation=recon.unproject(index) if observation is None else observation
    depth=recon.depth_at(index)  # SceneEventRecon supplies scene-scaled depth.
    valid=np.isfinite(depth)&(depth>1e-5)
    if recon.exclude_sky and index<len(recon.sky_files):
        valid &= ~_mask(recon.sky_files[index],depth.shape)
    if int(valid.sum())!=len(observation.xyz):
        raise ValueError("dynamic footprint pixel/point ordering mismatch")
    selected=np.asarray(observation.foreground)>=.5
    mask=np.zeros_like(valid);mask[valid]=selected
    fx,fy=map(float,recon.intrinsics[index,:2])
    footprint,caps=source_pixel_support(depth,mask,fx,fy)
    return observation.xyz[selected],observation.rgb[selected],footprint,caps
