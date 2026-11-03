"""Pinhole RGB-D conversion preserving Pi3's predicted camera gauge."""
import numpy as np

def fit_pinhole(local):
    local=np.asarray(local,dtype=np.float64)
    h,w=local.shape[:2];y,x=np.mgrid[:h,:w]
    values=[];errors=[]
    for dim,pixels in [(0,x+.5),(1,y+.5)]:
        ratio=local[...,dim]/np.maximum(local[...,2],1e-8)
        good=np.isfinite(local).all(-1)&(local[...,2]>1e-8)&np.isfinite(ratio)&(np.abs(ratio)<10)
        if good.sum()<16:raise ValueError('insufficient valid Pi3 rays')
        r=ratio[good];p=pixels[good];rc=r-r.mean();pc=p-p.mean()
        denom=np.dot(rc,rc)
        if denom<=1e-12:raise ValueError('degenerate Pi3 rays')
        focal=float(np.dot(rc,pc)/denom);center=float(p.mean()-focal*r.mean())
        if not np.isfinite(focal) or focal<=0:raise ValueError('invalid Pi3 focal')
        values.append((focal,center));errors.append(float(np.median(np.abs(focal*r+center-p))))
    return np.array([values[0][0],values[1][0],values[0][1],values[1][1]]),max(errors)

def native_rgbd(local_points,max_residual_pixels=2.):
    points=np.asarray(local_points,dtype=np.float32);intrinsics=[];errors=[]
    for local in points:
        k,error=fit_pinhole(local);intrinsics.append(k);errors.append(error)
    if not errors or max(errors)>max_residual_pixels:
        raise ValueError(f'Pi3 pinhole residual exceeds {max_residual_pixels}px: {max(errors,default=None)}')
    # No independent per-frame z-shift: depths and poses retain the same gauge.
    return points[...,2].copy(),np.asarray(intrinsics,np.float32),errors
