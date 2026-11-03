"""Bounded GPU working set, one shared z-buffer for all static/dynamic points."""
import time
import numpy as np
import torch

def render_streamed(memory,cameras,*,width,height,device,splat_mode='pixel',point_footprint=None,
    max_point_radius=1,entity_exclusion_points=None,dynamic_frames=None,
    dynamic_splat_mode='pixel',max_dynamic_radius=2,frame_sink=None,point_batch_size=1000000):
    if point_batch_size<1:raise ValueError('point batch size must be positive')
    from atas_geometry.render.render import _project_entity_exclusion_mask, _coverage_outside_mask
    from atas_geometry.render.dynamic_footprint import dynamic_render_options
    dynamic_render_options({'dynamic_splat_mode':dynamic_splat_mode,'max_dynamic_radius':max_dynamic_radius})
    device=torch.device(device);torch.cuda.set_device(device);t=time.perf_counter()
    dynamic=iter(dynamic_frames) if dynamic_frames is not None else None
    frames=[];coverages=[];centers=[];center_counts=[];dynamic_counts=[];dynamic_bounds=[]
    condition=[];condition_center=[];excluded_fractions=[]
    radii_hist={str(i):0 for i in range(max_point_radius+1)};dyn_hist={str(i):0 for i in range(3)}
    fragments=visible=0;max_batch_seen=0;last_total=len(memory.xyz)
    for camera in cameras:
        zbuf=torch.full((height*width,),float('inf'),device=device)
        color=torch.full((height*width,3),127,dtype=torch.uint8,device=device)
        owner=torch.zeros(height*width,dtype=torch.bool,device=device)
        center_support=torch.zeros_like(owner)
        w2c=torch.as_tensor(camera.world_to_camera,dtype=torch.float32,device=device)
        fx=float(camera.intrinsic[0,0])*width/camera.image_size[0]
        fy=float(camera.intrinsic[1,1])*height/camera.image_size[1]
        cx=float(camera.intrinsic[0,2])*width/camera.image_size[0]
        cy=float(camera.intrinsic[1,2])*height/camera.image_size[1]
        sources=[(memory.xyz,memory.rgb,False,None,None)]
        if dynamic is not None:
            data=next(dynamic);xyz,rgb=map(np.asarray,data[:2]);support=limits=None
            if len(rgb)!=len(xyz):raise ValueError('dynamic RGB length mismatch')
            if dynamic_splat_mode=='source_pixel':
                if len(data)!=4:raise ValueError('source_pixel requires support and limits')
                support,limits=map(np.asarray,data[2:])
                if (support.shape!=(len(xyz),) or limits.shape!=support.shape or
                    not np.isfinite(support).all() or np.any(support<=0) or
                    not np.isfinite(limits).all() or np.any((limits<0)|(limits>2)) or
                    np.any(limits!=np.floor(limits))):raise ValueError('invalid dynamic source-pixel support')
            sources.append((xyz,rgb,True,support,limits));last_total=len(memory.xyz)+len(xyz)
        dyn_uv=[]
        for source_xyz,source_rgb,is_dynamic,support,limits in sources:
            for start in range(0,len(source_xyz),point_batch_size):
                end=min(start+point_batch_size,len(source_xyz));max_batch_seen=max(max_batch_seen,end-start)
                xyz=torch.as_tensor(np.asarray(source_xyz[start:end]),dtype=torch.float32,device=device)
                rgb=torch.as_tensor(np.asarray(source_rgb[start:end]),dtype=torch.uint8,device=device)
                cam=xyz@w2c[:3,:3].T+w2c[:3,3];depth=cam[:,2]
                u=fx*cam[:,0]/depth.clamp(min=1e-6)+cx;v=fy*cam[:,1]/depth.clamp(min=1e-6)+cy
                # Match ReconStore's half-pixel centres and the resident renderer.
                uu=u.floor().long();vv=v.floor().long();front=torch.isfinite(cam).all(1)&(depth>1e-5)
                inside=front&(uu>=0)&(uu<width)&(vv>=0)&(vv<height)
                center_support[vv[inside]*width+uu[inside]]=True
                if is_dynamic and bool(inside.any()):dyn_uv.append(torch.stack((u[inside]/width,v[inside]/height),1).cpu())
                if is_dynamic:
                    if dynamic_splat_mode=='source_pixel':
                        edge=torch.as_tensor(support[start:end],device=device)*(fx+fy)*.5/depth.clamp(min=1e-6)
                        radii=torch.ceil((edge-1)*.5).clamp(0,max_dynamic_radius).long()
                        radii=torch.minimum(radii,torch.as_tensor(limits[start:end],device=device,dtype=torch.int64))
                    else:radii=torch.zeros(len(xyz),dtype=torch.int64,device=device)
                elif splat_mode=='bounded_square':
                    edge=float(point_footprint)*(fx+fy)*.5/depth.clamp(min=1e-6)
                    radii=torch.ceil((edge-1)*.5).clamp(0,max_point_radius).long()
                else:radii=torch.zeros(len(xyz),dtype=torch.int64,device=device)
                maximum=max_dynamic_radius if is_dynamic and dynamic_splat_mode=='source_pixel' else (0 if is_dynamic or splat_mode=='pixel' else max_point_radius)
                for radius in range(maximum+1):
                    selected=front&(radii==radius)&(uu>=-radius)&(uu<width+radius)&(vv>=-radius)&(vv<height+radius)
                    count=int(selected.sum());visible+=count;radii_hist[str(radius)]=radii_hist.get(str(radius),0)+count
                    if is_dynamic:dyn_hist[str(radius)]+=count
                    su,sv,sd,sc=uu[selected],vv[selected],depth[selected],rgb[selected]
                    # All chunks update this same depth buffer before a frame
                    # is materialized. This is not a paste of rendered images.
                    for dy in range(-radius,radius+1):
                        for dx in range(-radius,radius+1):
                            x,y=su+dx,sv+dy;valid=(x>=0)&(x<width)&(y>=0)&(y<height)
                            pix=y[valid]*width+x[valid];fragments+=len(pix)
                            zbuf.scatter_reduce_(0,pix,sd[valid],reduce='amin',include_self=True)
                    for dy in range(-radius,radius+1):
                        for dx in range(-radius,radius+1):
                            x,y=su+dx,sv+dy;valid=(x>=0)&(x<width)&(y>=0)&(y<height)
                            pix=y[valid]*width+x[valid];d=sd[valid];winners=(d-zbuf[pix]).abs()<=1e-4
                            color[pix[winners]]=sc[valid][winners];owner[pix[winners]]=is_dynamic
                del xyz,rgb,cam,depth,u,v,uu,vv,radii
        covered=torch.isfinite(zbuf);covered2=covered.reshape(height,width)
        coverages.append(float(covered.float().mean()));center_counts.append(float(center_support.float().mean()))
        y0,y1=height//4,height-height//4;x0,x1=width//4,width-width//4
        centers.append(float(covered2[y0:y1,x0:x1].float().mean()))
        if dynamic is not None:
            dynamic_counts.append(int(owner.sum()))
            dynamic_bounds.append(torch.quantile(torch.cat(dyn_uv),torch.tensor([.01,.99]),dim=0).flatten().tolist() if dyn_uv else None)
        if entity_exclusion_points is not None:
            exclusion=_project_entity_exclusion_mask(entity_exclusion_points,camera,width=width,height=height)
            cov=covered2.cpu().numpy();condition.append(_coverage_outside_mask(cov,exclusion))
            condition_center.append(_coverage_outside_mask(cov[y0:y1,x0:x1],exclusion[y0:y1,x0:x1]));excluded_fractions.append(float(exclusion.mean()))
        image=color.reshape(height,width,3).cpu().numpy()
        if frame_sink is None:frames.append(image)
        else:frame_sink(image)
    torch.cuda.synchronize(device)
    stats={'gpu_splat_ms':(time.perf_counter()-t)*1000,'mean_coverage':float(np.mean(coverages)) if coverages else 0.,
        'min_coverage':min(coverages,default=0.),'coverages':coverages,'mean_center_coverage':float(np.mean(centers)) if centers else 0.,
        'min_center_coverage':min(centers,default=0.),'mean_point_center_coverage':float(np.mean(center_counts)) if center_counts else 0.,
        'point_center_coverages':center_counts,'point_count':last_total,'splat_mode':splat_mode,
        'dynamic_splat_mode':dynamic_splat_mode if dynamic is not None else None,
        'dynamic_radius_histogram':dyn_hist,'dynamic_visible_pixels_per_frame':dynamic_counts,'dynamic_screen_bounds_per_frame':dynamic_bounds,
        'max_dynamic_radius':max_dynamic_radius if dynamic_splat_mode=='source_pixel' else 0,
        'point_footprint':point_footprint,'max_point_radius':max_point_radius,'radius_histogram':radii_hist,
        'splat_fragment_count':fragments,'mean_fragments_per_visible_point':fragments/max(1,visible),
        'occlusion':'single_nearest_depth_per_splat_fragment','streamed':True,
        'point_batch_size':point_batch_size,'max_gpu_point_batch':max_batch_seen,'points_discarded_for_budget':0}
    if entity_exclusion_points is not None:
        stats['static_condition_coverage']={'mean':float(np.mean(condition)),'min':min(condition,default=0.),
            'mean_center':float(np.mean(condition_center)),'min_center':min(condition_center,default=0.),
            'mean_entity_excluded_fraction':float(np.mean(excluded_fractions)),
            'mask_policy':'per_entity_projected_l2_bbox_p01_p99_pad5pct_min2px'}
    return frames,stats
