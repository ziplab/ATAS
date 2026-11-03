"""GPU dense projection kernels; no all-world per-frame transfer or search."""
import numpy as np
import torch


class Projection:
    def __init__(self, views, device):
        self.device = device
        self.views = [tuple(torch.as_tensor(x, device=device, dtype=torch.float64)
                            for x in view) for view in views]

    def lift(self, depth, pose, k, valid):
        # Only lift selected pixels; do not allocate a full H x W x 3 CPU grid.
        y, x = np.nonzero(valid)
        d = torch.as_tensor(depth[y,x], device=self.device, dtype=torch.float64)
        xx = torch.as_tensor(x, device=self.device)
        yy = torch.as_tensor(y, device=self.device)
        local = torch.stack(((xx+.5-k[2])/k[0]*d, (yy+.5-k[3])/k[1]*d, d), -1)
        p = torch.as_tensor(pose, device=self.device, dtype=torch.float64)
        return (local @ p[:3,:3].T + p[:3,3]).float().cpu().numpy()

    def visibility(self, points, tolerance, chunk_size=250000):
        surfaces, conflicts = [], []
        for start in range(0,len(points),chunk_size):
            xyz = torch.as_tensor(points[start:start+chunk_size], device=self.device, dtype=torch.float64)
            surface = torch.zeros(len(xyz),device=self.device,dtype=torch.bool)
            conflict = torch.zeros(len(xyz),device=self.device,dtype=torch.int16)
            for depth, pose, k in self.views:
                local = (xyz-pose[:3,3]) @ pose[:3,:3]
                z = local[:,2]
                u = k[0]*local[:,0]/z.clamp(min=1e-8)+k[2]
                v = k[1]*local[:,1]/z.clamp(min=1e-8)+k[3]
                h,w = depth.shape
                ids = torch.where((z>0)&(u>=0)&(u<w)&(v>=0)&(v<h))[0]
                d = depth[v[ids].long(),u[ids].long()]
                delta = z[ids]-d
                surface[ids[(d>0)&(delta.abs()<=tolerance)]] = True
                conflict[ids[(d>0)&(delta < -tolerance)]] += 1
            surfaces.append(surface.cpu().numpy()); conflicts.append(conflict.cpu().numpy())
        return (np.concatenate(surfaces) if surfaces else np.zeros(0,bool),
                np.concatenate(conflicts) if conflicts else np.zeros(0,np.int16))
