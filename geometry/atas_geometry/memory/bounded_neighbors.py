"""Exact radius rejection without searching for irrelevant distant neighbors."""
import numpy as np


def outside_radius(tree, points, radius, *, workers=4, coarse_tree=None, coarse_error=0.):
    radius=float(radius)
    if not np.isfinite(radius) or radius<0:
        raise ValueError('radius must be finite and nonnegative')
    if coarse_tree is not None:
        if not np.isfinite(coarse_error) or coarse_error < 0:
            raise ValueError('invalid coarse covering bound')
        # Representatives are actual world points. Their distance <= r proves
        # overlap; distance > r + covering bound proves absence of any neighbour.
        # Only the undecidable shell visits the full-resolution index.
        bound=np.nextafter(radius+coarse_error,np.inf)
        d=coarse_tree.query(points,k=1,workers=workers,distance_upper_bound=bound)[0]
        result=d>radius
        uncertain=(d>radius)&(d<=bound)
        if uncertain.any():
            result[uncertain]=outside_radius(tree,points[uncertain],radius,workers=workers)
        return result
    # cKDTree's bound is exclusive: include the exact radius boundary because
    # the original fusion predicate was distance > radius, not >= radius.
    bound=np.nextafter(radius,np.inf) if radius>=np.sqrt(np.finfo(float).tiny) else np.inf
    distance=tree.query(points,k=1,workers=workers,distance_upper_bound=bound)[0]
    return distance>radius
