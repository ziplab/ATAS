import numpy as np
import pytest
from scipy.spatial import cKDTree
from atas_geometry.memory.bounded_neighbors import outside_radius
@pytest.mark.parametrize('radius',[0.,.02,.25,2.])
def test_exact_equivalence_to_unbounded_nearest(radius):
    rng=np.random.default_rng(341)
    reference=rng.normal(size=(4000,3)).astype(np.float32)
    points=np.concatenate((rng.normal(size=(3000,3)).astype(np.float32),reference[:100],
                           np.full((100,3),100,dtype=np.float32)))
    tree=cKDTree(reference)
    expected=tree.query(points,k=1,workers=1)[0]>radius
    assert np.array_equal(outside_radius(tree,points,radius,workers=1),expected)


def test_exact_radius_boundary_and_empty_queries():
    tree=cKDTree([[0.,0.,0.]])
    query=np.array([[1.,0.,0.],[np.nextafter(1.,0.),0.,0.],
                    [np.nextafter(1.,np.inf),0.,0.],[0.,0.,0.]])
    assert outside_radius(tree,query,1.,workers=1).tolist()==[False,False,True,False]
    assert outside_radius(tree,np.empty((0,3)),1.,workers=1).shape==(0,)
