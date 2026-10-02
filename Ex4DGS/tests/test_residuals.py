import numpy as np
import pytest
from rt_pipeline.hard_regions import patch_mse, persistent_candidates, select_points, expand_patches


def test_partial_patches_do_not_dilute_error():
    image=np.ones((3,35,34));target=np.zeros_like(image)
    patches,mean=patch_mse(image,target,32)
    np.testing.assert_allclose(patches,np.ones((2,2)))
    assert mean==1


def test_persistence_and_stalled_learning_required():
    history=[{'mse':np.array([[.04,.08*(.5**i)],[.001,.04 if i==3 else .001]]),'global_mse':.005} for i in range(4)]
    selected,_=persistent_candidates(history)
    np.testing.assert_array_equal(selected,[[True,False],[False,False]])


def test_zero_error_never_produces_candidates():
    history=[{'mse':np.zeros((2,3)),'global_mse':0} for _ in range(4)]
    assert not persistent_candidates(history)[0].any()
    assert not select_points(np.zeros(100),np.zeros(100)).any()


def test_view_support_and_budget_do_not_force_selection():
    votes=np.array([3,2,1,0,0,0,0,0,0,0]);scores=np.arange(10)[::-1]
    np.testing.assert_array_equal(np.flatnonzero(select_points(votes,scores)),[0,1])
    assert not select_points(votes,scores,min_views=4).any()


def test_insufficient_observations_rejected():
    with pytest.raises(ValueError):persistent_candidates([])


def test_expand_retains_border_shape():
    mask=expand_patches(np.array([[0,1],[1,0]],bool),35,34,32)
    assert mask.shape==(35,34) and mask[34,0] and mask[0,33]
