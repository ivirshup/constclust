import numpy as np
import pandas as pd
from .test_aggregate import clustering_run

from constclust import ComponentList, Component, reconcile


def test_subsetting_basic(clustering_run):
    r = reconcile(*clustering_run)
    complist = r.get_components(0.9)
    assert type(complist) is ComponentList
    assert type(complist[0]) is Component
    assert complist[0] is complist.components[0]
    assert type(complist[:2]) is ComponentList


def test_reconciler_creation(clustering_run):
    params, clusts = clustering_run
    r = reconcile(params, clusts)
    complist = (
        r
        .subset_cells(clusts.index[:len(clusts) // 2])
        .get_components(0.9)
    )
    assert type(complist) is ComponentList
    # TODO: What else should be true about this?


def test_one_hot(clustering_run):
    params, clusts = clustering_run
    r = reconcile(params, clusts)
    complist = r.get_components(0.9).filter(min_solutions=5)

    encoded = complist.one_hot()
    encoded_sparse = complist.one_hot(as_sparse=True)

    actual_lengths = [len(c.intersect) for c in complist]
    assert list(encoded.sum(axis=1).values) == actual_lengths
    assert list(np.ravel(encoded_sparse.sum(axis=1))) == actual_lengths

    for name, comp in complist._comps.items():
        assert np.all(encoded[comp.intersect_names].loc[name] == True)


def test_obs_names(clustering_run):
    params, clusts = clustering_run
    r = reconcile(*clustering_run)
    pd.testing.assert_index_equal(r.obs_names, clusts.index)

    complist = r.get_components(0.9).filter(min_solutions=5)
    pd.testing.assert_index_equal(complist.obs_names, clusts.index)
