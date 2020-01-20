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
