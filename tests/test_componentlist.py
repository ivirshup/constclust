from .test_aggregate import clustering_run
from constclust import ComponentList, Component, reconcile

def test_subsetting_basic(clustering_run):
    r = reconcile(*clustering_run)
    complist = ComponentList(r.get_components(0.9))
    assert type(complist) is ComponentList
    assert type(complist[0]) is Component
    assert complist[0] is complist.components[0]
    assert type(complist[:2]) is ComponentList
