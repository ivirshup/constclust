from constclust.aggregate import reconcile, gen_neighbors
from typing import Tuple
from itertools import product
import pandas as pd
import pytest
import numpy as np

# from collections import namedtuple
# FakeRequest = namedtuple("FakeRequest", ("param"))
# s, c = clustering_run(FakeRequest(int))
# TODO: Rethink how I generate bad clustering to deterministically produce correct example
@pytest.fixture(params=[str, int])
def clustering_run(request) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cells = np.arange(100)
    params = {"a": list(range(4)),  # ordered
              "b": list(range(4)),  # ordered
              "c": list(range(3))}  # unordered
    settings = pd.DataFrame(product(*params.values()), columns=params.keys())
    clusterings = pd.DataFrame(
        np.zeros((len(cells), len(settings)), dtype=np.integer),
        index=cells.astype(request.param),
        columns=settings.index
    )
    mask = settings["a"] > settings["b"]
    # I should change this to be deterministic
    component1 = lambda : np.append(np.zeros(50, dtype=np.integer), np.random.randint(1, 6, 50))
    component2 = lambda : np.append(np.random.randint(1, 6, 50), np.zeros(50, dtype=np.integer))
    for i, b in mask.items():
        if b:
            clusterings[i] = component1()
        else:
            clusterings[i] = component2()
    return settings, clusterings


def test_reconcile(clustering_run: Tuple[pd.DataFrame, pd.DataFrame]):
    recon = reconcile(*clustering_run)
    comps = recon.get_components(0.9)
    assert len(comps) == 2
    # If I update to have intersect return cell names, this'll need to change
    assert all(comps[0].intersect == np.arange(50, 100))
    assert all(comps[1].intersect == np.arange(0, 50))


def test_subsetting(clustering_run: Tuple[pd.DataFrame, pd.DataFrame]):
    recon = reconcile(*clustering_run)
    by_cells = recon.subset_cells(recon._obs_names[slice(50, 100)])
    by_cells_comps = by_cells.get_components(0.9)
    assert all(by_cells._obs_names == recon._obs_names[slice(50, 100)])
    assert len(by_cells_comps) == 1
    assert all(np.sort(by_cells_comps[0].intersect_names) == np.sort(by_cells.clusterings.index))
    by_settings = recon.subset_clusterings(lambda x: x["a"] < 2)
    assert (len(recon.settings) / 2) == len(by_settings.settings)
    assert (recon.clusterings.shape[1] / 2) == by_settings.clusterings.shape[1]
    assert all(by_settings._mapping.index.get_level_values("clustering").unique() == by_settings.settings.index)
    assert all(by_settings.cluster_ids == np.unique(by_settings.clusterings.stack()))


def test_naming(clustering_run: Tuple[pd.DataFrame, pd.DataFrame]):
    s, c = clustering_run
    r = reconcile(s, c)
    cs = r.get_components(0.9)
    # Now for subset
    subset = np.sort(np.random.choice(len(c), 75, replace=False))
    c_sub = c.iloc[subset, :]
    r_sub = reconcile(s, c_sub)
    # cs_sub = r_sub.get_components(0.9)
    in_comp = r_sub.clusterings.iloc[-1, 0]
    comp_from_subset = r_sub.find_components(0.9, [in_comp])[0]
    assert all(np.isin(comp_from_subset.intersect_names, cs[0].intersect_names))


def test_component_neighbors(clustering_run):
    recon = reconcile(*clustering_run)
    cs = recon.get_components(.9)
    all_ns = gen_neighbors(clustering_run[0], "oou")
    for c in cs:
        comp_clusterings = c.settings.index
        subset_all_ns = set(filter(lambda x: (x[0] in comp_clusterings) and (x[1] in comp_clusterings), all_ns))
        subset_ns = set(gen_neighbors(c.settings, "oou"))
        assert subset_all_ns == subset_ns
