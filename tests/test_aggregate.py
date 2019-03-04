from constclust.aggregate import reconcile, gen_neighbors
from typing import Tuple
from itertools import product
import pandas as pd
import pytest
import numpy as np


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
    assert len(comps[0].intersect) == 50
    assert len(comps[1].intersect) == 50


def test_subsetting(clustering_run: Tuple[pd.DataFrame, pd.DataFrame]):
    recon = reconcile(*clustering_run)
    by_cells = recon.subset_cells(recon._obs_names[slice(50)])
    assert all(by_cells._obs_names == recon._obs_names[slice(50)])
    assert len(by_cells.get_components(0.9)) == 1
    by_settings = recon.subset_clusterings(lambda x: x["a"] < 2)
    assert (len(recon.settings) / 2) == len(by_settings.settings)
    assert (recon.clusterings.shape[1] / 2) == by_settings.clusterings.shape[1]


def test_component_neighbors(clustering_run):
    recon = reconcile(*clustering_run)
    cs = recon.get_components(.9)
    all_ns = gen_neighbors(clustering_run[0], "oou")
    for c in cs:
        comp_clusterings = c.settings.index
        subset_all_ns = set(filter(lambda x: (x[0] in comp_clusterings) and (x[1] in comp_clusterings), all_ns))
        subset_ns = set(gen_neighbors(c.settings, "oou"))
        assert subset_all_ns == subset_ns

