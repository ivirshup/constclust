from constclust.aggregate import Reconciler
from itertools import product
import pandas as pd
import pytest
import numpy as np


# TODO: Rethink how I generate bad clustering to deterministically produce correct example
@pytest.fixture
def clustering_run():
    cells = np.arange(100)
    params = {"a": list(range(4)), # ordered
              "b": list(range(4)), # ordered
              "c": list(range(3))} # unordered
    settings = pd.DataFrame(product(*params.values()), columns=params.keys())
    clusterings = pd.DataFrame(
        np.zeros((len(cells), len(settings)), dtype=np.integer),
        index=cells,
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


def test_reconcile(clustering_run):
    settings, clusterings = clustering_run
    recon = Reconciler(settings, clusterings)
    comps = recon.get_components(0.9)
    assert len(comps) == 2
    assert len(comps[0].intersect) == 50
    assert len(comps[1].intersect) == 50
