import numpy as np
import pandas as pd
import networkx as nx

from constclust.clustree import gen_clustree

import pytest


@pytest.mark.parametrize(
    "clusterings,expected",
    [
        (
            pd.DataFrame(
                {
                    "a": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    "b": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    "c": [0, 0, 0, 1, 1, 2, 2, 2, 2, 2],
                }
            ),
            nx.DiGraph(
                {
                    0: {1: {"weight": 0.5}, 2: {"weight": 0.5}},
                    1: {3: {"weight": 0.6}, 4: {"weight": 0.4}},
                    2: {6: {"weight": 1.0}},
                }
            ),
        ),
        (
            pd.DataFrame(
                {
                    "a": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    "b": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    "c": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                }
            ),
            nx.DiGraph(
                {
                    0: {1: {"weight": 0.5}, 2: {"weight": 0.5}},
                    1: {3: {"weight": 0.5}},
                    2: {3: {"weight": 0.5}},
                }
            ),
        ),
    ],
    ids=["tree", "diamond"],
)
def test_clustree_weights(clusterings, expected):
    import networkx.algorithms.isomorphism as iso

    result = gen_clustree(clusterings)

    assert nx.is_isomorphic(
        expected, result, edge_match=iso.numerical_edge_match("weight", default=1)
    )
