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
            nx.DiGraph({0: [1, 2], 1: [3, 4], 2: [6]}),
        ),
        (
            pd.DataFrame(
                {
                    "a": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    "b": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    "c": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                }
            ),
            nx.DiGraph({0: [1, 2], 1: [3], 2: [3]}),
        ),
    ],
    ids=["tree", "diamond"],
)
def test_clustree_generation_isomorphic(clusterings, expected):
    result = gen_clustree(clusterings)

    assert nx.is_isomorphic(expected, result)
