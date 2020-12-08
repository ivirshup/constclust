from collections import namedtuple
from itertools import product, repeat
import pandas as pd
from constclust.paramspace import gen_neighbors


def test_neighbors_basic():
    # Setup
    chars = list("abcd")
    for n in range(2, 5):
        nt = namedtuple("NamedTuple", chars[:n])
        s = pd.DataFrame(list(product(*repeat(range(3), n))), columns=chars[:n])
        nodes = list(s.itertuples(index=False))
        edge_indices = gen_neighbors(s)
        edges = [(nodes[i], nodes[j]) for i, j in edge_indices]
        # middle point, n incoming edges, n outgoing
        assert len([x for x in edges if nt(*repeat(1, n)) in x]) == (n * 2)
        # initial corner point, n outgoing edges
        assert len([x for x in edges if nt(*repeat(0, n)) in x]) == n
        assert (
            len([x for x in edges if nt(*repeat(2, n)) in x]) == n
        )  # last corner point, n incoming edges


def test_neighbors_unordered():
    nt = namedtuple("NamedTuple", list("abc"))
    s = pd.DataFrame(list(product(*repeat(range(3), 3))), columns=list("abc"))
    edges = gen_neighbors(s, list("oou"))
    # I think it should be (2 * n_ordered_fields) + (n_unique_unordered_keys - n_unordered_fields) # -n_unordered_fields for
