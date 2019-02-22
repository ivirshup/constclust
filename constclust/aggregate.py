from .paramspace import gen_neighbors

import igraph
import numba
import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype, is_integer_dtype, is_bool_dtype  # is_numeric_dtype
from sklearn import metrics

from collections import namedtuple
from functools import reduce, partial
from itertools import chain
from multiprocessing import Pool

# TODO: Generalize documentation
# TODO: Should _mapping be mapping? Would I want to give names then?

ClusterRef = namedtuple("ClusterRef", ("clustering_id", "cluster_id"))


# TODO: Have views map back to clustering index names
# Alternatively, do I just want to use a mapping?
# I wouldn't get those index names, but I can get that later.
class ClusterIndexer(object):
    """
    API convenience which allows indexing access to parent Reconciler's
    clusters.
    """

    def __init__(self, parent):
        self._parent = parent

    def __len__(self):
        return len(self._parent._mapping)

    # Note: I might want to modify this one in the future, for specialized access
    def __getitem__(self, key):
        return self._parent._mapping[key]

    def __iter__(self):
        return self._parent._mapping.__iter__()

    def __contains__(self, key):
        # This should be able to check for a clustering, or a cluster
        return self._parent._mapping.__contains__(key)


class Component(object):
    """
    A connected component from a `Reconciler`

    Attributes
    ----------
    _parent : Reconciler
        The Reconciler which generated this component.
    settings : pd.DataFrame
        Subset of parents settings. Contains only settings for clustering
        which appear in this component.
    _mapping : pd.Series
        Partial view of parents `_mapping`.
    intersect : np.array
        Intersect of clusters in this component.
    union : np.array
        Union of clusters in this component.
    """

    def __init__(self, reconciler, cluster_ids):
        self._parent = reconciler
        self._cluster_ids = cluster_ids
        self._mapping = self._parent._mapping.loc[:, cluster_ids]
        if self._parent.is_subset:
            empty = np.vectorize(len)(self._mapping) == 0
            clusterings = self._mapping[~empty].index.get_level_values("clustering")
        else:
            clusterings = self._mapping.index.get_level_values("clustering").unique()
        self.settings = self._parent.settings.loc[clusterings]
        # self.clusters = ClusterIndexer(self)
        self.intersect = reduce(
            partial(np.intersect1d, assume_unique=True), self._mapping.values
        )
        self.union = reduce(np.union1d, self._mapping.values)

    def one_hot(self, selection="intersect"):
        encoding = np.zeros(self._parent.clusterings.shape[0], dtype=bool)
        if selection == "intersect":
            encoding[self.intersect] = True
        elif selection == "union":
            encoding[self.union] = True
        else:
            raise ValueError(
                f"Parameter `selection` must be either 'intersect' or 'union', was '{selection}'"
            )

    def __len__(self):
        return len(self._mapping)

    def __repr__(self):
        return (
            f"<Component n_solutions={len(self)}, "
            f"max_cells={len(self.union)}, "
            f"min_cells={len(self.intersect)}>"
        )


def reconcile(settings, clusterings, nprocs=1):
    """Constructor for reconciler object.

    Args:
        settings (`pd.DataFrame`)
        clusterings (`pd.DataFrame`)
        nprocs (`int`)
    """
    assert all(settings.index == clusterings.columns)  # I should probably save these, right?
    # Check clusterings:
    clust_dtypes = clusterings.dtypes
    if not all(map(is_integer_dtype, clust_dtypes)):
        wrong_types = {t for t in clust_dtypes if not is_integer_dtype(t)}
        raise TypeError(
            "Contents of `clusterings` must be integers dtypes. Found:"
            " {}".format(wrong_types)
        )
    clusterings = clusterings.copy()  # This gets mutated
    mapping = gen_mapping(clusterings)

    # TODO: cleanup, this is for transition from networkx to igraph
    # Fix mapping
    frame = mapping.index.to_frame()
    mapping.index = pd.MultiIndex.from_arrays(
        (frame["clustering"].values, np.arange(frame.shape[0])),
        names=("clustering", "cluster"),
    )
    # Set cluster names to be unique
    cvals = clusterings.values
    cvals[:, 1:] += (cvals[:, :-1].max(axis=0) + 1).cumsum()
    assert all(np.unique(cvals) == mapping.index.levels[1])

    edges = build_graph(settings, clusterings, mapping=mapping, nprocs=nprocs)
    graph = igraph.Graph(
        n=len(mapping),
        edges=list(((i, j) for i, j, k in edges)),
        edge_attrs={"weight": list(k for i, j, k in edges)},
    )
    return Reconciler(settings, clusterings, mapping, graph)


class Reconciler(object):
    """
    Collects and reconciles many clusterings by local (in parameter space)
    stability.

    Attributes
    ----------
    settings : pd.DataFrame
        Contains settings for all clusterings. Index corresponds to
        `.clusterings` columns, while columns should correspond to the
        parameters which were varied.
    clusterings : pd.DataFrame
        Contains cluster assignments for each cell, for each clustering.
        Columns correspond to `.settings` index, while the index correspond
        to the cells. Each cluster is encoded with a unique cluster id.
    _obs_names : pd.Index
        Ordered set for names of the cells. Internally they are refered to by
        integer positions.
    _mapping : pd.Series
        Series which maps clustering and cluster id to that clusters contents.
    graph : igraph.Graph
        Weighted graph. Nodes are clusters (identified by unique cluster id
        integer, same as in `.clusterings`). Edges connect clusters with shared
        contents. Weight is the Jaccard similarity between the contents of the
        clusters.
    """

    def __init__(self, settings, clusterings, mapping, graph):
        assert all(settings.index == clusterings.columns)
        self.settings = settings
        self.clusterings = clusterings
        self._obs_names = clusterings.index
        self._mapping = mapping
        self.graph = graph
        self.is_subset = False  # Place holder, until I implement view type
        # self.clusters = ClusterIndexer(self)

    def get_components(self, min_weight, min_cells=2):
        """
        Return connected components of graph, with edges filtered by min_weight
        """
        over_cutoff = np.where(np.array(self.graph.es["weight"]) >= min_weight)[0]
        # If vertices are deleted, indices (ids) change
        sub_g = self.graph.subgraph_edges(over_cutoff, delete_vertices=False)
        comps = []
        # Filter out 1 node components
        for graph_comp in filter(lambda x: len(x) > 1, sub_g.components()):
            comp = Component(self, list(graph_comp))
            if len(comp.intersect) < min_cells:
                continue
            comps.append(comp)
        comps.sort(key=lambda x: (len(x.settings), len(x.intersect)), reverse=True)
        return comps

    def get_param_range(self, clusters):
        """
        Given a set of clusters, returns the range of parameters for which they were calculated.

        Parameters
        ----------
        clusters : Union[Collection[ClusterRef], Collection[Int]]
            If its a collection of ints, I'll say that was a range of parameter ids. 
        """
        idx = []
        for c in clusters:
            if isinstance(c, int):
                idx.append(c)
            else:
                idx.append(c[0])
        return self.settings.loc[idx]

    def subset_clusterings(self, clusterings_to_keep):
        """
        Take subset of Reconciler, where only `clusterings_to_keep` are present.

        Reduces size of both `.settings` and `.clusterings`

        Args:
            clusterings_to_keep: Indexer into `Reconciler.settings`. Anything that should
                give the correct result for `reconciler.settings.loc[clusterings_to_keep]`.
        """
        clusterings_to_keep = self.settings.loc[clusterings_to_keep].index.values
        new_settings = self.settings.loc[
            clusterings_to_keep
        ]  # Should these be copies?
        new_clusters = self.clusterings.loc[:, clusterings_to_keep]  # This is a copy?
        new_mapping = self._mapping.copy()
        to_remove = ~(
            new_mapping.index.get_level_values("clustering").isin(clusterings_to_keep)
        )
        new_mapping.loc[to_remove] = [
            np.array([], dtype=self.clusterings.values.dtype) for i in range(to_remove.sum())
        ]
        new_rec = Reconciler(new_settings, new_clusters, new_mapping, self.graph)
        new_rec.is_subset = True
        return new_rec

    def subset_cells(self, cells_to_keep):
        """
        Take subset of Reconciler, where only `cells_to_keep` are present

        Args:
            cells_to_keep: I
        """
        intmap = pd.Series(
            np.arange(self.clusterings.shape[0]), index=self.clusterings.index
        )
        cells_to_keep = intmap[self.clusterings.loc[cells_to_keep].index.values]
        get_subset = partial(np.intersect1d, cells_to_keep, assume_unique=True)
        new_mapping = self._mapping.apply(get_subset)
        new_rec = Reconciler(
            self.settings,
            self.clusterings.iloc[cells_to_keep, :],
            new_mapping,
            self.graph
        )
        new_rec.is_subset = True
        return new_rec


def _prep_neighbors(neighbors, mapping):
    for clustering1_id, clustering2_id in neighbors:
        clusters1 = mapping[clustering1_id]
        clusters2 = mapping[clustering2_id]
        yield (
            list(clusters1.values),  # Numba hates array of arrays?
            list(clusters2.values),
            # The commented out below do not work for numba reasons 😿
            # list(map(set, clusters1.values)),
            # list(map(set, clusters2.values)),
            # np.array([set(x) for x in clusters1.values]),
            # np.array([set(x) for x in clusters2.values]),
            clusters1.index.values,
            clusters2.index.values,
        )


def _call_get_edges(args):
    """Helper function for Pool.map"""
    return _get_edges(*args)


@numba.njit(cache=True)
def _get_edges(clustering1, clustering2, cluster_ids1, cluster_ids2):
    edges = []
    # Cache set sizes since I mutate the sets
    ls1 = [len(c) for c in clustering1]
    ls2 = [len(c) for c in clustering2]
    cs1 = [set(c) for c in clustering1]
    cs2 = [set(c) for c in clustering2]
    for idx1 in range(len(ls1)):
        for idx2 in range(len(ls2)):
            c1 = cs1[idx1]
            c2 = cs2[idx2]
            intersect = c1.intersection(c2)
            isize = len(intersect)
            if isize > 0:
                c1.difference_update(intersect)
                c2.difference_update(intersect)
                jaccard_sim = isize / (ls1[idx1] + ls2[idx2] - isize)
                edge = (cluster_ids1[idx1], cluster_ids2[idx2], jaccard_sim)
                edges.append(edge)
    return edges


def build_graph(settings, clusters, mapping=None, nprocs=1):
    """
    Build a graph of overlapping clusters (for neighbors in parameter space).
    """
    graph = list()  # edge list
    neighbors = gen_neighbors(settings, "oou")  # TODO: Pass ordering args
    if mapping is None:
        mapping = gen_mapping(clusters)
    args = _prep_neighbors(neighbors, mapping)
    if nprocs > 1:
        # TODO: Consider replacing with joblib
        with Pool(nprocs) as p:
            edges = p.map(_call_get_edges, args)
        graph = chain.from_iterable(edges)
    else:
        graph = chain.from_iterable(map(_call_get_edges, args))
    return list(graph)


# def build_graph(settings, clusters, mapping=None):
#     """
#     Build a graph of overlapping clusters (for neighbors in parameter space).
#     """
#     graph = list()  # edge list
#     neighbors = gen_neighbors(settings, "oou") # TODO: Pass ordering args
#     if mapping is not None:
#         mapping = gen_mapping(clusters)
#     for clustering1, clustering2 in neighbors:
#         clusters1 = mapping[clustering1]
#         clusters2 = mapping[clustering2]
#         for (id1, clust1), (id2, clust2) in product(
#             clusters1.items(), clusters2.items()
#         ):
#             intersect = np.intersect1d(clust1, clust2, assume_unique=True)
#             if len(intersect) > 0:
#                 jaccard_sim = intersect.size / (
#                     clust1.size + clust2.size - intersect.size
#                 )
#                 edge = (
#                     ClusterRef(clustering1, id1),
#                     ClusterRef(clustering2, id2),
#                     jaccard_sim,
#                 )
#                 graph.append(edge)
#     return graph


def build_global_graph(settings, clusters):
    graph = list()  # edge list
    neighbors = gen_neighbors(settings, "oou")
    for clustering1, clustering2 in neighbors:
        clusters1 = clusters[clustering1]
        clusters2 = clusters[clustering2]
        score = metrics.adjusted_rand_score(clusters1, clusters2)
        edge = (clustering1, clustering2, score)
        graph.append(edge)
    return graph


def gen_mapping(clusterings):
    """
    Create a mapping from clustering and cluster to cluster contents.

    Parameters
    ----------
    clusterings : pd.DataFrame

    Returns
    -------
    pd.Series
        Mapping to cluster contents. Index is a MultiIndex with levels
        "clustering", "cluster". Values are np.arrays of cluster contents.
    """
    keys, values = _gen_mapping(clusterings.values)
    mapping = pd.Series(
        values, index=pd.MultiIndex.from_tuples(keys, names=["clustering", "cluster"])
    )
    return mapping


@numba.njit(cache=True)
def _gen_mapping(clusterings):
    n_cells, n_clusts = clusterings.shape
    keys = []
    values = []
    for clustering_id in np.arange(n_clusts):
        clustering = clusterings[:, clustering_id]
        sorted_idxs = np.argsort(clustering, kind="mergesort")
        sorted_vals = clustering[sorted_idxs]
        indices = list(np.where(np.diff(sorted_vals))[0] + 1)
        # With numba (numba doesn't have np.split)
        indices.append(n_cells)
        split_vals = list()
        cur = 0
        for i in range(len(indices)):
            nxt = indices[i]
            split_vals.append(sorted_idxs[cur:nxt])
            cur = nxt
        # Without numba
        # split_vals = np.split(sorted_idxs, indices)
        values.extend(split_vals)
        keys.extend([(clustering_id, i) for i in range(len(split_vals))])
    return keys, values


def comp_stats(comps):
    stats = pd.DataFrame({
        "n_clusts": [len(c) for c in comps],
        "intersect": [len(c.intersect) for c in comps],
        "union": [len(c.union) for c in comps]
    })
    for col in stats:
        stats[f"log1p_{col}"] = np.log1p(stats[col])
    return stats
