from .paramspace import gen_neighbors
from .utils import reverse_series_map, pairs_to_dict

from typing import List, Collection
import igraph
import numba
import numpy as np
import pandas as pd
from pandas.api.types import (
    is_categorical_dtype,
    is_integer_dtype,
    is_bool_dtype,
    is_numeric_dtype
)
from sklearn import metrics

from functools import reduce, partial
from itertools import chain
from multiprocessing import Pool

# TODO: Generalize documentation
# TODO: Should _mapping be mapping? Would I want to give names then?


class Component(object):
    """
    A connected component from a `Reconciler`

    Attributes
    ----------
    _parent : Reconciler
        The ``Reconciler`` which generated this component.
    settings : pandas.DataFrame
        Subset of parents settings. Contains only settings for clustering
        which appear in this component.
    cluster_ids : numpy.ndarray
        Which clusters are in this component.
    intersect : numpy.ndarray[int]
        Intersection of samples in this component.
    intersect_names : numpy.ndarray[str]
        Names of samples in the intersection of this component.
    union : numpy.ndarray[int]
        Union of samples in this component.
    union_names : numpy.ndarray[str]
        Names of samples in the union of this component.
    """

    def __init__(self, reconciler, cluster_ids):
        self._parent = reconciler
        self.cluster_ids = np.sort(cluster_ids)
        # TODO: Should a component found via a subset only return the clusters in the subset?
        clusterings = self._parent._mapping.index.get_level_values("clustering")[cluster_ids]  # Should already be unique
        self.settings = self._parent.settings.loc[clusterings]
        cells = self._parent._mapping.iloc[cluster_ids].values
        # TODO: I could just figure out how big these are, and make their contents lazily evaluated. I think this would be a pretty big speed up for interactive work.
        self.intersect = reduce(partial(np.intersect1d, assume_unique=True), cells)
        self.union = reduce(np.union1d, cells)
        # self._value_counts = None

    @property
    def intersect_names(self):
        return self._parent._obs_names[self.intersect].values

    @property
    def union_names(self):
        return self._parent._obs_names[self.union].values

    # @property
    # def cell_frequency(self):
    #     return self.value_counts() / len(self.settings)

    # def value_counts(self):
    #     if self._value_counts is None:
    #         cell_ids, counts = pd.value_counts(
    #             self._parent._mapping.iloc[self.cluster_ids].values,
    #             sort=False
    #         )
    #         self._value_counts = pd.Series(data=counts, values=cell_ids)
    #     value_counts = self._value_counts.copy()
    #     value_counts.index = self.union_names  # Is this slow?
    #     return value_counts

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
        return len(self.cluster_ids)

    def __repr__(self):
        return (
            f"<Component n_solutions={len(self)}, "
            f"max_cells={len(self.union)}, "
            f"min_cells={len(self.intersect)}>"
        )


class ReconcilerBase(object):
    """
    Base type for reconciler.

    Has methods for subsetting implemented, providing data is up to subclass.
    """
    def __repr__(self):
        kls = self.__class__.__name__
        n_cells = len(self.clusterings)
        n_clusterings, n_params = self.settings.shape
        n_clusters = len(self._mapping)
        return f"<{kls} {n_clusterings} clusterings, {n_clusters} clusters, {n_cells} cells>"

    @property
    def cluster_ids(self):
        return self._mapping.index.get_level_values("cluster")

    # TODO should I remove this?
    def get_param_range(self, clusters):
        """
        Given a set of clusters, returns the range of parameters for which they were calculated.

        Parameters
        ----------
        clusters : Collection[Int]
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
        Take subset of Reconciler, where only ``clusterings_to_keep`` are present.

        Reduces size of both ``.settings`` and ``.clusterings``.

        Parameters
        ----------
        clusterings_to_keep
            Indexer into ``Reconciler.settings``. Anything that should give the correct
            result for ``reconciler.settings.loc[clusterings_to_keep]``.

        Returns
        -------
        ``ReconcilerSubset``
        """
        clusterings_to_keep = self.settings.loc[clusterings_to_keep].index.values
        new_settings = self.settings.loc[clusterings_to_keep]  # Should these be copies?
        new_clusters = self.clusterings.loc[:, clusterings_to_keep]  # This is a copy?
        new_mapping = self._mapping[np.isin(self._mapping.index.get_level_values("clustering"), clusterings_to_keep)]
        return ReconcilerSubset(
            self._parent, new_settings, new_clusters, new_mapping, self.graph
        )

    def subset_cells(self, cells_to_keep):
        """
        Take subset of Reconciler, where only ``cells_to_keep`` are present.

        Parameters
        ----------
        cells_to_keep :
            Indexer into ``Reconciler.clusterings``. Anything that should give the correct
            result for ``reconciler.clusterings.loc[cells_to_keep]``.

        Returns
        -------
        ``ReconcilerSubset``
        """
        intmap = reverse_series_map(self._obs_names)
        cells_to_keep = intmap[self.clusterings.loc[cells_to_keep].index.values]
        new_clusterings = self.clusterings.iloc[cells_to_keep, :]
        new_mapping = gen_mapping(new_clusterings).apply(lambda x: cells_to_keep.values[x])
        # TODO: test how long this takes
        new_rec = ReconcilerSubset(
            self._parent,
            self.settings,
            new_clusterings,
            new_mapping,
            self.graph,
        )
        return new_rec

    # TODO: Should these describe methods only work on full reconciler?
    # Is it intuitive enough that the results will change when subset by cells?
    def describe_clusters(self, log1p: bool = False) -> pd.DataFrame:
        """
        Describe the clusters in this Reconciler.

        Params
        ------
        log1p
            Whether to also return log transformed values for numeric cols.

        Returns
        -------

        DataFrame containing some summary statistics on the clusters in this reconciler. 
        Good for plotting.

        Example
        -------
        >>> import hvplot.pandas
        >>> clusters = reconciler.describe_clusters(log1p=True)
        >>> clusters.hvplot.scatter("log1p_resolution", "log1p_n_obs", datashade=True, dynspread=True)
        """
        lens = pd.DataFrame(self._mapping.apply(len), columns=["n_obs"])
        lens = pd.merge(lens, self.settings,
                        left_on="clustering", right_index=True)
        if log1p:
            for k in lens.columns[lens.dtypes.apply(is_numeric_dtype)]:
                lens[f"log1p_{k}"] = np.log1p(lens[k])
        return lens

    def describe_clusterings(self) -> pd.DataFrame:
        """
        Convenience function to generate summary statistics for clusterings in a reconciler.

        Example
        -------
        >>> import seaborn as sns
        >>> clusterings = reconciler.describe_clusterings()
        >>> sns.jointplot(data=clusterings, x="resolution", y="max_n_obs")
        """
        # r = r._parent
        m = self._mapping
        cdf = pd.DataFrame(index=pd.Index(self.settings.index, name="clustering"))
        cdf["n_clusters"] = m.index.get_level_values("clustering").value_counts(sort=False).sort_index()
        ls = m.apply(len)
        gb = ls.groupby(level="clustering")
        cdf["min_n_obs"] = gb.min()
        cdf["max_n_obs"] = gb.max()
        cdf["mean_n_obs"] = gb.mean()
        cdf["n_singletons"] = (ls == 1).groupby("clustering").sum()
        cdf = cdf.join(self.settings)
        return cdf


class ReconcilerSubset(ReconcilerBase):
    """
    Subset of a Reconciler

    Attributes
    ----------
    _parent: Reconciler
        `Reconciler` this subset was derived from.
    settings: pandas.DataFrame
        Settings for clusterings in this subset.
    clusterings: pandas.DataFrame
        Clusterings contained in this subset.
    graph: igraph.Graph
        Reference to graph from parent.
    cluster_ids: np.ndarray[int]
        Integer ids of all clusters in this subset.
    _mapping: pandas.Series
        ``pd.Series`` with a ``MultiIndex``. Unlike the ``_mapping`` from ``Reconciler``,
        this does not necessarily have all clusters, so ranges of clusters cannot be
        assumed to be contiguous. Additionally, you can't just index into this with
        ``cluster_ids`` as positions.
    _obs_names: ``pd.Series``
        Maps from integer position to input cell name.
    """

    def __init__(self, parent, settings, clusterings, mapping, graph):
        # assert isinstance(parent, Reconciler)  #  Can fail when using autoreloader
        assert all(settings.index == clusterings.columns)
        assert all(mapping.index.get_level_values("clustering").unique().isin(settings.index))
        self._parent = parent
        self.settings = settings
        self.clusterings = clusterings
        self._obs_names = parent._obs_names[parent._obs_names.isin(clusterings.index)]  # Ordering
        # self._obs_names = self.clusterings.index
        self._mapping = mapping
        self.graph = graph

    def find_contained_components(self, min_presence: float, min_weight: float = 0.9, min_cells: int = 2):
        """
        Find components contained in a subset.
        """
        m1 = self._mapping
        m2 = self._parent._mapping
        presence = m1.apply(len) / m2.loc[m1.index].apply(len)
        clusters = np.array(presence.index.get_level_values("cluster")[presence > min_presence])
        return self.find_components(min_weight, clusters, min_cells)

    def find_components(self, min_weight, clusters, min_cells=2) -> List[Component]:
        # This is actually a very slow check. Maybe I should wait on it?
        # TODO: I've modified the code, check speed and consider if this is really what I want.
        # if not np.isin(clusters, self._mapping.get_level_values("cluster")).all():
        #     raise ValueError("")
        return self._parent.find_components(min_weight, clusters, min_cells=min_cells)

    def get_components(self, min_weight: float, min_cells: int = 2) -> List[Component]:
        """
        Return connected components of graph, with edges filtered by min_weight.

        Parameters
        ----------
        min_weight
            Minimum edge weight for inclusion of a clustering.
        min_cells
            Minimum cells a component should have.
        """
        clusters = self._mapping.index.get_level_values("cluster")
        return self._parent.find_components(min_weight, clusters, min_cells)


class Reconciler(ReconcilerBase):
    """
    Collects and reconciles many clusterings by local (in parameter space)
    stability.

    Attributes
    ----------
    settings : pandas.DataFrame
        Contains settings for all clusterings. Index corresponds to
        `.clusterings` columns, while columns should correspond to the
        parameters which were varied.
    clusterings : pandas.DataFrame
        Contains cluster assignments for each cell, for each clustering.
        Columns correspond to `.settings` index, while the index correspond
        to the cells. Each cluster is encoded with a unique cluster id.
    graph : igraph.Graph
        Weighted graph. Nodes are clusters (identified by unique cluster id
        integer, same as in `.clusterings`). Edges connect clusters with shared
        contents. Weight is the Jaccard similarity between the contents of the
        clusters.
    cluster_ids : numpy.ndarray[int]
        Integer ids of all clusters in this Reconciler.
    _obs_names : pandas.Index
        Ordered set for names of the cells. Internally they are refered to by
        integer positions.
    _mapping : pandas.Series
        ``pd.Series`` with a ``MultiIndex``. Index has levels ``clustering``
        and ``cluster``. Each position in index should have a unique value at
        level "cluster", which corresponds to a cluster in the clustering 
        dataframe. Values are ``np.arrays`` with indices of cells in relevant
        cluster. This should be considered immutable, though this is not the
        case for ``ReconcilerSubset``s.
    """

    def __init__(self, settings: pd.DataFrame, clusterings: pd.DataFrame, mapping: pd.Series, graph: igraph.Graph):
        assert all(settings.index == clusterings.columns)
        self._parent = self  # Kinda hacky, could maybe remove
        self.settings = settings
        self.clusterings = clusterings
        self._obs_names = pd.Series(
            clusterings.index.values, index=np.arange(len(clusterings))
        )
        self._mapping = mapping
        self.graph = graph

    # TODO: Allow passing function for clusters
    def find_components(
        self, min_weight: float, clusters: Collection[int], min_cells: int = 2
    ) -> List[Component]:
        """
        Return components from filtered graph which contain specified clusters.

        Parameters
        ----------
        min_weight : ``float``
            Minimum weight for edges to be kept in graph. Should be over 0.5.
        clusters : ``np.array[int]``
            Clusters which you'd like to search from.
        """
        # Subset graph
        over_cutoff = np.where(np.array(self.graph.es["weight"]) >= min_weight)[0]
        sub_g = self.graph.subgraph_edges(over_cutoff, delete_vertices=True)
        # Mapping from cluster_id to node_id
        cidtonode = dict(map(tuple, map(reversed, enumerate(sub_g.vs["cluster_id"]))))
        # nodemap = np.frompyfunc(cidtonode.get, 1, 1)

        # Get mapping from clustering to clusters
        # Because this is an Reconciler object, we can just index by position into the mapping
        clusteringtocluster = {
            k: np.array(v) for k, v in pairs_to_dict(iter(self._mapping[clusters].index)).items()
        }
        # Only look for clusters in graph
        clusters = np.intersect1d(clusters, sub_g.vs["cluster_id"], assume_unique=True)

        # Create sieve
        to_visit = np.zeros(self.graph.vcount(), dtype=bool)
        to_visit[clusters] = True

        # Figure out the clusterings I want to visit
        # I know that no two clusters from one clustering can be in the same component (should probably put limit on jaccard score to ensure this)
        # This means I can explore each component at the same time, or at least know I won't be exploring the same one twice
        frame = self._mapping.index.to_frame().reset_index(drop=True)
        clusterings = frame.loc[lambda x: x["cluster"].isin(clusters)]["clustering"]
        clustering_queue = clusterings.tolist()

        components = list()
        while len(clustering_queue) > 0:
            clustering = clustering_queue.pop()
            foundclusters = clusteringtocluster[clustering]
            foundclusters = foundclusters[np.where(to_visit[foundclusters])[0]]  # Filtering
            if not any(to_visit[foundclusters]):
                continue
            for cluster in foundclusters:
                component = np.sort(
                    [x["cluster_id"] for x in sub_g.bfsiter(cidtonode[cluster])]
                )
                components.append(component)
                to_visit[component] = False
        # Format and return components
        out = [
            Component(self, c) for c in components
        ]  # This is slow, probably due to finding the union and intersect
        return sorted(
            filter(lambda x: len(x.intersect) >= min_cells, out),
            key=lambda x: (len(x.settings), len(x.intersect)),
            reverse=True,
        )

    def get_components(self, min_weight: float, min_cells: int = 2):
        """
        Return connected components of graph, with edges filtered by ``min_weight``.

        Parameters
        ----------
        min_weight
            Minimum weight for edges to be kept in graph. Should be in range ``[0.5, 1]``.
        min_cells
            Minimum number of cells in a component.

        Returns
        -------
        ``List[Component]``:
           All components from ``Reconciler``, sorted by number of clusterings.
        """
        over_cutoff = np.where(np.array(self.graph.es["weight"]) >= min_weight)[0]
        # If vertices are deleted, indices (ids) change
        sub_g = self.graph.subgraph_edges(over_cutoff, delete_vertices=True)
        comps = []
        # Filter out 1 node components
        cids = np.array(sub_g.vs["cluster_id"])
        for graph_comp in filter(lambda x: len(x) > 1, sub_g.components()):
            comp = Component(self, cids[list(graph_comp)])
            if len(comp.intersect) < min_cells:
                continue
            comps.append(comp)
        comps.sort(key=lambda x: (len(x.settings), len(x.intersect)), reverse=True)
        return comps


def reconcile(
    settings: pd.DataFrame, clusterings: pd.DataFrame, nprocs: int = 1
) -> Reconciler:
    """
    Constructor for reconciler object.

    Parameters
    ----------
    settings
        Parameterizations of each clustering.
    clusterings
        Assignments from each clustering.
    nprocs
        Number of processes to use

    Example
    -------
    >>> params, clusterings = cluster(adata, ... )
    >>> reconciler = reconcile(params, clusterings)
    """
    assert all(
        settings.index == clusterings.columns
    )  # I should probably save these, right?
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
    # TODO: This can just get deleted, right?
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
        vertex_attrs={"cluster_id": np.arange(len(mapping))},
        edge_attrs={"weight": list(k for i, j, k in edges)},
    )
    return Reconciler(settings, clusterings, mapping, graph)


def _prep_neighbors(neighbors, clusterings):
    for i, j in neighbors:
        yield clusterings.values[:, i], clusterings.values[:, j]


def _call_get_edges(args):
    """Helper function for Pool.map"""
    return _get_edges(*args)


@numba.njit(cache=True)
def _get_edges(clustering1: np.array, clustering2: np.array):
    edges = []
    offset1 = clustering1.min()
    offset2 = clustering1.min()
    # Because of how I've done unique node names, potentially this
    # could be done in a more generic way by creating a mapping here.
    offset_clusts1 = clustering1 - offset1
    offset_clusts2 = clustering2 - offset2
    # Allocate coincidence matrix
    nclusts1 = offset_clusts1.max() + 1
    nclusts2 = offset_clusts2.max() + 1
    coincidence = np.zeros((nclusts1, nclusts2))
    # Allocate cluster size arrays
    ncells1 = np.zeros(nclusts1)
    ncells2 = np.zeros(nclusts2)
    # Compute lengths of the intersects
    for cell in range(len(clustering1)):
        c1 = offset_clusts1[cell]
        c2 = offset_clusts2[cell]
        coincidence[c1, c2] += 1
        ncells1[c1] += 1
        ncells2[c2] += 1
    for cidx1, cidx2 in np.ndindex(coincidence.shape):
        isize = coincidence[cidx1, cidx2]
        if isize < 1:
            continue
        jaccard_sim = isize / (ncells1[cidx1] + ncells2[cidx2] - isize)
        edge = (cidx1 + offset1, cidx2 + offset2, jaccard_sim)
        edges.append(edge)
    return edges


def build_graph(settings, clusters, mapping=None, nprocs=1):
    """
    Build a graph of overlapping clusters (for neighbors in parameter space).
    """
    graph = list()  # edge list
    neighbors = gen_neighbors(settings, "oou")  # TODO: Pass ordering args
    # if mapping is None:
    # mapping = gen_mapping(clusters)
    args = _prep_neighbors(neighbors, clusters)
    if nprocs > 1:
        # TODO: Consider replacing with joblib
        with Pool(nprocs) as p:
            edges = p.map(_call_get_edges, args)
        graph = chain.from_iterable(edges)
    else:
        graph = chain.from_iterable(map(_call_get_edges, args))
    return list(graph)


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
    clusterings : pandas.DataFrame

    Returns
    -------
    pandas.Series
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
        unique_vals = np.unique(clustering)
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
        keys.extend([(clustering_id, u) for u in unique_vals])
    return keys, values


def comp_stats(comps):
    stats = pd.DataFrame(
        {
            "n_clusts": [len(c) for c in comps],
            "intersect": [len(c.intersect) for c in comps],
            "union": [len(c.union) for c in comps],
        }
    )
    for col in stats:
        stats[f"log1p_{col}"] = np.log1p(stats[col])
    return stats
