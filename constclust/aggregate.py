from .paramspace import gen_neighbors
import numba
import numpy as np
import pandas as pd
from functools import reduce, partial
from itertools import product
import networkx as nx
from collections import namedtuple
from sklearn import metrics

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
        The reconciler which generated this component
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
        self._mapping = self._parent._mapping.loc[cluster_ids]
        clusterings = self._mapping.index.get_level_values(
            "clustering").unique()
        self.settings = self._parent.settings.loc[clusterings]
        # self.clusters = ClusterIndexer(self)
        self.intersect = reduce(partial(np.intersect1d, assume_unique=True), self._mapping.values)
        self.union = reduce(np.union1d, self._mapping.values)

    def one_hot(self, selection="intersect"):
        encoding = np.zeros(self._parent.clusterings.shape[0], dtype=bool)
        if selection == "intersect":
            encoding[self.intersect] = True
        elif selection == "union":
            encoding[self.union] = True
        else:
            raise ValueError(f"Parameter `selection` must be either 'intersect' or 'union', was '{selection}'")
    
    def __len__(self):
        return len(self._mapping)
    
    def __repr__(self):
        return f"<Component n_solutions={len(self)}, max_cells={len(self.union)}, min_cells={len(self.intersect)}>" 


class Reconciler(object):
    """
    Clusters from many clusterings and their neighbors in parameter space.

    Attributes
    ----------
    settings : pd.DataFrame
        Contains settings for all clusterings. Index corresponds to 
        `.clusterings` columns, while columns should correspond to the 
        parameters which were varied.
    clusterings : pd.DataFrame
        Contains cluster assignments for each cell, for each clustering. 
        Columns correspond to `.settings` index, while the index correspond 
        to the cells.
    _obs_names : pd.Index
        Ordered set for names of the cells. Internally they are refered to by 
        integer positions.
    _mapping : pd.Series
        Series which maps clustering and cluster id to that contents clusters.
    graph : nx.Graph
        Weighted graph. Nodes are clusters (identified by tuple of clustering
        and cluster ids). Edges connect clusters with shared contents. Weight
        is the Jaccard similarity between the contents of the clusters.
    """
    def __init__(self, settings, clusterings):
        assert all(settings.index == clusterings.columns)
        self.settings = settings
        self.clusterings = clusterings
        self._obs_names = clusterings.index
        self._mapping = gen_mapping(clusterings)
        self.clusters = ClusterIndexer(self)
        self.graph = nx.Graph()
        self.graph.add_weighted_edges_from(build_graph(settings, clusterings))
    
    def get_components(self, min_weight):
        """
        Return connected components of graph, with edges filtered by min_weight
        """
        sub_g = self.graph.edge_subgraph(filter(
            lambda x: self.graph.edges[x]["weight"] > min_weight, 
            self.graph.edges))
        comps = []
        for graph_comp in nx.connected_components(sub_g):
            comps.append(Component(self, list(graph_comp)))
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


def build_graph(settings, clusters):
    """
    Build a graph of overlapping clusters (for neighbors in parameter space).
    """
    graph = list()  # edge list
    neighbors = gen_neighbors(settings, "oou")
    mapping = gen_mapping(clusters)
    for clustering1, clustering2 in neighbors:
        clusters1 = mapping[clustering1]
        clusters2 = mapping[clustering2]
        for (id1, clust1), (id2, clust2) in product(clusters1.items(), clusters2.items()):
            intersect = np.intersect1d(clust1, clust2, 
                                       assume_unique=True)
            if len(intersect) > 0:
                jaccard_sim = intersect.size / \
                    (clust1.size + clust2.size - intersect.size)
                edge = (ClusterRef(clustering1, id1),
                        ClusterRef(clustering2, id2), jaccard_sim)
                graph.append(edge)
    return graph


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
        "clustering", "cluster". Values are np.arrays of 
    """
    keys = []
    values = []
    for clustering_id, clustering in clusterings.items():
        for cluster_id in clustering.unique():
            keys.append((clustering_id, cluster_id))
            values.append(np.where(clustering == cluster_id)[0])
    mapping = pd.Series(values,
                        index=pd.MultiIndex.from_tuples(keys, names=["clustering", "cluster"]))
    return mapping



# @numba.njit
# def _product(a, b):
#     for el1 in a:
#         for el2 in b:
#             yield (el1, el2)

# @numba.jit
# def _build_graph(neighbors, mapping):
#     graph = list()  # edge list
#     for clustering1, clustering2 in neighbors:
#         clusters1 = mapping[clustering1]
#         clusters2 = mapping[clustering2]
#         for clust1, clust2 in _product(list(clusters1.keys()), list(clusters2.keys())):
#             intersect = np.intersect1d(
#                 clusters1[clust1], clusters2[clust2], assume_unique=True)
#             if len(intersect) > 0:
#                 edge = ((clustering1, clust1), (clustering2, clust2))
#                 graph.append(edge)
#     return graph



# def gen_mapping(settings, clusters):
#     """
#     Create a mapping from parameters to clusters to contents
    
#     Args:
#         settings (pd.DataFrame)
#         clusters (pd.DataFrame)
#     """
#     mapping = dict()
#     for s in settings.itertuples(index=True):
#         solution = dict()
#         cluster = clusters[s.Index]
#         mapping[s.Index] = solution
#         for cluster_id in cluster.unique():
#             solution[cluster_id] = np.where(cluster == cluster_id)[0]
#     return mapping



