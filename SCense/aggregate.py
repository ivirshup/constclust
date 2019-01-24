from .paramspace import gen_neighbors
import numba
import numpy as np
from itertools import product
import networkx as nx
from collections import namedtuple

ClusterRef = namedtuple("ClusterRef", ("clustering_id", "cluster_id"))


class Reconciler(object):
    """
    Clusters from many clusterings and their neighbors in parameter space.
    """
    def __init__(self, settings, clusterings):
        assert all(settings.index == clusterings.columns)
        self.settings = settings
        self.clusterings = clusterings
        self.graph = nx.Graph()
        self.graph.add_weighted_edges_from(build_graph(settings, clusterings))
    
    @property
    def cluster(self)

    def get_components(self, min_weight):
        """
        Return connected components of graph, with edges filtered by min_weight
        """
        sub_g = g.edge_subgraph(filter(
            lambda x: self.graph.edges[x]["weight"] > min_weight, 
            self.graph.edges))
        comps = list(nx.connected_components(sub_g))



def build_graph(settings, clusters):
    """
    Build a graph of overlapping clusters (for neighbors in parameter space).
    """
    graph = list()  # edge list
    neighbors = gen_neighbors(settings, "oou")
    mapping = gen_mapping(settings, clusters)
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

def nx_from_edges(edges):
    g = nx.Graph()
    for i, j, similarity in edges:
        g.add_weighted_edges_from(edges)

@numba.njit
def _product(a, b):
    for el1 in a:
        for el2 in b:
            yield (el1, el2)

@numba.jit
def _build_graph(neighbors, mapping):
    graph = list()  # edge list
    for clustering1, clustering2 in neighbors:
        clusters1 = mapping[clustering1]
        clusters2 = mapping[clustering2]
        for clust1, clust2 in _product(list(clusters1.keys()), list(clusters2.keys())):
            intersect = np.intersect1d(
                clusters1[clust1], clusters2[clust2], assume_unique=True)
            if len(intersect) > 0:
                edge = ((clustering1, clust1), (clustering2, clust2))
                graph.append(edge)
    return graph



def gen_mapping(settings, clusters):
    """
    Create a mapping from parameters to clusters to contents
    
    Args:
        settings (pd.DataFrame)
        clusters (pd.DataFrame)
    """
    mapping = dict()
    for s in settings.itertuples(index=True):
        solution = dict()
        cluster = clusters[s.Index]
        mapping[s.Index] = solution
        for cluster_id in cluster.unique():
            solution[cluster_id] = np.where(cluster == cluster_id)[0]
    return mapping


def gen_mapping2(settings, clusters):
    """
    Create a mapping from parameters to clusters to contents
    
    Args:
        settings (pd.DataFrame)
        clusters (pd.DataFrame)
    """
    mapping = dict()
    for s in settings.itertuples(index=True):
        # solution = dict()
        cluster = clusters[s.Index]
        # mapping[s.Index] = solution
        for cluster_id in cluster.unique():
            mapping
            solution[cluster_id] = np.where(cluster == cluster_id)[0]
    return mapping
