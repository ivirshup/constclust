"""
This module contains the code to cluster a single cell experiment many times.
"""

from itertools import product
import scanpy as sc
import numpy as np
import pandas as pd
import leidenalg
from multiprocessing import Pool

# TODO: Is random_state being passed to the right thing?


def cluster(adata, n_neighbors, resolutions, random_state, n_procs=1):
    """
    Parameters
    ----------
    adata : AnnData
        Object to be clustered.
    n_neighbors : Sequence[Int]
        What should the number of neighbors be?
    resolutions : Sequence[Float]
        Values for resolution for leiden algorithm
    random_state : Sequence[Int]
        Random seeds to start with.
    n_procs : Int
        Number of processes to use.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Pair of dataframes, where the first contains the settings for each 
        partitioning, and the second contains the partitionings.
    """
    n_neighbors = sorted(n_neighbors)
    resolutions = sorted(resolutions)
    random_state = sorted(random_state)
    neighbor_graphs = []
    for n, seed in product(n_neighbors, random_state):
        # Neighbor finding is already multithreaded
        sc.pp.neighbors(adata, n_neighbors=n, random_state=seed)
        g = sc.utils.get_igraph_from_adjacency(
            adata.uns['neighbors']['connectivities'], directed=True)
        neighbor_graphs.append(
            {"n_neighbors": n, "random_state": seed, "graph": g})
    cluster_jobs = []
    for graph, res in product(neighbor_graphs, resolutions):
        job = graph.copy()
        job["resolution"] = res
        cluster_jobs.append(job)
    with Pool(n_procs) as p:
        solutions = p.map(_cluster_single, cluster_jobs)
    clusters = pd.DataFrame(index=adata.obs_names)
    settings = pd.DataFrame(index=range(len(solutions)), columns=[
                            "n_neighbors", "resolution", "random_state"])
    for i in range(len(solutions)):
        job = cluster_jobs[i]
        clusters[i] = solutions[i]
        settings.loc[i, :] = job["n_neighbors"], job["resolution"], job["random_state"]
    return settings, clusters


def _cluster_single(argdict):
    part = leidenalg.find_partition(argdict["graph"],
                                    resolution_parameter=argdict["resolution"],
                                    partition_type=leidenalg.RBConfigurationVertexPartition,
                                    weights="weight")
    return np.array(part.membership)
