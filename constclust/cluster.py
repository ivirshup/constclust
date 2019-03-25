"""
This module contains the code to cluster a single cell experiment many times.
"""

from itertools import product
import scanpy as sc
import numpy as np
import pandas as pd
import leidenalg
from multiprocessing import Pool
from functools import partial

# TODO: Is random_state being passed to the right thing?


def cluster(
    adata,
    n_neighbors,
    resolutions,
    random_state,
    n_procs=1,
    neighbor_kwargs={},
    leiden_kwargs={}
):
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
    neighbor_kwargs : dict, optional
        Key word arguments to pass to all calls to `sc.pp.neighbors`. For
        example: `{"use_rep": "X"}`.
    leiden_kwargs : dict, optional
        Key word argument to pass to all calls to `leidenalg.find_partition`.
        For example, `{"partition_type": leidenalg.CPMVertexPartition}`.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Pair of dataframes, where the first contains the settings for each
        partitioning, and the second contains the partitionings.
    """
    leiden_kwargs = leiden_kwargs.copy()
    neighbor_kwargs = neighbor_kwargs.copy()

    if "partition_type" not in leiden_kwargs:
        leiden_kwargs["partition_type"] = leidenalg.RBConfigurationVertexPartition
    if "weights" not in leiden_kwargs:
        leiden_kwargs["weights"] = "weight"

    def _check_params(kwargs, vals, arg_name):
        for val in vals:
            if val in kwargs:
                raise ValueError(f"You cannot pass value for key `{val}` in `{arg_name}`")

    _check_params(neighbor_kwargs, ["adata", "n_neighbors", "random_state"], "neighbor_kwargs")
    _check_params(leiden_kwargs, ["graph", "resolution_parameter", "resolution"], "leiden_kwargs")

    n_neighbors = sorted(n_neighbors)
    resolutions = sorted(resolutions)
    random_state = sorted(random_state)
    neighbor_graphs = []
    for n, seed in product(n_neighbors, random_state):
        # Neighbor finding is already multithreaded (sorta)
        sc.pp.neighbors(adata, n_neighbors=n, random_state=seed, **neighbor_kwargs)
        g = sc.utils.get_igraph_from_adjacency(
            adata.uns["neighbors"]["connectivities"], directed=True
        )
        neighbor_graphs.append({"n_neighbors": n, "random_state": seed, "graph": g})
    cluster_jobs = []
    for graph, res in product(neighbor_graphs, resolutions):
        job = graph.copy()
        job.update({"resolution": res})
        cluster_jobs.append(job)
    _cluster_single_kwargd = partial(_cluster_single, leiden_kwargs=leiden_kwargs)
    with Pool(n_procs) as p:
        # solutions = p.map(_cluster_single, cluster_jobs)
        solutions = p.map(_cluster_single_kwargd, cluster_jobs)
    clusters = pd.DataFrame(index=adata.obs_names)
    for i, clustering in enumerate(solutions):
        clusters[i] = clustering
    settings_iter = (
        (job["n_neighbors"], job["resolution"], job["random_state"])
        for job in cluster_jobs
    )
    settings = pd.DataFrame.from_records(
        settings_iter,
        columns=["n_neighbors", "resolution", "random_state"],
        index=range(len(solutions)),
    )
    return settings, clusters


def _cluster_single(argdict, leiden_kwargs):
    part = leidenalg.find_partition(
        argdict["graph"],
        resolution_parameter=argdict["resolution"],
        **leiden_kwargs
    )
    return np.array(part.membership)
