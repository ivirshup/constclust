from constclust import cluster
import scanpy as sc

# Just testing for errors being thrown. Could probably chuck a regression test on here.


def test_clustering():
    adata = sc.datasets.pbmc68k_reduced()
    cluster(adata, n_neighbors=[15, 30], resolutions=[1.0, 2.0], random_state=[0, 1])


def test_clustering_kwargs():
    adata = sc.datasets.pbmc68k_reduced()
    cluster(
        adata,
        n_neighbors=[15, 30],
        resolutions=[1.0, 2.0],
        random_state=[0, 1],
        leiden_kwargs={"weights": None},
        neighbor_kwargs={"use_rep": "X", "metric": "cosine"},
    )
