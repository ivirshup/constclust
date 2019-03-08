from pathlib import Path
import pandas as pd
import numpy as np
import scanpy as sc
from itertools import product

sc.settings.verbosity = 2

import sys
sys.path.append(str(Path(__file__).parent.parent))

from constclust import cluster, reconcile


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="This script subsamples samples by type from a anndata object, then clusters and reconciles those subsets."
    )
    parser.add_argument(
        "adata",
        help="Path to AnnData object. This should contain normalized expression values.",
        type=Path,
    )
    parser.add_argument(
        "output_dir", help="Directory to write output objects to.", type=Path
    )
    parser.add_argument("field", help="Which field to downsample by.", type=str)
    parser.add_argument("n_procs", help="Number of processes to use.", type=int)
    parser.add_argument("--n_reps", help="Number of repetitions for each subsample", type=int, default=3)

    args = parser.parse_args()

    outdir = args.output_dir
    field = args.field

    if not outdir.is_dir():
        outdir.mkdir()

    adata = sc.read(args.adata)

    n_field_vals = len(np.unique(adata.obs[args.field]))
    fracs = np.linspace(0.3, 0.9, 7)
    n_reps = args.n_reps

    total_runs = n_reps * len(fracs) * n_field_vals
    run = 0

    for (val, frac, seed), adata_sub in gen_subsamples(adata, field, fracs, n_reps):
        print(f"Starting clustering: {run}/{total_runs}")
        sc.pp.pca(adata_sub)
        settings, clusterings = cluster(
            adata_sub,
            n_neighbors=np.linspace(15, 150, 10, dtype=np.int),
            resolutions=np.geomspace(0.05, 20, 50),
            random_state=[0, 1, 2],
            n_procs=args.n_procs,
        )
        r = reconcile(settings, clusterings, nprocs=args.n_procs)
        settings.to_pickle(outdir / f"{val}_{frac:g}_{seed}_settings.pkl.zip")
        clusterings.to_pickle(outdir / f"{val}_{frac:g}_{seed}_clusterings.pkl.zip")
        pd.to_pickle(r, outdir / f"{val}_{frac:g}_{seed}_recon.pkl.zip")


def gen_subsamples(adata, field, fracs, n_reps):
    field_vals = np.unique(adata.obs[field])
    run = 0
    for val in field_vals:
        without_val = adata[adata.obs[field] != val].copy()
        with_val = adata[adata.obs[field] == val]
        for frac, seed in product(fracs, range(n_reps)):
            subset = sc.pp.subsample(with_val, fraction=frac, copy=True, random_state=run)
            to_cluster = without_val.concatenate(subset)
            run += 1
            yield (val, frac, seed), to_cluster


if __name__ == "__main__":
    main()
