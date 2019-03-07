from pathlib import Path
import pandas as pd
import numpy as np
import scanpy as sc

sc.settings.verbosity = 2

import sys
sys.path.append(str(Path(__file__).parent.parent))

from constclust import cluster, reconcile


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="This script subsamples and clusters an AnnData object multiple times for cross validation purposes."
    )
    parser.add_argument(
        "adata",
        help="Path to AnnData object. This should contain normalized expression values.",
        type=Path,
    )
    parser.add_argument(
        "output_dir", help="Directory to write output objects to.", type=Path
    )
    parser.add_argument("n_procs", help="Number of processes to use.", type=int)

    args = parser.parse_args()

    outdir = args.output_dir

    if not outdir.is_dir():
        outdir.mkdir()

    adata = sc.read(args.adata)

    subsamples = np.linspace(0.3, 0.9, 7)
    n_reps = 3

    total_runs = n_reps * len(subsamples)
    run = 0

    for ifrac, frac in enumerate(subsamples):
        for iseed, seed in enumerate(range(n_reps)):
            run += 1
            print(f"Starting run: {run} / {total_runs}")
            adata_sub = sc.pp.subsample(adata, frac, random_state=seed**iseed, copy=True)
            sc.pp.pca(adata_sub)
            settings, clusterings = cluster(
                adata_sub,
                n_neighbors=np.linspace(15, 150, 10, dtype=np.int),
                resolutions=np.geomspace(0.05, 20, 50),
                random_state=[0, 1, 2],
                n_procs=args.n_procs,
            )
            r = reconcile(settings, clusterings, nprocs=args.n_procs)
            settings.to_pickle(outdir / f"{frac:g}_{seed}_settings.pkl.zip")
            clusterings.to_pickle(outdir / f"{frac:g}_{seed}_clusterings.pkl.zip")
            pd.to_pickle(r, outdir / f"{frac:g}_{seed}_recon.pkl.zip")


if __name__ == "__main__":
    main()
