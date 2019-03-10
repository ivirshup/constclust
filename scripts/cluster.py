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
        description="This script clusters and reconciles an AnnData object."
    )
    parser.add_argument(
        "adata",
        help="Path to AnnData object. This should contain normalized expression values.",
        type=Path,
    )
    parser.add_argument(
        "output_dir", help="Directory to write output objects to.", type=Path
    )
    parser.add_argument("name", help="Name to prepend output files with.", type=str)
    parser.add_argument("n_procs", help="Number of processes to use.", type=int)

    args = parser.parse_args()

    outdir = args.output_dir

    if not outdir.is_dir():
        outdir.mkdir()

    adata = sc.read(args.adata)

    sc.pp.pca(adata)
    settings, clusterings = cluster(
        adata,
        n_neighbors=np.linspace(15, 150, 10, dtype=np.int),
        resolutions=np.geomspace(0.05, 20, 50),
        random_state=[0, 1, 2],
        n_procs=args.n_procs,
    )
    r = reconcile(settings, clusterings, nprocs=args.n_procs)
    settings.to_pickle(outdir / f"{args.name}_settings.pkl.zip")
    clusterings.to_pickle(outdir / f"{args.name}_clusterings.pkl.zip")
    pd.to_pickle(r, outdir / f"{args.name}_recon.pkl.zip")


if __name__ == "__main__":
    main()
