import seaborn as sns
import scanpy as sc
import pandas as pd
import numpy as np
from .aggregate import Component

#TODO: Make it possible to have plot objects returned
#TODO: Think up visualization for the set of components

#TODO: Add title arg, & fix formatting. 
#TODO: Change colorbar to discrete
#TODO: crosstab instead of pivot_table?
def component_param_range(component, x, y, ax=None):
    """
    Given a component, show which parameters it's found at as a heatmap.
    """
    # Initialize blank array
    data = pd.pivot_table(
        component._parent.settings[[x, y]], index=y, columns=x, aggfunc=lambda x: 0)
    params = pd.pivot_table(
        component.settings[[x, y]], index=y, columns=x, aggfunc=len, fill_value=0)
    # Maybe use this to calculate color bar?
    # all_params = pd.pivot_table(
        # component._parent.settings, index=y, columns=x, aggfunc=len, fill_value=0)
    data = (data + params).fillna(0).astype(int)
    sns.heatmap(data, annot=True, linewidths=.5, ax=ax)   # fmt="d",

def umap(component, adata, ax=None):
    cell_value = pd.Series(0, index=adata.obs_names, dtype=float)
    for cluster in component._mapping:
        cell_value[cluster] += 1
    cell_value = cell_value / cell_value.max()
    adata.obs["_tmp"] = cell_value
    sc.pl.umap(adata, color="_tmp", ax=ax)
    adata.obs.drop(columns="_tmp", inplace=True)


def plot_component(component, adata, x="n_neighbors", y="resolution"):
    fig = plt.figure()
    gs = fig.add_gridspec(1, 2)
    umap_ax = fig.add_subplot(gs[0, 1])
    heatmap_ax = fig.add_subplot(gs[0, 0])
    component_param_range(component, x, y, ax=heatmap_ax)
    umap(component, adata, ax=umap_ax)
    return fig
# gs = gridspec.GridSpec(nrows=n_panels_y,
#                        ncols=n_panels_x,
#                        left=left,
#                        right=1-(n_panels_x-1)*left-0.01/n_panels_x,
#                        bottom=bottom,
#                        top=1-(n_panels_y-1)*bottom-0.1/n_panels_y,
#                        hspace=hspace,
#                        wspace=wspace)
