from . import aggregate
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd
import networkx as nx
import numpy as np
from .aggregate import Component

# TODO: Make it possible to have plot objects returned
# TODO: Think up visualization for the set of components


# TODO: Add title arg, & fix formatting.
# TODO: Change colorbar to discrete
# TODO: crosstab instead of pivot_table?
def component_param_range(component, x, y, ax=None):
    """
    Given a component, show which parameters it's found at as a heatmap.
    """
    # Initialize blank array
    data = pd.pivot_table(
        component._parent.settings[[x, y]], index=y, columns=x, aggfunc=lambda x: 0
    )
    params = pd.pivot_table(
        component.settings[[x, y]], index=y, columns=x, aggfunc=len, fill_value=0
    )
    # Maybe use this to calculate color bar?
    # all_params = pd.pivot_table(
    # component._parent.settings, index=y, columns=x, aggfunc=len, fill_value=0)
    data = (data + params).fillna(0).astype(int)
    sns.heatmap(data, annot=True, linewidths=0.5, ax=ax)  # fmt="d",


def umap(component, adata, ax=None, umap_kwargs={}):
    cell_value = pd.Series(0, index=adata.obs_names, dtype=float)
    for cluster in component._mapping:
        cell_value[cluster] += 1
    cell_value = cell_value / cell_value.max()
    adata.obs["_tmp"] = cell_value
    sc.pl.umap(adata, color="_tmp", ax=ax, title="UMAP", **umap_kwargs)
    adata.obs.drop(columns="_tmp", inplace=True)


def global_stability(settings, clusters, x="n_neighbors", y="resolution", cmap=sns.cm.rocket, ax=None):
    # This should probably aggregate, currently do hacky thing of just subsetting
    simple_settings = settings[settings["random_state"] == 0]
    simple_clusters = clusters[simple_settings.index]
#     if len(set(settings[[x,y]].itertuples(index=False))) != len(settings):
#         raise NotImplementedError("Aggregation of multiple plots not yet implemented.")
    mapping = dict(zip(simple_settings.index, simple_settings[["n_neighbors", "resolution"]].itertuples(index=False, name=None)))
    edges = aggregate.build_global_graph(simple_settings, simple_clusters)
    lines = []
    colors = []
    for edge in edges:
        color = edge[2]
        line = [mapping[edge[0]], mapping[edge[1]]]
        colors.append(color)
        lines.append(line)
    lc = mpl.collections.LineCollection(lines, cmap=cmap)
    lc.set_array(np.array(colors))
    if ax is not None:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.set_xticks(list(simple_settings[x].unique()))
    ax.set_yticks(list(simple_settings[y].unique()))
    ax.invert_yaxis()
    ax.autoscale()
    ax.set_frame_on(False)
    cb = fig.colorbar(lc, ax=ax)
    cb.outline.set_visible(False)


def plot_component(component,
                   adata,
                   x="n_neighbors",
                   y="resolution",
                   plot_global=False,
                   aspect=None,
                   umap_kwargs={}):
    if aspect is None:
        if plot_global:
            aspect = 1/3
        else:
            aspect = 1/2
    fig = plt.figure(figsize=mpl.figure.figaspect(aspect))
    if plot_global:
        reconciler = component._parent
        gs = fig.add_gridspec(1, 3)
        global_ax = fig.add_subplot(gs[0, 1])
        global_stability(reconciler.settings, reconciler.clusterings, x, y, ax=global_ax)
    else:
        gs = fig.add_gridspec(1, 2)
    heatmap_ax = fig.add_subplot(gs[0, 0])
    umap_ax = fig.add_subplot(gs[0, -1])
    component_param_range(component, x, y, ax=heatmap_ax)
    umap(component, adata, ax=umap_ax, umap_kwargs=umap_kwargs)
    return fig


def edge_weight_distribution(recon, **kwargs):
    return sns.distplot(nx.to_pandas_edgelist(recon.graph)["weight"], **kwargs)
# gs = gridspec.GridSpec(nrows=n_panels_y,
#                        ncols=n_panels_x,
#                        left=left,
#                        right=1-(n_panels_x-1)*left-0.01/n_panels_x,
#                        bottom=bottom,
#                        top=1-(n_panels_y-1)*bottom-0.1/n_panels_y,
#                        hspace=hspace,
#                        wspace=wspace)
