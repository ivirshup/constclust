from . import aggregate
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd
from pandas.api.types import is_float_dtype
import numpy as np
from .aggregate import Component

# TODO: Make it possible to have plot objects returned
# TODO: Think up visualization for the set of components


def _fix_seaborn_float_labels(axis):
    """
    Seaborn is bad at formatting floating point on axes, and fixing it there
    would probably involve learning about unicode in python2. This takes an
    axis (like, x-axis, not plt.Axes) and formats the floats better.
    """
    fmt = axis.get_major_formatter()
    fmt.seq = ["{:g}".format(float(s)) for s in fmt.seq]
    axis.set_major_formatter(fmt)


# TODO: Add title arg
# TODO: Change colorbar to discrete
# TODO: crosstab instead of pivot_table?
def component_param_range(component, x="n_neighbors", y="resolution", ax=None):
    """
    Given a component, show which parameters it's found at as a heatmap.
    """
    # Calculate colorbar
    param_states = pd.Series(component._parent.settings[[x, y]].itertuples(index=False))
    ncolors = param_states.value_counts().max() + 1
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "dummy_name",
        sns.cm.rocket(np.linspace(20, 240, ncolors, dtype=int)),  # Shifted a little so it's prettier
        ncolors
    )

    # Initialize blank array
    data = pd.pivot_table(
        component._parent.settings[[x, y]], index=y, columns=x, aggfunc=lambda x: 0
    )
    params = pd.pivot_table(
        component.settings[[x, y]], index=y, columns=x, aggfunc=len, fill_value=0
    )
    data = (data + params).fillna(0).astype(int)

    ax = sns.heatmap(
        data,
        linewidths=0.2,
        ax=ax,
        vmin=0,
        vmax=ncolors,
        cmap=cmap,
    )

    # Fix axis labels
    if is_float_dtype(component._parent.settings[x].dtype):
        _fix_seaborn_float_labels(ax.xaxis)
    if is_float_dtype(component._parent.settings[y].dtype):
        _fix_seaborn_float_labels(ax.yaxis)

    # Discretize colorbar
    cb = ax.collections[0].colorbar
    new_pos = np.stack([cb._boundaries[:-1], cb._boundaries[1:]], 1).mean(axis=1)
    assert len(new_pos) == ncolors
    cb.set_ticks(new_pos)
    cb.set_ticklabels(list(range(ncolors)))

    return ax


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
    if len(set(settings[[x, y]].itertuples(index=False))) != len(settings):
        # raise NotImplementedError("Aggregation of multiple global solutions not yet implemented.")
        settings = settings[settings["random_state"] == 0].copy()
    clusters = clusters[settings.index].copy()
    mapping = dict(zip(settings.index, settings[["n_neighbors", "resolution"]].itertuples(index=False, name=None)))
    edges = aggregate.build_global_graph(settings, clusters)
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
    ax.set_xticks(list(settings[x].unique()))
    ax.set_yticks(list(settings[y].unique()))
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
    return sns.distplot(recon.graph.es["weight"], **kwargs)
