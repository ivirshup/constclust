from . import aggregate
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import scanpy as sc
from anndata import AnnData
import pandas as pd
from pandas.api.types import is_float_dtype, is_numeric_dtype, is_categorical_dtype
import numpy as np
from .aggregate import Component
from itertools import product
from typing import Optional

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
# TODO: crosstab instead of pivot_table?
def component_param_range(
    component: Component,
    x: str = "n_neighbors",
    y: str = "resolution",
    ax: Optional[mpl.axis.Axis] = None
) -> mpl.axis.Axis:
    """
    Given a component, show which parameters it's found at as a heatmap.

    Params
    ------
    component
        The component to plot.
    x
        The parameter for the x axis.
    y
        The parameter to place on the y axis.
    ax
        Optional axis to plot on.

    Example
    -------
    >>> comps = reconciler.get_comps(0.9)
    >>> plotting.component_param_range(comps[0])
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


def umap_cells(cells, adata, ax=None, umap_kwargs={}):
    if hasattr(cells, "keys") and hasattr(cells, "values"):
        cells = pd.Series(cells)
    else:
        cells = pd.Series(1, index=cells)
    cell_values = pd.Series(0, index=adata.obs_names, dtype=float)
    cell_values[cells.index] += cells
    adata.obs["_tmp"] = cell_values
    sc.pl.umap(adata, color="_tmp", ax=ax, title="UMAP", **umap_kwargs)
    adata.obs.drop(columns="_tmp", inplace=True)


def umap(component, adata, ax=None, umap_kwargs={}):
    # TODO: Views should have parents, which is where I should get my obs names
    cell_names = component._parent._obs_names
    # if not all(cell_names.isin(adata.obs_names)):
        # raise ValueError("Counldn't find all cells in component's parent in adata.")
    cell_value = pd.Series(0, index=adata.obs_names, dtype=float)
    # present_freqs = component.cell_frequency
    # cell_value[present_freqs.index] = present_freqs
    for cluster in component.cluster_ids:
        cell_value[component._parent._mapping.iloc[cluster]] += 1
    cell_value = cell_value / cell_value.max()
    adata.obs["_tmp"] = cell_value
    if (len(cell_names) < len(adata.obs_names)):
        # Take view
        sc.pl.umap(adata[cell_names, :], color="_tmp", ax=ax, title="UMAP", **umap_kwargs)
    else: 
        sc.pl.umap(adata[cell_names, :], color="_tmp", ax=ax, title="UMAP", **umap_kwargs)
    adata.obs.drop(columns="_tmp", inplace=True)


def global_stability(settings, clusters, x="n_neighbors", y="resolution", cmap=sns.cm.rocket, ax=None):
    # This should probably aggregate, currently do hacky thing of just subsetting
    if len(set(settings[[x, y]].itertuples(index=False))) != len(settings):
        # raise NotImplementedError("Aggregation of multiple global solutions not yet implemented.")
        settings = settings[settings["random_state"] == 0].copy()

    xlabs = sorted(settings[x].unique())
    ylabs = sorted(settings[y].unique())

    clusters = clusters[settings.index].copy()
    mapping = dict(zip(settings.index, settings[[x, y]].itertuples(index=False, name=None)))
    # + .5 might make it align better with local plot
    xpos = np.arange(len(xlabs))# + .5
    ypos = np.arange(len(ylabs))# + .5
    pos_map = {}
    for (xlab, x), (ylab, y) in product(zip(xlabs, xpos), zip(ylabs, ypos)):
        pos_map[(xlab, ylab)] = (x, y)
    edges = aggregate.build_global_graph(settings, clusters)

    lines = []
    colors = []
    for edge in edges:
        color = edge[2]
        line = [pos_map[mapping[edge[0]]], pos_map[mapping[edge[1]]]]
        colors.append(color)
        lines.append(line)
    lc = mpl.collections.LineCollection(lines, cmap=cmap)
    lc.set_array(np.array(colors))
    if ax is not None:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots()
    ax.add_collection(lc)

    xstep = _tick_step(ax, xlabs, 0)
    ystep = _tick_step(ax, ylabs, 1)
    ax.set(xticks=xpos[::xstep], yticks=ypos[::ystep])
    fmat = np.vectorize("{:g}".format)
    ax.set_xticklabels(labels=fmat(xlabs[::xstep]))
    ax.set_yticklabels(labels=fmat(ylabs[::ystep]))
    ax.invert_yaxis()
    ax.autoscale()
    ax.set_frame_on(False)
    cb = fig.colorbar(lc, ax=ax)
    cb.outline.set_visible(False)


def component(component: Component,
              adata: AnnData,
              x: str = "n_neighbors",
              y: str = "resolution",
              plot_global: bool = False,
              aspect: float = None,
              umap_kwargs: dict = {}
):
    """
    Plot stability and UMAP for component.

    Params
    ------
    component
        Component object to plot.
    adata
        AnnData to use for plotting UMAP. Should have same cell names as `component`s parent `Reconciler`.
    x
        Parameter to plot on the X-axis.
    y
        Parameter to plot on the Y-axis.
    aspect
        Aspect ratio of entire plot. Defaults to 1/2.
    umap_kwargs
        Arguments passed to `sc.pl.umap`.
    """
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

# Modified from seaborn
def _tick_step(ax, labels, axis):
    transform = ax.figure.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(transform)
    size = [bbox.width, bbox.height][axis]
    axis = [ax.xaxis, ax.yaxis][axis]
    tick, = axis.set_ticks([0])
    fontsize = tick.label.get_size()
    max_ticks = int(size // (fontsize / 72))
    tick_every = len(labels) // max_ticks + 1
    tick_every = 1 if tick_every == 0 else tick_every
    return tick_every