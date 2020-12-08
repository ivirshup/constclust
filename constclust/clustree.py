import base64
from collections import Counter
from collections.abc import Mapping, Iterable
from functools import singledispatch
from itertools import product, chain
from io import BytesIO

from bokeh.plotting import from_networkx
from bokeh.models.graphs import NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.models import (
    Plot,
    MultiLine,
    Circle,
    HoverTool,
    ResetTool,
    SaveTool,
    Range1d,
    LabelSet,
    ColumnDataSource,
)

import datashader as ds
import datashader.transfer_functions as tf

import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc

from .aggregate import ReconcilerBase

# # TODO: Speed this up


def gen_clustree(cluster_df):
    """
    Generates a nx.Digraph based clustree.

    Can be plotted with:

    ```python
    dg = gen_clustree(cluster_df)
    nx.draw_networkx(dg, pos=nx.nx_agraph.graphviz_layout(dg, prog="dot"))
    ```

    Args:
        cluster_df (pd.DataFrame):
            Dataframe where column names are the clustering solution, and indices are the sample names.
            Values of dataframe indicate which cluster the sample is in for each solution.
    """
    g = nx.DiGraph()
    grouped = dict()
    # Build nodes
    node_idx = 0
    for level, (solution_name, solution) in enumerate(cluster_df.items()):
        grouped[level] = list()
        for node_name, node_values in solution.groupby(solution):
            grouped[level].append(node_idx)
            g.add_node(
                node_idx,
                contents=node_values.index.values,
                level=level,
                solution_name=solution_name,
                partition_id=node_name,
                n_items=len(node_values),
            )
            node_idx += 1
        # for node_name, node_values in solution.groupby(solution):
        #     grouped[level].append((solution_name, node_name))
        #     g.add_node((solution_name, node_name),
        #                contents=node_values.index.values, level=level)
    # Add edges
    for level in range(cluster_df.shape[1] - 1):
        current_nodes = grouped[level]
        next_nodes = grouped[level + 1]
        for current_node, next_node in product(current_nodes, next_nodes):
            current_contents = g.node[current_node]["contents"]
            next_contents = g.node[next_node]["contents"]
            intersect = np.intersect1d(
                current_contents, next_contents, assume_unique=True
            )
            intersect_size = len(intersect)
            union_size = len(np.union1d(current_contents, next_contents))
            if intersect_size > 0:
                g.add_edge(
                    current_node,
                    next_node,
                    weight=intersect_size / union_size,
                    contents=intersect,
                    out_frac=intersect_size / len(current_contents),
                    in_frac=intersect_size / len(next_contents),
                    n_cells=intersect_size,
                )
    return g


# ################
# # Plotting stuff
# ################
import bokeh
import bokeh.plotting
from bokeh.io import show, output_notebook


def gen_clustree_plot(
    g: nx.Graph,
    pos: dict = None,
    plot_kwargs: dict = None,
    node_kwargs: dict = None,
    edge_kwargs: dict = None,
):
    """
    Takes a graph, basically just instantiates a plot

    Args:
        g: clustree graph.
        pos: dict containing calculated layout positions
    """
    if pos is None:
        pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
    if plot_kwargs is None:
        plot_kwargs = dict(plot_width=1000, plot_height=600)
    if node_kwargs is None:
        node_kwargs = dict(size=15)
    if edge_kwargs is None:
        edge_kwargs = dict(line_alpha="edge_alpha", line_width=1)

    g_p = g.copy()
    # set_edge_alpha(g_p)

    plot = Plot(**get_ranges(pos), **plot_kwargs)

    graph_renderer = from_networkx(g_p, pos)
    graph_renderer.node_renderer.glyph = Circle(**node_kwargs)
    graph_renderer.edge_renderer.glyph = MultiLine(**edge_kwargs)

    plot.renderers.append(graph_renderer)

    # node_hover = HoverTool(
    #     tooltips=[("solution_name", "@solution_name"),
    #               ("partition_id", "@partition_id"), ("n_items", "@n_items")]
    # )

    # plot.add_tools(node_hover)

    return plot


def get_ranges(pos):
    """
    Return appropriate range of x and y from position dict.

    Usage:
        >>> pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
        >>> plot = Plot(plot_width=1000, plot_height=600, **get_ranges(pos))
    """
    all_pos = np.array(list(zip(*pos.values())))
    max_x, max_y = all_pos.max(axis=1)
    min_x, min_y = all_pos.min(axis=1)
    x_margin = max((max_x - min_x) / 10, 0.5)
    y_margin = max((max_y - min_y) / 10, 0.5)
    return {
        "x_range": Range1d(min_x - x_margin, max_x + x_margin),
        "y_range": Range1d(min_y - y_margin, max_y + y_margin),
    }


# Hacky, probably worth re-implementing
def set_edge_alpha(g):
    """Create edge_alpha attribute for edges based on normalized log scaled weights."""
    edge_weights = pd.Series(
        {(rec[0], rec[1]): float(rec[2]) for rec in g.edges.data("weight")}
    )
    np.log1p(edge_weights, edge_weights.values)  # inplace log
    nx.set_edge_attributes(
        g, (edge_weights / edge_weights.max()).to_dict(), "edge_alpha"
    )


def clustree(
    r: ReconcilerBase,
    min_edge_weight: float = 0.0,
    plot_kwargs: dict = None,
    node_kwargs: dict = None,
    edge_kwargs: dict = None,
) -> Plot:
    n_values = r.settings.apply(lambda x: len(x.unique()))
    if sum(n_values != 1) != 1:
        raise ValueError(
            "Must only exactly one non-unique parameter for generating a clustree."
        )

    param = n_values.index[np.where(n_values != 1)[0]][0]
    order = np.argsort(r.settings[param])
    clusterings = r.clusterings.iloc[:, order]

    g = gen_clustree(clusterings)
    g_sub = g.edge_subgraph(
        [
            k
            for k, v in nx.get_edge_attributes(g, "weight").items()
            if v > min_edge_weight
        ]
    )
    pos = nx.nx_agraph.graphviz_layout(g_sub, prog="dot")

    p = gen_clustree_plot(
        g,
        pos,
        plot_kwargs=plot_kwargs,
        node_kwargs=node_kwargs,
        edge_kwargs=edge_kwargs,
    )

    return p


#########################
# Hierarchy
#########################


def to_img_element(s):
    if isinstance(s, bytes):
        s = s.decode("utf-8")
    return f'<img src="data:image/png;base64,{s}"/>'


def plot_to_bytes(pil_im):
    with BytesIO() as buf:
        pil_im.save(buf, format="png")
        buf.seek(0)
        byteimage = base64.b64encode(buf.read())
    return byteimage


# For converting numpy values to python values
@singledispatch
def json_friendly(x):
    return x


def _identity(x):
    return x


for t in [int, float, str, bool, bytes]:
    json_friendly.register(t)(_identity)

json_friendly.register(np.integer)(int)
json_friendly.register(np.floating)(float)
json_friendly.register(np.str_)(str)
json_friendly.register(np.string_)(str)
json_friendly.register(np.bool_)(bool)


@json_friendly.register(Mapping)
def json_friendly_mapping(d):
    return {json_friendly(k): json_friendly(v) for k, v in d.items()}


@json_friendly.register(Iterable)
def json_friendly_iterable(l):
    return [json_friendly(x) for x in l]


# Suprisingly fast
def calc_freq(comp):
    rec = comp._parent
    samples_idx = rec.clusterings.index
    s = pd.Series(np.zeros(len(samples_idx)), index=samples_idx)
    c = Counter(chain.from_iterable(rec._mapping.iloc[comp.cluster_ids].values))
    s.iloc[list(c.keys())] += list(c.values())
    return s


def ds_umap(
    df: pd.DataFrame,
    x="x",
    y="y",
    *,
    agg=None,
    width: int = 150,
    height: int = 150,
    shade_kwargs: Mapping = {},
):
    shade_kwargs = {"how": "linear", **shade_kwargs}
    cvs = ds.Canvas(width, height)
    pts = cvs.points(df, x, y, agg=agg)
    im = tf.shade(pts, **shade_kwargs)
    return to_img_element(plot_to_bytes(im.to_pil()))


def make_umap_plots(clist, coords: pd.DataFrame, scatter_kwargs={}):
    x, y = coords.columns
    plots = {}
    for cid, c in zip(clist.components.index, clist):
        coords[str(cid)] = calc_freq(c).loc[coords.index]
        plots[cid] = ds_umap(coords, x=x, y=y, agg=ds.max(str(cid)), **scatter_kwargs)
    return plots


def plot_hierarchy(
    complist: "ComponentList", coords: pd.DataFrame, *, scatter_kwargs={}
):
    """
    Params
    ------
    complist
        List of components that will be plotted in this graph.
    """
    coords = coords.copy()
    scatter_kwargs = scatter_kwargs.copy()
    g = complist.to_graph()
    assert len(list(nx.components.weakly_connected_components(g))) == 1
    for k, v in make_umap_plots(
        complist, coords, scatter_kwargs=scatter_kwargs
    ).items():
        g.nodes[k]["img"] = v

    pos = json_friendly(
        nx.nx_agraph.graphviz_layout(nx.DiGraph(g.edges(data=False)), prog="dot")
    )

    graph_renderer = from_networkx(g, pos)
    graph_renderer.node_renderer.glyph = Circle(size=15)
    graph_renderer.edge_renderer.glyph = MultiLine(line_width=1)

    node_hover = HoverTool(
        tooltips=[
            ("img", "@img{safe}"),
            ("component_id", "@index"),
            ("# solutions:", "@n_solutions"),
            ("# samples in intersect", "@n_intersect"),
            ("# samples in union", "@n_union"),
        ],
        attachment="vertical",
    )

    # Adding labels
    label_src = pd.DataFrame.from_dict(
        graph_renderer.layout_provider.graph_layout, orient="index", columns=["x", "y"]
    )
    label_src.index.name = "nodeid"
    label_src = ColumnDataSource(label_src)

    node_label = LabelSet(
        x="x",
        y="y",
        text="nodeid",
        level="annotation",
        source=label_src,
        text_align="center",
    )

    # layout = graph_renderer.layout_provider.graph_layout
    # label, x, y = zip(*((str(label), x, y) for label, (x, y) in layout.items()))
    # node_label = LabelSet(
    #     x=x, y=y, text=label, level="glyph"
    # )

    p = Plot(plot_width=1000, plot_height=500, **get_ranges(pos))
    p.renderers.append(graph_renderer)
    p.add_layout(node_label)
    p.add_tools(node_hover, SaveTool())
    return p
