"""
Methods for generating and plotting parameters space visualizations 3d plotly.
"""

import networkx as nx
import numpy as np
import pandas as pd
from sklearn import metrics
from itertools import repeat, groupby, product, combinations
# import plotly.graph_objs as go


# def gen_neighbors(settings):
#     """
#     Settings is a dataframe with parameters of the experiment.
#     """
#     settings_idx = {x: sorted(settings[x].unique()) for x in settings}
#     edges = []
#     for s in settings.itertuples(index=False):
#         for k in s._fields:
#             v = getattr(s, k)
#             pos = settings_idx[k].index(v) + 1
#             # Prevents finding elements outside of graph, could be more efficient
#             if pos >= len(settings_idx[k]):
#                 continue
#             # Is there a better way to do this?
#             new_s = s._replace(**{k: settings_idx[k][pos]})
#             edges.append((s, new_s))
#     return edges

def gen_neighbors(settings, orders=None):
    """
    Parameters
    ----------
    settings : `pd.DataFrame`
        Parameters of the experiment.
    orders : `Optional[list]` (default: None)
        List of characters specifying ordering of elements. Elements are
        either "o" for ordered, or "u" for unordered. If nothing is passed,
        all elements are considered ordered.

    Returns
    -------
    List(Tuple)
        Edge list of indices for connected parameter settings.
    """
    if orders is None:
        orders = list(repeat("o", len(settings.columns)))
    else:  # Assert only "o", "u" are in orders
        assert set(orders).issubset({"o", "u"})
    # Consider using natsort
    ordered_settings = {x: sorted(settings[x].unique()) for x, o in zip(
        settings, orders) if o == "o"}
    unordered_settings = {x: list(settings[x].unique()) for x, o in zip(
        settings, orders) if o == "u"}
    reverse_map = dict(zip(settings.itertuples(index=False), settings.index))
    edges = []
    for s in settings.itertuples(index=False):
        for k in ordered_settings.keys():
            v = getattr(s, k)
            pos = ordered_settings[k].index(v) + 1
            # Prevents finding elements outside of graph, could be more efficient
            if pos >= len(ordered_settings[k]):
                continue
            # Is there a better way to do this? Depends on how slow this is
            new_s = s._replace(**{k: ordered_settings[k][pos]})
            if (new_s in reverse_map) and (s in reverse_map):
                edges.append((reverse_map[s], reverse_map[new_s]))
        for k, vs in unordered_settings.items():
            if getattr(s, k) != vs[0]:
                continue
            for u1, u2 in combinations(vs, 2):
                s1 = s._replace(**{k: u1})
                s2 = s._replace(**{k: u2})
                if (s1 in reverse_map) and (s2 in reverse_map):
                    edges.append((reverse_map[s1], reverse_map[s2]))
    return edges


# def build_network(cluster_df, settings_df, params, edge_metric=metrics.adjusted_rand_score):
#     """
#     Create network of clustering solutions.
    
#     Args:
#         cluster_df (pd.DataFrame):
#             Dataframe where each column is a clustering solution for the dataset
#         settings_df (pd.DataFrame):
#             Contains settings for each clustering run. 
#         params (list(str)):
#             The parameters which were varied.
#         edge_metric (function):
#             How to measure difference between solutions.
#     """
#     nodedf = settings_df.set_index(params, verify_integrity=True)
#     edges = gen_neighbors(settings_df[params])
#     edge_cls = edges[0][0].__class__
#     g = nx.Graph()
#     g.add_nodes_from(map(lambda x: (edge_cls._make(x[0]), dict(
#         x[1])), nodedf.iterrows()))  # Converting series to dict
#     for n1, n2 in edges:
#         n1_key = nodedf.loc[n1, "proc_id"]
#         n2_key = nodedf.loc[n2, "proc_id"]
#         value = edge_metric(cluster_df[n1_key], cluster_df[n2_key])
#         g.add_edge(n1, n2, weight=value)
#     return g


# def gen_edge_trace(g, x_param, y_param, z_key):
#     """
#     Args:
#         g (nx.Graph):
#             Graph of clustering solutions.
#         x_param (str):
#             Varying parameter to plot along the x_axis.
#         y_param (str):
#             Varying parameter to plot along the y_axis.
#         z_key: (str):
#             Dependent variable to plot along the z_axis.
#     returns:
#         edge_plt (go.Scatter3d)
#     """
#     x = np.zeros(len(g.edges) * 3, dtype=np.float)
#     y = x.copy()
#     z = x.copy()
#     c = x.copy()
#     for idx, edge in enumerate(g.edges):
#         idx_base = idx * 3
#         x[idx_base:idx_base +
#             3] = [getattr(edge[0], x_param), getattr(edge[1], x_param), None]
#         y[idx_base:idx_base +
#             3] = [getattr(edge[0], y_param), getattr(edge[1], y_param), None]
#         z[idx_base:idx_base+3] = [g.nodes[edge[0]]
#                                   [z_key], g.nodes[edge[1]][z_key], None]
#         # TODO: Should this be hardcoded?
#         c[idx_base:idx_base+3] = list(repeat(g.edges[edge]["weight"], 3))
#     edge_plt = go.Scatter3d(
#         x=x,
#         y=y,
#         z=z,
#         mode="lines",
#         line=go.scatter3d.Line(
#             color=c,  # New, for plotly.py 3.0.0
#             colorscale="Viridis",
#             # showscale=True,  # TODO: Not working, https://github.com/plotly/plotly.py/issues/1085
#         ),
#         hoverinfo="none"
#     )
#     return edge_plt


# def gen_color_bar(line_trace):
#     """
#     Because the line plot won't show a colour bar, this function generates a trace which should show the color bar.
    
#     Generally, this function is waiting on: https://github.com/plotly/plotly.py/issues/1085
#     """
#     return go.Scatter3d(
#         x=line_trace.x,
#         y=line_trace.y,
#         z=line_trace.z,
#         mode="markers",
#         marker=go.scatter3d.Marker(
#             color=line_trace.line.color,
#             # Waiting on https://github.com/plotly/plotly.py/issues/1087
#             colorscale=line_trace.line.to_plotly_json()["colorscale"],
#             showscale=True,
#             opacity=0.00000000000001  # Make invisible
#         ),
#         hoverinfo="none"
#     )


# def gen_hover_trace(g, x_param, y_param, z_key):
#     """
#     Generate a trace which shows hover info for network plot.
#     """
#     x = np.zeros(len(g.nodes), dtype=float)
#     y = x.copy()
#     z = x.copy()
#     text = list(repeat("", len(x)))
#     for idx, n in enumerate(g.nodes):
#         x[idx] = getattr(n, x_param)
#         y[idx] = getattr(n, y_param)
#         z[idx] = g.nodes[n][z_key]
#         text[idx] = f"{x_param}: {x[idx]}<br>{y_param}: {y[idx]}<br>{z_key}: {z[idx]}"
#     hover_plt = go.Scatter3d(
#         x=np.array([getattr(n, x_param) for n in g.nodes]),
#         y=np.array([getattr(n, y_param) for n in g.nodes]),
#         z=np.array([g.nodes[n][z_key] for n in g.nodes]),
#         marker=go.scatter3d.Marker(
#             opacity=0.00000000000001
#         ),
#         mode="markers",
#         hoverinfo="text",
#         hovertext=text
#     )
#     return hover_plt


# def plot_network(g, x_param, y_param, z_key, layout_kwargs={}):
#     """
#     Plots grid network of parameters.
    
#     Args:
#         g (nx.Graph):
#             Graph of clustering solutions.
#         x_param (str):
#             Varying parameter to plot along the x_axis.
#         y_param (str):
#             Varying parameter to plot along the y_axis.
#         z_key: (str):
#             Dependent variable to plot along the z_axis.
#     """
#     edge_plt = gen_edge_trace(g, x_param, y_param, z_key)
#     hover_plt = gen_hover_trace(g, x_param, y_param, z_key)
#     colorbar_plt = gen_color_bar(edge_plt)  # Hopefully temporary
#     layout = go.Layout(
#         scene=dict(
#             xaxis=dict(
#                 title=x_param
#             ),
#             yaxis=dict(
#                 title=y_param
#             ),
#             zaxis=dict(
#                 title=z_key,
#             )
#         ),
#         showlegend=False,
#         **layout_kwargs
#     )
#     return go.Figure(data=[edge_plt, hover_plt, colorbar_plt],
#                      layout=layout)


# def plot_network_widget(g, x_param, y_param, z_key, layout_kwargs={}):
#     """
#     Plots grid network of parameters.
    
#     Args:
#         g (nx.Graph):
#             Graph of clustering solutions.
#         x_param (str):
#             Varying parameter to plot along the x_axis.
#         y_param (str):
#             Varying parameter to plot along the y_axis.
#         z_key: (str):
#             Dependent variable to plot along the z_axis.
#     """
#     edge_plt = gen_edge_trace(g, x_param, y_param, z_key)
#     hover_plt = gen_hover_trace(g, x_param, y_param, z_key)
#     colorbar_plt = gen_color_bar(edge_plt)  # Hopefully temporary
#     layout = go.Layout(
#         scene=dict(
#             xaxis=dict(
#                 title=x_param
#             ),
#             yaxis=dict(
#                 title=y_param
#             ),
#             zaxis=dict(
#                 title=z_key,
#             )
#         ),
#         showlegend=False,
#         **layout_kwargs
#     )
#     return go.FigureWidget(data=[edge_plt, hover_plt, colorbar_plt],
#                            layout=layout)
