# from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
# from bokeh.models import Plot, MultiLine, Circle, HoverTool, ResetTool, Range1d
# import networkx as nx
# import numpy as np
# import pandas as pd
# from itertools import product

# # TODO: Speed this up


# def gen_clustree(cluster_df):
#     """
#     Generates a nx.Digraph based clustree.

#     Can be plotted with:
    
#     ```python
#     dg = gen_clustree(cluster_df)
#     nx.draw_networkx(dg, pos=nx.nx_agraph.graphviz_layout(dg, prog="dot"))
#     ```

#     Args:
#         cluster_df (pd.DataFrame):
#             Dataframe where column names are the clustering solution, and indices are the sample names.
#             Values of dataframe indicate which cluster the sample is in for each solution.
#     """
#     g = nx.DiGraph()
#     grouped = dict()
#     # Build nodes
#     node_idx = 0
#     for level, (solution_name, solution) in enumerate(cluster_df.items()):
#         grouped[level] = list()
#         for node_name, node_values in solution.groupby(solution):
#             grouped[level].append(node_idx)
#             g.add_node(node_idx,
#                        contents=node_values.index.values, level=level,
#                        solution_name=solution_name, partition_id=node_name,
#                        n_items=len(node_values))
#             node_idx += 1
#         # for node_name, node_values in solution.groupby(solution):
#         #     grouped[level].append((solution_name, node_name))
#         #     g.add_node((solution_name, node_name),
#         #                contents=node_values.index.values, level=level)
#     # Add edges
#     for level in range(cluster_df.shape[1] - 1):
#         current_nodes = grouped[level]
#         next_nodes = grouped[level + 1]
#         for current_node, next_node in product(current_nodes, next_nodes):
#             current_contents = g.node[current_node]["contents"]
#             next_contents = g.node[next_node]["contents"]
#             intersect = np.intersect1d(
#                 current_contents, next_contents, assume_unique=True)
#             intersect_size = len(intersect)
#             union_size = len(np.union1d(current_contents, next_contents))
#             if intersect_size > 0:
#                 g.add_edge(current_node, next_node,
#                            weight=intersect_size/union_size, contents=intersect,
#                            out_frac=intersect_size/len(current_contents),
#                            in_frac=intersect_size/len(next_contents),
#                            n_cells=intersect_size)
#     return g


# ################
# # Plotting stuff
# ################
# # import bokeh
# # import bokeh.plotting
# # from bokeh.io import show, output_notebook

# def gen_clustree_plot(g: nx.Graph, pos: dict = None,
#                       plot_kwargs: dict = None, node_kwargs: dict = None,
#                       edge_kwargs: dict = None):
#     """
#     Takes a graph, basically just instantiates a plot

#     Args:
#         g: clustree graph.
#         pos: dict containing calculated layout positions
#     """
#     if pos is None:
#         pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
#     if plot_kwargs is None:
#         plot_kwargs = dict(plot_width=1000, plot_height=600)
#     if node_kwargs is None:
#         node_kwargs = dict(size=15)
#     if edge_kwargs is None:
#         edge_kwargs = dict(line_alpha="edge_alpha", line_width=1)

#     g_p = g.copy()
#     # set_edge_alpha(g_p)

#     plot = Plot(**get_ranges(pos), **plot_kwargs)

#     graph_renderer = from_networkx(g_p, pos)
#     graph_renderer.node_renderer.glyph = Circle(**node_kwargs)
#     graph_renderer.edge_renderer.glyph = MultiLine(**edge_kwargs)

#     plot.renderers.append(graph_renderer)

#     node_hover = HoverTool(
#         tooltips=[("solution_name", "@solution_name"),
#                   ("partition_id", "@partition_id"), ("n_items", "@n_items")]
#     )

#     plot.add_tools(node_hover)

#     return plot


# def get_ranges(pos):
#     """
#     Return appropriate range of x and y from position dict.

#     Usage:
#         >>> pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
#         >>> plot = Plot(plot_width=1000, plot_height=600, **get_ranges(pos))
#     """
#     all_pos = np.array(list(zip(*pos.values())))
#     max_x, max_y = all_pos.max(axis=1)
#     min_x, min_y = all_pos.min(axis=1)
#     x_margin = (max_x - min_x) / 10
#     y_margin = (max_y - min_y) / 10
#     return {"x_range": Range1d(min_x-x_margin, max_x+x_margin),
#             "y_range": Range1d(min_y-y_margin, max_y+y_margin)}


# # Hacky, probably worth re-implementing
# def set_edge_alpha(g):
#     """Create edge_alpha attribute for edges based on normalized log scaled weights."""
#     edge_weights = pd.Series({(rec[0], rec[1]): float(
#         rec[2]) for rec in g.edges.data("weight")})
#     np.log1p(edge_weights, edge_weights.values)  # inplace log
#     nx.set_edge_attributes(
#         g, (edge_weights / edge_weights.max()).to_dict(), "edge_alpha")
