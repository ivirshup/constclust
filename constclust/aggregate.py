import anndata
from collections.abc import Collection, Callable, Iterable, Hashable, Mapping
from functools import reduce, partial
from itertools import chain, combinations
from multiprocessing import Pool
from types import MappingProxyType
from typing import List, Optional, Union

import igraph
import networkx as nx
import numba
import numpy as np
import pandas as pd
from pandas.api.types import (
    is_categorical_dtype,
    is_integer_dtype,
    is_bool_dtype,
    is_numeric_dtype,
)
from sklearn import metrics
from scipy import sparse
import matplotlib.pyplot as plt

from .paramspace import gen_neighbors
from .utils import reverse_series_map, pairs_to_dict
from . import plotting

# TODO: Generalize documentation
# TODO: Should _mapping be mapping? Would I want to give names then?


class Component(object):
    """
    A connected component from a `Reconciler`

    Attributes
    ----------
    _parent : Reconciler
        The ``Reconciler`` which generated this component.
    settings : pandas.DataFrame
        Subset of parents settings. Contains only settings for clustering
        which appear in this component.
    cluster_ids : numpy.ndarray
        Which clusters are in this component.
    intersect : numpy.ndarray[int]
        Intersection of samples in this component.
    intersect_names : numpy.ndarray[str]
        Names of samples in the intersection of this component.
    union : numpy.ndarray[int]
        Union of samples in this component.
    union_names : numpy.ndarray[str]
        Names of samples in the union of this component.
    """

    def __init__(self, reconciler, cluster_ids):
        self._parent = reconciler
        self.cluster_ids = np.sort(cluster_ids)
        # TODO: Should a component found via a subset only return the clusters in the subset?
        clusterings = self._parent._mapping.index.get_level_values("clustering")[
            cluster_ids
        ]  # Should already be unique
        self.settings = self._parent.settings.loc[clusterings]
        cells = self._parent._mapping.iloc[cluster_ids].values
        # TODO: I could just figure out how big these are, and make their contents lazily evaluated. I think this would be a pretty big speed up for interactive work.
        self.intersect = reduce(partial(np.intersect1d, assume_unique=True), cells)
        self.union = reduce(np.union1d, cells)
        # self._value_counts = None

    @property
    def intersect_names(self):
        return self._parent._obs_names[self.intersect].values

    @property
    def union_names(self):
        return self._parent._obs_names[self.union].values

    # @property
    # def cell_frequency(self):
    #     return self.value_counts() / len(self.settings)

    # def value_counts(self):
    #     if self._value_counts is None:
    #         cell_ids, counts = pd.value_counts(
    #             self._parent._mapping.iloc[self.cluster_ids].values,
    #             sort=False
    #         )
    #         self._value_counts = pd.Series(data=counts, values=cell_ids)
    #     value_counts = self._value_counts.copy()
    #     value_counts.index = self.union_names  # Is this slow?
    #     return value_counts

    def one_hot(self, selection="intersect"):
        encoding = np.zeros(self._parent.clusterings.shape[0], dtype=bool)
        if selection == "intersect":
            encoding[self.intersect] = True
        elif selection == "union":
            encoding[self.union] = True
        else:
            raise ValueError(
                f"Parameter `selection` must be either 'intersect' or 'union', was '{selection}'"
            )

    def __len__(self):
        return len(self.cluster_ids)

    def __repr__(self):
        return (
            f"<Component n_solutions={len(self)}, "
            f"max_cells={len(self.union)}, "
            f"min_cells={len(self.intersect)}>"
        )


# TODO: Consider making a Mapping
class ComponentList(Collection):
    """A set of consistent components identified from many clustering solutions.

    This is considered to be an immutable list, so operations values will be cached.
    """

    def __init__(self, components):
        # Possible checks
        # 1. Set of parameters should be the same
        # 2. Should be same reconciler? I'm assuming this for now
        self._rec = next(iter(components))._parent
        self._comps = pd.Series(components)
        if len(self._comps.index) != len(self._comps.index.unique()):
            raise KeyError("Components must have unique names")
        self._comps.index.name = "component"

    def __repr__(self):
        return repr(self._comps)

    def __iter__(self):
        # NOTE: Should iterate over keys if this is a mapping
        for comp in self._comps:
            yield comp

    def __contains__(self, obj):
        # NOTE: Should check keys if this is a mapping
        return obj in self._comps.values

    def __len__(self):
        return len(self._comps)

    # TODO: Decide on kind of indexing this supports. Currently just key based, but accepts iterable key lists
    def __getitem__(self, key):
        if isinstance(key, Hashable) and key in self._comps.index:
            return self._comps[key]
        elif isinstance(key, Iterable):
            if not all(subkey in self._comps.index for subkey in key):
                raise KeyError(f"Could not find component '{key}'.")
            else:
                return ComponentList(self._comps[key])
        elif isinstance(key, slice):
            return ComponentList(self._comps[key])
        return self._comps[key]

    @property
    def components(self):
        return self._comps.copy()

    @property
    def obs_names(self) -> pd.Index:
        """The set of observations these components were found on."""
        return self._rec.obs_names

    def to_graph(self, overlap="intersect") -> nx.DiGraph:
        """Builds a hierarchichal graph of the components"""
        comps = self._comps
        assert overlap in {"intersect", "union"}
        # get_overlap = lambda x: getattr(x, overlap)
        assert len(comps.index.unique()) == len(comps)
        g = nx.DiGraph()
        for cidx, c in zip(comps.index, comps):
            g.add_node(
                cidx,
                n_solutions=len(c),
                n_intersect=len(c.intersect),
                n_union=len(c.union),
            )
        sets = pd.Series([set(c.intersect) for c in comps], index=comps.index)
        # sets = pd.Series([set(get_overlap(c)) for c in comps], index=comps.index)
        for i, j in combinations(comps.index, 2):
            ci = set(comps[i].intersect)
            cj = set(comps[j].intersect)
            intersect = ci & cj
            if not intersect:
                continue
            union = ci | cj
            direction = np.array([i, j])[np.argsort([len(ci), len(cj)])][::-1]
            g.add_edge(*direction, weight=len(intersect) / len(union))
        # Remove edges where all contributing cells are shared with predecessor
        for n1 in comps.index:
            adj1 = set(g.successors(n1))
            to_remove = set()
            for n2 in adj1:
                adj2 = set(g.successors(n2))
                shared = adj1 & adj2
                if not shared:
                    continue
                for n3 in shared:
                    shared_cells = sets[n3] & sets[n2]
                    if len(shared_cells & sets[n1]) == len(shared_cells):
                        to_remove.add((n1, n3))
            g.remove_edges_from(to_remove)
        return g

    # Hangs sometimes
    # def to_subgraphs(self):
    #     g = self.to_graph()
    #     return [nx.DiGraph(g.subgraph(c).edges()) for c in list(nx.components.weakly_connected_components(g)) if len(c) > 1]

    # TODO: Add summary of ordered parameters
    def describe(self) -> pd.DataFrame:
        """Calculates summary statistics for components.

        Example
        -------
        >>> stats = comp_list.describe()
        """
        comps = self._comps
        df = pd.DataFrame(index=comps.index)
        df["n_solutions"] = comps.apply(lambda x: len(x.settings.index))
        df["n_intersect"] = comps.apply(lambda x: len(x.intersect))
        df["n_union"] = comps.apply(lambda x: len(x.union))
        return df

    def filter(
        self,
        func: Callable = None,
        *,
        min_intersect: int = None,
        max_intersect: int = None,
        min_union: int = None,
        max_union: int = None,
        min_solutions: int = None,
        max_solutions: int = None,
    ) -> "ComponentList":
        """Filter components from this collection, returns a copy.

        Example
        -------
        >>> to_examine = comp_list.filter(min_intersect=20, min_solutions=100)
        """

        def calc_mask(values, vmin, vmax):
            mask = np.ones_like(values, dtype=bool)
            if vmin is not None:
                mask &= values >= vmin
            if vmax is not None:
                mask &= values <= vmax
            return mask

        comps = self._comps.copy()
        mask = np.ones_like(comps, dtype=bool)

        if func is not None:
            mask &= comps.apply(func)
        mask &= calc_mask(
            comps.apply(lambda x: len(x.intersect)), min_intersect, max_intersect
        )
        mask &= calc_mask(
            comps.apply(lambda x: len(x.union)), min_union, max_union
        )
        mask &= calc_mask(
            comps.apply(lambda x: len(x.settings.index)), min_solutions, max_solutions
        )
        return ComponentList(comps[mask])

    def one_hot(self, as_sparse=False):
        shape = (len(self), len(self[0]._parent.clusterings))
        indptr = np.zeros(len(self) + 1, dtype=int)
        curr = 0
        for i, comp in enumerate(self._comps):
            curr += len(comp.intersect)
            indptr[i + 1] = curr
        indices = np.zeros(curr, dtype=int)
        data = np.ones(curr, dtype=bool)
        for i, comp in enumerate(self._comps):
            indices[indptr[i] : indptr[i + 1]] = comp.intersect
        mtx = sparse.csr_matrix((data, indices, indptr), shape=shape)
        if not as_sparse:
            mtx = pd.DataFrame(
                mtx.toarray(),
                index=self._comps.index,
                columns=self[0]._parent.clusterings.index,
            )
        return mtx

    # Plotting methods, probably move to own attribute
    def plot_components(
        self,
        adata: "anndata.AnnData",
        *,
        x_param: str = "n_neighbors",
        y_param: str = "resolution",
        embedding_basis: str = "X_umap",
        embedding_kwargs: Mapping = MappingProxyType({}),
    ):
        """Plot parameter space and scatter plot for each component.

        The parameter space is a heatmap, showing the range of parameters each component was found in.
        The scatter plot shows which observations were included in the component in a 2d embedding of the dataset.

        Params
        ------
        x_param
            Which key from the parameters will be along the y-axis of the heatmaps.
        y_param
            Which key from the parameters will be along the y-axis of the heatmaps.
        embedding_basis
            Basis from adata to use for embedding plot.
        embedding_kwargs
            Keyword arguments to pass to sc.pl.embedding.

        Example
        -------

        >>> comps.plot_components(coords=adata.obsm["X_umap"])
        """
        comps = self._comps

        embedding_kwargs = embedding_kwargs.copy()
        embedding_kwargs["show"] = False

        def title_string(comp_id, entry):
            return f"Component {comp_id}: " + ", ".join(
                f"{k}: {v}" for k, v in entry.items()
            )

        stats = self.describe()
        for k in comps.index:
            title = title_string(k, stats.loc[k])
            fig = plotting.component(
                comps[k],
                adata,
                embedding_basis=embedding_basis,
                x=x_param,
                y=y_param,
                embedding_kwargs=embedding_kwargs,
            )
            fig.suptitle(title)
            plt.show()

    # def plot_graph(self):
    #     """Plots hierarchies present in components in this object."""
    #     g = self.to_graph()
    #     subgs = [nx.DiGraph(g.subgraph(c).edges()) for c in list(nx.components.weakly_connected_components(g)) if len(c) > 1]
    #     for subg in subgs:
    #         nx.draw(subg, pos=nx.nx_agraph.graphviz_layout(subg, prog="dot"), with_labels=True)
    #         plt.show()

    def plot_hierarchies(
        self,
        coords: Union[np.ndarray, pd.DataFrame],
        *,
        overlap="intersect",
        scatter_kwargs: Mapping = MappingProxyType({}),
    ):
        """Find and plot interactive hierarchies of components.

        Params
        ------
        coords
            Coordinates to use in scatter plots. Should have shape (n_obs, 2). If it's a
            dataframe, it's index should contain the same elements as `self.obs_names`.
        scatter_kwargs
            Key word arguments passed to ds_umap

        Example
        -------

        >>> from bokeh.io import show
        >>> comps = reconciler.get_components(0.9, min_cells=5)
        >>> show(
                comps
                .filter(min_solutions=100)
                .plot_hierarchies(coords=adata.obsm["X_umap"])
            )
        """
        from .clustree import plot_hierarchy
        from bokeh.layouts import column

        if coords.shape != (len(self.obs_names), 2):
            raise ValueError(
                f"`coords` must have shape: {(len(self.obs_names), 2)}, had shape: {coords.shape}."
            )

        if isinstance(coords, pd.DataFrame):
            coords.reindex(index=self.obs_names)
            assert coords.index.equals(
                self.obs_names
            ), "coords index did not match self.obs_names"
        else:
            coords = pd.DataFrame(coords, columns=["x", "y"], index=self.obs_names)

        plots = []
        for hierarchy in nx.components.weakly_connected.weakly_connected_components(
            self.to_graph(overlap=overlap)
        ):
            if len(hierarchy) < 2:
                print(f"Component {list(hierarchy)[0]} was not found in a hierarchy.")
                continue
            clist_sub = ComponentList(self._comps[self._comps.index.isin(hierarchy)])
            plots.append(
                plot_hierarchy(clist_sub, coords, scatter_kwargs=scatter_kwargs)
            )
        return column(*plots)


class ReconcilerBase(object):
    """
    Base type for reconciler.

    Has methods for subsetting implemented, providing data is up to subclass.
    """

    def __repr__(self):
        kls = self.__class__.__name__
        n_cells = len(self.clusterings)
        n_clusterings, n_params = self.settings.shape
        n_clusters = len(self._mapping)
        return f"<{kls} {n_clusterings} clusterings, {n_clusters} clusters, {n_cells} cells>"

    @property
    def cluster_ids(self):
        return self._mapping.index.get_level_values("cluster")

    @property
    def obs_names(self) -> pd.Index:
        """The set of observations clusters were found on."""
        return pd.Index(self._obs_names.values)

    # TODO should I remove this?
    def get_param_range(self, clusters):
        """
        Given a set of clusters, returns the range of parameters for which they were calculated.

        Parameters
        ----------
        clusters : Collection[Int]
            If its a collection of ints, I'll say that was a range of parameter ids.
        """
        idx = []
        for c in clusters:
            if isinstance(c, int):
                idx.append(c)
            else:
                idx.append(c[0])
        return self.settings.loc[idx]

    def subset_clusterings(self, clusterings_to_keep):
        """
        Take subset of Reconciler, where only ``clusterings_to_keep`` are present.

        Reduces size of both ``.settings`` and ``.clusterings``.

        Parameters
        ----------
        clusterings_to_keep
            Indexer into ``Reconciler.settings``. Anything that should give the correct
            result for ``reconciler.settings.loc[clusterings_to_keep]``.

        Returns
        -------
        ``ReconcilerSubset``
        """
        clusterings_to_keep = self.settings.loc[clusterings_to_keep].index.values
        new_settings = self.settings.loc[clusterings_to_keep]  # Should these be copies?
        new_clusters = self.clusterings.loc[:, clusterings_to_keep]  # This is a copy?
        new_mapping = self._mapping[
            np.isin(
                self._mapping.index.get_level_values("clustering"), clusterings_to_keep
            )
        ]
        return ReconcilerSubset(
            self._parent, new_settings, new_clusters, new_mapping, self.graph
        )

    def subset_cells(self, cells_to_keep):
        """
        Take subset of Reconciler, where only ``cells_to_keep`` are present.

        Parameters
        ----------
        cells_to_keep :
            Indexer into ``Reconciler.clusterings``. Anything that should give the correct
            result for ``reconciler.clusterings.loc[cells_to_keep]``.

        Returns
        -------
        ``ReconcilerSubset``
        """
        intmap = reverse_series_map(self._obs_names)
        cells_to_keep = intmap[self.clusterings.loc[cells_to_keep].index.values]
        new_clusterings = self.clusterings.iloc[cells_to_keep, :]
        new_mapping = gen_mapping(new_clusterings).apply(
            lambda x: cells_to_keep.values[x]
        )
        # TODO: test how long this takes
        new_rec = ReconcilerSubset(
            self._parent,
            self.settings,
            new_clusterings,
            new_mapping,
            self.graph,
        )
        return new_rec

    # TODO: Should these describe methods only work on full reconciler?
    # Is it intuitive enough that the results will change when subset by cells?
    def describe_clusters(self, log1p: bool = False) -> pd.DataFrame:
        """
        Describe the clusters in this Reconciler.

        Params
        ------
        log1p
            Whether to also return log transformed values for numeric cols.

        Returns
        -------

        DataFrame containing summary statistics on the clusters in this reconciler. Good
        for plotting.

        Example
        -------
        >>> import hvplot.pandas
        >>> clusters = reconciler.describe_clusters(log1p=True)
        >>> clusters.hvplot.scatter(
            "log1p_resolution",
            "log1p_n_obs",
            datashade=True,
            dynspread=True
        )
        """
        lens = pd.DataFrame(self._mapping.apply(len), columns=["n_obs"])
        lens = pd.merge(lens, self.settings, left_on="clustering", right_index=True)
        if log1p:
            for k in lens.columns[lens.dtypes.apply(is_numeric_dtype)]:
                lens[f"log1p_{k}"] = np.log1p(lens[k])
        return lens

    def describe_clusterings(self) -> pd.DataFrame:
        """
        Convenience function to generate summary statistics for clusterings in a reconciler.

        Example
        -------
        >>> import seaborn as sns
        >>> clusterings = reconciler.describe_clusterings()
        >>> sns.jointplot(data=clusterings, x="resolution", y="max_n_obs")
        """
        # r = r._parent
        m = self._mapping
        cdf = pd.DataFrame(index=pd.Index(self.settings.index, name="clustering"))
        cdf["n_clusters"] = (
            m.index.get_level_values("clustering").value_counts(sort=False).sort_index()
        )
        ls = m.apply(len)
        gb = ls.groupby(level="clustering")
        cdf["min_n_obs"] = gb.min()
        cdf["max_n_obs"] = gb.max()
        cdf["mean_n_obs"] = gb.mean()
        cdf["n_singletons"] = (ls == 1).groupby("clustering").sum()
        cdf = cdf.join(self.settings)
        return cdf


# TODO: Figure out what I want this to be. Do I want this to refer to the full set
# of observations at all, or should that information go away?
class ReconcilerSubset(ReconcilerBase):
    """
    Subset of a Reconciler

    Attributes
    ----------
    _parent: Reconciler
        `Reconciler` this subset was derived from.
    settings: pandas.DataFrame
        Settings for clusterings in this subset.
    clusterings: pandas.DataFrame
        Clusterings contained in this subset.
    graph: igraph.Graph
        Reference to graph from parent.
    cluster_ids: np.ndarray[int]
        Integer ids of all clusters in this subset.
    _mapping: pandas.Series
        ``pd.Series`` with a ``MultiIndex``. Unlike the ``_mapping`` from ``Reconciler``,
        this does not necessarily have all clusters, so ranges of clusters cannot be
        assumed to be contiguous. Additionally, you can't just index into this with
        ``cluster_ids`` as positions.
    _obs_names: ``pd.Series``
        Maps from integer position to input cell name.
    """

    def __init__(self, parent, settings, clusterings, mapping, graph):
        # assert isinstance(parent, Reconciler)  #  Can fail when using autoreloader
        assert all(settings.index == clusterings.columns)
        assert all(
            mapping.index.get_level_values("clustering").unique().isin(settings.index)
        )
        self._parent = parent
        self.settings = settings
        self.clusterings = clusterings
        self._obs_names = parent._obs_names[
            parent._obs_names.isin(clusterings.index)
        ]  # Ordering
        # self._obs_names = self.clusterings.index
        self._mapping = mapping
        self.graph = graph

    def find_contained_components(
        self, min_presence: float, min_weight: float = 0.9, min_cells: int = 2
    ):
        """
        Find components contained in a subset.
        """
        m1 = self._mapping
        m2 = self._parent._mapping
        presence = m1.apply(len) / m2.loc[m1.index].apply(len)
        clusters = np.array(
            presence.index.get_level_values("cluster")[presence > min_presence]
        )
        return self.find_components(min_weight, clusters, min_cells)

    def find_components(self, min_weight, clusters, min_cells=2) -> List[Component]:
        # This is actually a very slow check. Maybe I should wait on it?
        # TODO: I've modified the code, check speed and consider if this is really what I want.
        # if not np.isin(clusters, self._mapping.get_level_values("cluster")).all():
        #     raise ValueError("")
        return self._parent.find_components(min_weight, clusters, min_cells=min_cells)

    def get_components(self, min_weight: float, min_cells: int = 2) -> List[Component]:
        """
        Return connected components of graph, with edges filtered by min_weight.

        Parameters
        ----------
        min_weight
            Minimum edge weight for inclusion of a clustering.
        min_cells
            Minimum cells a component should have.
        """
        clusters = self._mapping.index.get_level_values("cluster")
        return self._parent.find_components(min_weight, clusters, min_cells)


class Reconciler(ReconcilerBase):
    """
    Collects and reconciles many clusterings by local (in parameter space)
    stability.

    Attributes
    ----------
    settings : pandas.DataFrame
        Contains settings for all clusterings. Index corresponds to
        `.clusterings` columns, while columns should correspond to the
        parameters which were varied.
    clusterings : pandas.DataFrame
        Contains cluster assignments for each cell, for each clustering.
        Columns correspond to `.settings` index, while the index correspond
        to the cells. Each cluster is encoded with a unique cluster id.
    graph : igraph.Graph
        Weighted graph. Nodes are clusters (identified by unique cluster id
        integer, same as in `.clusterings`). Edges connect clusters with shared
        contents. Weight is the Jaccard similarity between the contents of the
        clusters.
    cluster_ids : numpy.ndarray[int]
        Integer ids of all clusters in this Reconciler.
    _obs_names : pandas.Series
        Ordered set for names of the cells. Internally they are refered to by
        integer positions.
    _mapping : pandas.Series
        ``pd.Series`` with a ``MultiIndex``. Index has levels ``clustering``
        and ``cluster``. Each position in index should have a unique value at
        level "cluster", which corresponds to a cluster in the clustering
        dataframe. Values are ``np.arrays`` with indices of cells in relevant
        cluster. This should be considered immutable, though this is not the
        case for ``ReconcilerSubset``s.
    """

    def __init__(
        self,
        settings: pd.DataFrame,
        clusterings: pd.DataFrame,
        mapping: pd.Series,
        graph: igraph.Graph,
    ):
        assert all(settings.index == clusterings.columns)
        self._parent = self  # Kinda hacky, could maybe remove
        self.settings = settings
        self.clusterings = clusterings
        self._obs_names = pd.Series(
            clusterings.index.values, index=np.arange(len(clusterings))
        )
        self._mapping = mapping
        self.graph = graph

    # TODO: Allow passing function for clusters
    def find_components(
        self, min_weight: float, clusters: "Collection[int]", min_cells: int = 2
    ) -> List[Component]:
        """
        Return components from filtered graph which contain specified clusters.

        Parameters
        ----------
        min_weight : ``float``
            Minimum weight for edges to be kept in graph. Should be over 0.5.
        clusters : ``np.array[int]``
            Clusters which you'd like to search from.
        """
        # Subset graph
        over_cutoff = np.where(np.array(self.graph.es["weight"]) >= min_weight)[0]
        sub_g = self.graph.subgraph_edges(over_cutoff, delete_vertices=True)
        # Mapping from cluster_id to node_id
        cidtonode = dict(map(tuple, map(reversed, enumerate(sub_g.vs["cluster_id"]))))
        # nodemap = np.frompyfunc(cidtonode.get, 1, 1)

        # Get mapping from clustering to clusters
        # Because this is an Reconciler object, we can just index the mapping by position
        clusteringtocluster = {
            k: np.array(v)
            for k, v in pairs_to_dict(
                iter(self._mapping.loc[(slice(None), clusters)].index)
            ).items()
        }
        # Only look for clusters in graph
        clusters = np.intersect1d(clusters, sub_g.vs["cluster_id"], assume_unique=True)

        # Create sieve
        to_visit = np.zeros(self.graph.vcount(), dtype=bool)
        to_visit[clusters] = True

        # Figure out the clusterings I want to visit
        # I know that no two clusters from one clustering can be in the same component
        # (should probably put limit on jaccard score to ensure this).
        # This means I can explore each component at the same time, or at least know
        # I won't be exploring the same one twice
        frame = self._mapping.index.to_frame().reset_index(drop=True)
        clusterings = frame.loc[lambda x: x["cluster"].isin(clusters)]["clustering"]
        clustering_queue = clusterings.tolist()

        components = list()
        while len(clustering_queue) > 0:
            clustering = clustering_queue.pop()
            foundclusters = clusteringtocluster[clustering]
            foundclusters = foundclusters[
                np.where(to_visit[foundclusters])[0]
            ]  # Filtering
            if not any(to_visit[foundclusters]):
                continue
            for cluster in foundclusters:
                component = np.sort(
                    [x["cluster_id"] for x in sub_g.bfsiter(cidtonode[cluster])]
                )
                components.append(component)
                to_visit[component] = False
        # Format and return components
        out = [
            Component(self, c) for c in components
        ]  # This is slow, probably due to finding the union and intersect
        return ComponentList(
            sorted(
                filter(lambda x: len(x.intersect) >= min_cells, out),
                key=lambda x: (len(x.settings), len(x.intersect)),
                reverse=True,
            )
        )

    def get_components(self, min_weight: float, min_cells: int = 2):
        """
        Return connected components of graph, with edges filtered by ``min_weight``.

        Parameters
        ----------
        min_weight
            Minimum weight for edges to be kept in graph. Should be in range ``[0.5, 1]``.
        min_cells
            Minimum number of cells in a component.

        Returns
        -------
        ``List[Component]``:
           All components from ``Reconciler``, sorted by number of clusterings.
        """
        over_cutoff = np.where(np.array(self.graph.es["weight"]) >= min_weight)[0]
        # If vertices are deleted, indices (ids) change
        sub_g = self.graph.subgraph_edges(over_cutoff, delete_vertices=True)
        comps = []
        # Filter out 1 node components
        cids = np.array(sub_g.vs["cluster_id"])
        for graph_comp in filter(lambda x: len(x) > 1, sub_g.components()):
            comp = Component(self, cids[list(graph_comp)])
            if len(comp.intersect) < min_cells:
                continue
            comps.append(comp)
        comps.sort(key=lambda x: (len(x.settings), len(x.intersect)), reverse=True)
        return ComponentList(comps)


def reconcile(
    settings: pd.DataFrame, clusterings: pd.DataFrame, paramtypes="oou", nprocs: int = 1
) -> Reconciler:
    """
    Constructor for reconciler object.

    Parameters
    ----------
    settings
        Parameterizations of each clustering.
    clusterings
        Assignments from each clustering.
    nprocs
        Number of processes to use

    Example
    -------
    >>> params, clusterings = cluster(adata, ... )
    >>> reconciler = reconcile(params, clusterings)
    """
    assert all(
        settings.index == clusterings.columns
    )  # I should probably save these, right?
    # Check clusterings:
    clust_dtypes = clusterings.dtypes
    if not all(map(is_integer_dtype, clust_dtypes)):
        wrong_types = {t for t in clust_dtypes if not is_integer_dtype(t)}
        raise TypeError(
            "Contents of `clusterings` must be integers dtypes. Found:"
            " {}".format(wrong_types)
        )
    # Set cluster names to be unique
    # Rank data to be sure values are consecutive integers per cluster
    clusterings = clusterings.rank(axis=0, method="dense").astype(int) - 1
    cvals = clusterings.values
    cvals[:, 1:] += (cvals[:, :-1].max(axis=0) + 1).cumsum()

    mapping = gen_mapping(clusterings)

    assert all(np.unique(cvals) == mapping.index.levels[1])

    edges = build_graph(
        settings, clusterings, mapping=mapping, paramtypes=paramtypes, nprocs=nprocs
    )
    graph = igraph.Graph(
        n=len(mapping),
        edges=list(((i, j) for i, j, k in edges)),
        vertex_attrs={"cluster_id": np.arange(len(mapping))},
        edge_attrs={"weight": list(k for i, j, k in edges)},
    )
    return Reconciler(settings, clusterings, mapping, graph)


def _prep_neighbors(neighbors, clusterings):
    for i, j in neighbors:
        yield clusterings.values[:, i], clusterings.values[:, j]


def _call_get_edges(args):
    """Helper function for Pool.map"""
    return _get_edges(*args)


@numba.njit(cache=True)
def _get_edges(clustering1: np.array, clustering2: np.array):
    edges = []
    offset1 = clustering1.min()
    offset2 = clustering2.min()
    # Because of how I've done unique node names, potentially this
    # could be done in a more generic way by creating a mapping here.
    offset_clusts1 = clustering1 - offset1
    offset_clusts2 = clustering2 - offset2
    # Allocate coincidence matrix
    nclusts1 = offset_clusts1.max() + 1
    nclusts2 = offset_clusts2.max() + 1
    coincidence = np.zeros((nclusts1, nclusts2))
    # Allocate cluster size arrays
    ncells1 = np.zeros(nclusts1)
    ncells2 = np.zeros(nclusts2)
    # Compute lengths of the intersects
    for cell in range(len(clustering1)):
        c1 = offset_clusts1[cell]
        c2 = offset_clusts2[cell]
        coincidence[c1, c2] += 1
        ncells1[c1] += 1
        ncells2[c2] += 1
    for cidx1, cidx2 in np.ndindex(coincidence.shape):
        isize = coincidence[cidx1, cidx2]
        if isize < 1:
            continue
        jaccard_sim = isize / (ncells1[cidx1] + ncells2[cidx2] - isize)
        edge = (cidx1 + offset1, cidx2 + offset2, jaccard_sim)
        edges.append(edge)
    return edges


def build_graph(settings, clusters, mapping=None, paramtypes="oou", nprocs=1):
    """
    Build a graph of overlapping clusters (for neighbors in parameter space).
    """
    graph = list()  # edge list
    neighbors = gen_neighbors(settings, paramtypes)
    # if mapping is None:
    # mapping = gen_mapping(clusters)
    args = _prep_neighbors(neighbors, clusters)
    if nprocs > 1:
        # TODO: Consider replacing with joblib
        with Pool(nprocs) as p:
            edges = p.map(_call_get_edges, args)
        graph = chain.from_iterable(edges)
    else:
        graph = chain.from_iterable(map(_call_get_edges, args))
    return list(graph)


def build_global_graph(settings, clusters):
    graph = list()  # edge list
    neighbors = gen_neighbors(settings, "oou")
    for clustering1, clustering2 in neighbors:
        clusters1 = clusters[clustering1]
        clusters2 = clusters[clustering2]
        score = metrics.adjusted_rand_score(clusters1, clusters2)
        edge = (clustering1, clustering2, score)
        graph.append(edge)
    return graph


def gen_mapping(clusterings):
    """
    Create a mapping from clustering and cluster to cluster contents.

    Parameters
    ----------
    clusterings : pandas.DataFrame

    Returns
    -------
    pandas.Series
        Mapping to cluster contents. Index is a MultiIndex with levels
        "clustering", "cluster". Values are np.arrays of cluster contents.
    """
    keys, values = _gen_mapping(clusterings.values)
    mapping = pd.Series(
        values, index=pd.MultiIndex.from_tuples(keys, names=["clustering", "cluster"])
    )
    return mapping


@numba.njit(cache=True)
def _gen_mapping(clusterings):
    n_cells, n_clusts = clusterings.shape
    keys = []
    values = []
    for clustering_id in np.arange(n_clusts):
        clustering = clusterings[:, clustering_id]
        unique_vals = np.unique(clustering)
        sorted_idxs = np.argsort(clustering, kind="mergesort")
        sorted_vals = clustering[sorted_idxs]
        indices = list(np.where(np.diff(sorted_vals))[0] + 1)
        # With numba (numba doesn't have np.split)
        indices.append(n_cells)
        split_vals = list()
        cur = 0
        for i in range(len(indices)):
            nxt = indices[i]
            split_vals.append(sorted_idxs[cur:nxt])
            cur = nxt
        # Without numba
        # split_vals = np.split(sorted_idxs, indices)
        values.extend(split_vals)
        keys.extend([(clustering_id, u) for u in unique_vals])
    return keys, values


def comp_stats(comps):
    stats = pd.DataFrame(
        {
            "n_clusts": [len(c) for c in comps],
            "intersect": [len(c.intersect) for c in comps],
            "union": [len(c.union) for c in comps],
        }
    )
    # TODO: Add some summary of param ranges
    # if param_stats:

    for col in stats:
        stats[f"log1p_{col}"] = np.log1p(stats[col])
    return stats
