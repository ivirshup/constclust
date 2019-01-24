import seaborn as sns
import scanpy as sc
import pandas as pd
import numpy as np
from .aggregate import Component



#TODO: Add title arg, & fix formatting. 
#TODO: Change colorbar to discrete
def component_param_range(component, x, y):
    """Given a component, show which parameters it's found at as a heatmap."""
    # Initialize blank array
    data = pd.pivot_table(
        component._parent.settings[[x, y]], index=y, columns=x, aggfunc=lambda x: 0)
    params = pd.pivot_table(
        component.settings[[x, y]], index=y, columns=x, aggfunc=len, fill_value=0)
    # Maybe use this to calculate color bar?
    # all_params = pd.pivot_table(
        # component._parent.settings, index=y, columns=x, aggfunc=len, fill_value=0)
    data = data + params
    sns.heatmap(data, annot=True, fmt="d", linewidths=.5)

def umap(component, adata):
    cell_value = pd.Series(0, index=adata.obs_names, dtype=float)
    for cluster in component._mapping:
        cell_value[cluster] += 1
    cell_value = cell_value / cell_value.max()
    adata.obs["_tmp"] = cell_value
    sc.pl.umap(adata, color="_tmp")
    adata.obs.drop(columns="_tmp", inplace=True)
