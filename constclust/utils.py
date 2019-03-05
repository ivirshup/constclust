import pandas as pd
import numpy as np


def reverse_series_map(series):
    """Reverse a mapping"""
    return pd.Series(series.index.values, index=series.values)


def pairs_to_dict(pairs):
    cur = next(pairs)
    curfirst = cur[0]
    l = [cur[1]]
    d = {curfirst: l}
    for cur in pairs:
        if curfirst != cur[0]:
            curfirst = cur[0]
            l = [cur[1]]
            d[curfirst] = l
        else:
            l.append(cur[1])
    return d