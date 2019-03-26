# Top level functions
from .aggregate import reconcile, comp_stats
from .cluster import cluster

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
