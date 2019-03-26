# Top level functions
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

import sys
from scanpy.utils import annotate_doc_types
from .aggregate import reconcile, comp_stats
from .cluster import cluster

annotate_doc_types(sys.modules[__name__], 'constclust')
del annotate_doc_types
