|Build Status| |Docs Status|

``constclust``: Consistent Clusters for scRNA-seq
=================================================

|Overview|

Unsupervised clustering to identify distinct cell types is a crucial early step in analyses of scRNA-seq data. Current methods are dependent on a number of parameters, whose effect on the resulting solution's accuracy and reproducibility are frequently uncharacterized – resulting in ad-hoc selection by users. ``constclust`` is a novel meta-clustering method based on the idea that if the data contains distinct groups which a clustering method can identify, those clusters should be robust to small changes in parameters. By reconciling solutions from state-of-the-art clustering methods over multiple parameters, we can identify locally robust clusters of cells and the regions of parameter space that identified them. As these consistent clusters are found at different levels of resolution, ``constclust`` reveals multilevel structure of cellular identity. Additionally ``constclust`` requires significantly fewer computational resources than current consensus clustering methods for scRNA-seq data.

For usage information, see the docs at: https://constclust.readthedocs.io.

.. |Build Status| image:: https://travis-ci.com/ivirshup/constclust.svg?token=L4NxyJjqtYoWAtJRWfUE&branch=master
    :target: https://travis-ci.com/ivirshup/constclust
.. |Docs Status| image:: https://readthedocs.org/projects/constclust/badge/?version=latest
    :target: https://constclust.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. |Overview| image:: https://github.com/ivirshup/constclust/raw/master/docs/_static/img/repo_fig.png
