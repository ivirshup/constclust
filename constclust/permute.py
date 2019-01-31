"""
This module contains helper functions to permute input data to test the 
robustness methods.

I'm thinking the basic API will be that of iterators. You pass an `AnnData` object 
and specify the way you'd like it to be permuted. This is inspired by 
the `conos` paper. Possible types of pertubation:

* Global subsampling
* Cell mixing 
    * Replace each count (with probablility $p_{mix}$) with a count from 
    a background profile
* Cluster dropping
    * Randomly remove groups
* Sensitivity to very rare cell types
"""

