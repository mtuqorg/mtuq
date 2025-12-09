# mtuq

[![Build Status](https://github.com/mtuqorg/mtuq/actions/workflows/python-app.yaml/badge.svg)](https://github.com/rmodrak/mtuq/blob/master/.github/workflows/python-app.yaml)
[![SCOPED](https://img.shields.io/endpoint?url=https://runkit.io/wangyinz/scoped/branches/master/MTUQ)](https://github.com/SeisSCOPED/container/pkgs/container/MTUQ)

MTUQ provides *m*oment *t*ensor estimates and *u*ncertainty *q*uantification from broadband seismic data.  


## Getting started

[Installation](https://mtuqorg.github.io/mtuq/install/index.html)

[Quick start](https://mtuqorg.github.io/mtuq/quick_start.html)



## Documentation

[Acquiring seismic data](https://mtuqorg.github.io/mtuq/user_guide/02.html)

[Acquiring Green's functions](https://mtuqorg.github.io/mtuq/user_guide/03.html)

[Data processing](https://mtuqorg.github.io/mtuq/user_guide/04.html)

[Visualization galleries](https://mtuqorg.github.io/mtuq/user_guide/05.html)

[Library reference](https://mtuqorg.github.io/mtuq/library/index.html)


## Highlights

Common use cases include [double couple moment tensor](https://github.com/mtuqorg/mtuq/blob/master/examples/SerialGridSearch.DoubleCouple.py), [full moment tensor](https://github.com/mtuqorg/mtuq/blob/master/examples/GridSearch.FullMomentTensor.py), [depth](https://github.com/rmodrak/mtuq/blob/master/examples/GridSearch.DoubleCouple%2BMagnitude%2BDepth.py) and [hypocenter](https://github.com/rmodrak/mtuq/blob/master/examples/GridSearch.DoubleCouple%2BMagnitude%2BHypocenter.py) uncertainty analysis.  Applications involving composite sources, force sources, constrained moment tensor sources, source-time functions, and other source parameters are also possible.


### Solver interfaces

[I/O functions](https://mtuqorg.github.io/mtuq/library/index.html#data-i-o)
are included for reading AxiSEM, SPECFEM3D, and FK Green's functions as well as
downloading Green's functions from remote [syngine](http://ds.iris.edu/ds/products/syngine/) databases.



### Misfit evaluation

Waveform difference and cross-correlation time-shift [misfit evaluation](https://mtuqorg.github.io/mtuq/library/index.html#data-processing-and-inversion)
on body-wave and surface-wave windows is implemented in numba-accelerated Python.

These misfit functions can be used with [mtuq.grid_search](https://mtuqorg.github.io/mtuq/library/generated/mtuq.grid_search.grid_search.html), which automatically partitions the grid over multiple MPI processes if invoked from an MPI environment.  For efficient and unbiased uncertainty quantification, [uniform grids](https://mtuqorg.github.io/mtuq/library/index.html#moment-tensor-and-force-grids) can be used for the grid search, drawing from [Tape2015](https://academic.oup.com/gji/article/202/3/2074/613765).

Alternatively, MTUQ misfit functions can be used as a starting point for Bayesian uncertainty quantification using pymc or other MCMC libraries.


### Visualization

[Visualization utilities](https://mtuqorg.github.io/mtuq/user_guide/05/gallery_mt.html) are included for both the [eigenvalue lune](https://onlinelibrary.wiley.com/doi/10.1111/j.1365-246X.2012.05491.x) and [v,w rectangle](https://academic.oup.com/gji/article/202/3/2074/613765), with matplotlib and Generic Mapping Tools graphics backends.


### Testing

The package has been tested against [legacy Perl/C codes](https://github.com/mtuqorg/mtuq/blob/master/tests/benchmark_cap_vs_mtuq.py) as well as [published studies](https://github.com/rmodrak/mtbench).

## Citation

If you use MTUQ in your research, please cite:

J Thurin, R Modrak, C Tape, A M McPherson, F R Rodríguez-Cardozo, J Kintner, L Ding, Q Liu, J Braunmiller, MTUQ: a framework for estimating moment tensors, point forces, and their uncertainties, *Geophysical Journal International*, Volume 241, Issue 2, May 2025, Pages 1373–1390, <https://doi.org/10.1093/gji/ggaf080>

Scripts for reproducing the grid-search results and figures from Cases 1 to 3 of this publication, along with useful files (best-fitting moment tensor parameters, weight files, first-motion polarity picks, etc.), are available in the Zenodo collection at <https://doi.org/10.5281/zenodo.13868450>.

[Instaseis]: http://instaseis.net/

[obspy]: https://github.com/obspy/obspy/wiki

[ZhaoHelmberger1994]: https://pubs.geoscienceworld.org/ssa/bssa/article-abstract/84/1/91/102552/Source-estimation-from-broadband-regional?redirectedFrom=fulltext

[ZhuHelmberger1996]: https://pubs.geoscienceworld.org/ssa/bssa/article-abstract/86/5/1634/120218/Advancement-in-source-estimation-techniques-using?redirectedFrom=fulltext

