
#
# graphics/uq/force.py - uncertainty quantification of forces on the unit sphere
#

import numpy as np

from matplotlib import pyplot

from mtuq.graphics.uq._gmt import _plot_force_gmt
from mtuq.graphics.uq._matplotlib import _plot_force_matplotlib
from mtuq.grid_search import DataFrame, DataArray, MTUQDataArray, MTUQDataFrame
from mtuq.util import defaults, warn
from mtuq.util import dataarray_idxmin, dataarray_idxmax
from mtuq.util.math import closed_interval, open_interval


def plot_misfit_force(filename, ds, **kwargs):
    """ Plots misfit values with respect to force orientation

    .. rubric :: Required input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing forces and corresponding misfit values


    .. rubric :: Optional input arguments

    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_force.html>`_


    """
    defaults(kwargs, {
        'colormap': 'viridis',
        'colorbar_label': 'Misfit',
        })

    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        misfit = _misfit_regular(ds)

    elif issubclass(type(ds), DataFrame):
        misfit = _misfit_random(ds)

    _plot_force(filename, misfit, **kwargs)


def plot_likelihood_force(filename, ds, var, **kwargs):
    """ Plots maximum likelihood values with respect to force orientation 

    .. rubric :: Required input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing forces and corresponding misfit values

    ``var`` (`float` or `array`):
    Data variance


    .. rubric :: Optional input arguments

    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_force.html>`_


    """

    defaults(kwargs, {
        'colormap': 'hot_r',
        'colorbar_label': 'Likelihood',
        })

    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        likelihoods = _likelihoods_regular(ds, var)

    elif issubclass(type(ds), DataFrame):
        likelihoods = _likelihoods_random(ds, var)

    _plot_force(filename, likelihoods, **kwargs)


def plot_marginal_force(filename, ds, var, **kwargs):
    """ Plots marginal likelihood values with respect to force orientation 

    .. rubric :: Required input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing forces and corresponding misfit values

    ``var`` (`float` or `array`):
    Data variance


    .. rubric :: Optional input arguments

    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_force.html>`_

    """
    defaults(kwargs, {
        'colormap': 'hot_r',
        'colorbar_label': 'Likelihood',
        })

    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        marginals = _marginals_regular(ds, var)

    elif issubclass(type(ds), DataFrame):
        marginals = _marginals_random(ds, var)

    _plot_force(filename, marginals, **kwargs)


def plot_magnitude_tradeoffs_force(filename, ds, **kwargs):
    """ Plots magnitude versus force orientation tradeoffs

    .. rubric :: Required input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing forces and corresponding misfit values


    .. rubric :: Optional input arguments

    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_force.html>`_


    """
    defaults(kwargs, {
        'colormap': 'gray',
        })

    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        marginals = _magnitudes_regular(ds)

    elif issubclass(type(ds), DataFrame):
        raise NotImplementedError

    _plot_force(filename, marginals, **kwargs)


def _plot_force(filename, da, show_best=True, show_tradeoffs=False, 
    backend=_plot_force_matplotlib, **kwargs):

    """ Plots values with respect to force orientation

    .. rubric :: Keyword arguments

    ``colormap`` (`str`):
    Color palette used for plotting values 
    (choose from GMT or MTUQ built-ins)

    ``show_best`` (`bool`):
    Show orientation of best-fitting force

    ``title`` (`str`):
    Optional figure title

    ``backend`` (`function`):
    Choose from `_plot_force_gmt` (default) or user-supplied function

    """
    if not issubclass(type(da), DataArray):
        raise Exception()


    best_force = None
    if show_best:
        if 'best_force' in da.attrs:
            best_force = da.attrs['best_force']
        else:
            warn("Best-fitting force not given")

    backend(filename,
        da.coords['phi'],
        da.coords['h'], 
        da.values.transpose(),
        best_force=best_force,
        **kwargs)


#
# for extracting misfit or likelihood from regularly-spaced grids
#

def _misfit_regular(da):
    """ For each force orientation, extracts minimum misfit
    """
    misfit = da.min(dim=('origin_idx', 'F0'))

    return misfit.assign_attrs({
        'best_force': _min_force(da)
        })


def _likelihoods_regular(da, var):
    """ For each force orientation, calculates maximum likelihood value
    """
    likelihoods = da.copy()
    likelihoods.values = np.exp(-likelihoods.values/(2.*var))
    likelihoods.values /= likelihoods.values.sum()

    likelihoods = likelihoods.max(dim=('origin_idx', 'F0'))
    likelihoods.values /= 4.*np.pi*likelihoods.values.sum()

    return likelihoods.assign_attrs({
        'best_force': _max_force(likelihoods)
        })


def _marginals_regular(da, var):
    """ For each force orientation, calculates marginal likelihood value
    """

    likelihoods = da.copy()
    likelihoods.values = np.exp(-likelihoods.values/(2.*var))
    likelihoods.values /= likelihoods.values.sum()

    marginals = likelihoods.sum(dim=('origin_idx', 'F0'))
    marginals.values /= 4.*np.pi*marginals.values.sum()

    return marginals.assign_attrs({
        'best_force': _max_force(da)
        })


def _magnitudes_regular(da):
    """ For each source type, calculates magnitude of best-fitting moment tensor
    """
    phi = da.coords['phi']
    h = da.coords['h']

    nphi = len(phi)
    nh = len(h)

    misfit = da.min(dim=('origin_idx'))
    magnitudes = np.empty((nphi,nh))

    for ip in range(nphi):
        for ih in range(nh):
            sliced = misfit[:,ip,ih]
            argmin = np.argmin(sliced.values, axis=None)
            magnitudes[ip,ih] = da['F0'][argmin]

    magnitudes = DataArray(
        dims=('phi','h'),
        coords=(phi,h),
        data=magnitudes
        )

    return magnitudes.assign_attrs({
        'best_force': _min_force(da)
        })


def _min_force(da):
    """ Returns force corresponding to minimum overall value
    """
    da = dataarray_idxmin(da)
    keys = ['phi', 'h']
    vals = [da[key].values for key in keys]
    return vals


def _max_force(da):
    """ Returns force corresponding to maximum overall value
    """
    da = dataarray_idxmax(da)
    keys = ['phi', 'h']
    vals = [da[key].values for key in keys]
    return vals



#
# for extracting misfit or likelihood from irregularly-spaced grids
#

def _misfit_random(df, **kwargs):
    df = df.copy()
    df = df.reset_index()
    da = _bin(df, lambda df: df.min(), **kwargs)

    return da.assign_attrs({
        'best_force': _min_force(da)
        })


def _likelihoods_random(df, var, **kwargs):
    df = df.copy()
    df = np.exp(-df/(2.*var))
    df = df.reset_index()

    da = _bin(df, lambda df: df.max(), **kwargs)
    da.values /= 4.*np.pi*da.values.sum()

    return da.assign_attrs({
        'best_force': _max_force(da)
        })


def _marginals_random(df, var, **kwargs):
    df = df.copy()
    df = np.exp(-df/(2.*var))
    df = df.reset_index()

    da = _bin(df, lambda df: df.sum()/len(df))
    da.values /= 4.*np.pi*da.values.sum()

    return da.assign_attrs({
        'best_force': _max_force(da)
        })


#
# bins irregularly-spaced forces into phi,h rectangles
#

def _bin(df, handle, npts_phi=60, npts_h=30):
    """ Bins DataFrame into rectangular cells
    """
    # define centers of cells
    centers_phi = open_interval(0., 360., npts_phi)
    centers_h = open_interval(-1., +1., npts_h)

    # define corners of cells
    phi = closed_interval(0., 360, npts_phi+1)
    h = closed_interval(-1., +1., npts_h+1)

    binned = np.empty((npts_h, npts_phi))
    for _i in range(npts_h):
        for _j in range(npts_phi):
            # which grid points lie within cell (i,j)?
            subset = df.loc[
                df['phi'].between(phi[_j], phi[_j+1]) &
                df['h'].between(h[_i], h[_i+1])]

            if len(subset)==0:
                print("Encountered empty bin\n"
                      "phi: %f, %f\n"
                      "h: %f, %f\n" %
                      (phi[_j], phi[_j+1], h[_i], h[_i+1]) )

            binned[_i, _j] = handle(subset[0])

    return DataArray(
        dims=('phi', 'h'),
        coords=(centers_phi, centers_h),
        data=binned.transpose()
        )



#
# utility functions
#

def _check(ds):
    """ Checks data structures
    """
    if type(ds) not in (DataArray, DataFrame, MTUQDataArray, MTUQDataFrame):
        raise TypeError("Unexpected grid format")


def defaults(kwargs, defaults):
    for key in defaults:
        if key not in kwargs:
           kwargs[key] = defaults[key]

