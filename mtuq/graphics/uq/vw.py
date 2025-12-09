
# 
# graphics/uq/vw.py - uncertainty quantification on the v,w rectangle
#

import numpy as np
import pandas
import xarray

from mtuq.grid.moment_tensor import _semiregular
from mtuq.grid_search import DataArray, DataFrame, MTUQDataArray, MTUQDataFrame
from mtuq.graphics.uq._gmt import _plot_vw_gmt
from mtuq.graphics.uq._matplotlib import _plot_vw_matplotlib
from mtuq.util import dataarray_idxmin, dataarray_idxmax, defaults, product
from mtuq.util.math import closed_interval, open_interval,\
    to_v, to_w, to_gamma, to_delta, to_mij, to_Mw


#
# For details about the v,w rectangle see 
# Tape2015 - A uniform parameterization of moment tensors
# (https://doi.org/10.1093/gji/ggv262)
#

v_min = -1./3.
v_max = +1./3.
w_min = -3.*np.pi/8.
w_max = +3.*np.pi/8.
vw_area = (v_max-v_min)*(w_max-w_min)



def plot_misfit_vw(filename, ds, **kwargs):
    """ Plots misfit values on v,w rectangle

    .. rubric :: Required input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing moment tensors and corresponding misfit values


    .. rubric :: Optional input arguments

    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_vw.html>`_

    """
    defaults(kwargs, {
        'colormap': 'viridis',
        })

    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        misfit = _misfit_vw_regular(ds)

    elif issubclass(type(ds), DataFrame):
        misfit = _misfit_vw_random(ds)

    _plot_vw(filename, misfit, **kwargs)


def plot_likelihood_vw(filename, ds, var, **kwargs):
    """ Plots maximum likelihood values on v,w rectangle

    .. rubric :: Required input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing moment tensors and corresponding misfit values

    ``var`` (`float` or `array`):
    Data variance


    .. rubric :: Optional input arguments

    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_vw.html>`_

    """
    defaults(kwargs, {
        'colormap': 'hot_r',
        })

    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        likelihoods = _likelihoods_vw_regular(ds, var)

    elif issubclass(type(ds), DataFrame):
        likelihoods = _likelihoods_vw_random(ds, var)

    _plot_vw(filename, likelihoods, **kwargs)


def plot_marginal_vw(filename, ds, var, **kwargs):
    """ Plots marginal likelihoods on v,w rectangle


    .. rubric :: Required input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing moment tensors and corresponding misfit values

    ``var`` (`float` or `array`):
    Data variance


    .. rubric :: Optional input arguments

    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_vw.html>`_

    """
    defaults(kwargs, {
        'colormap': 'hot_r',
        })

    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        marginals = _marginals_vw_regular(ds, var)

    elif issubclass(type(ds), DataFrame):
        marginals = _marginals_vw_random(ds, var)

    _plot_vw(filename, marginals, **kwargs)



def plot_variance_reduction_vw(filename, ds, data_norm, **kwargs):
    """ Plots variance reduction on v,w rectangle

    .. rubric :: Required input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing moment tensors and corresponding misfit values

    ``var`` (`float` or `array`):
    Data variance


    .. rubric :: Optional input arguments

    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_vw.html>`_

    """
    defaults(kwargs, {
        'colormap': 'viridis_r',
        })

    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        variance_reduction = _variance_reduction_vw_regular(ds, data_norm)

    elif issubclass(type(ds), DataFrame):
        variance_reduction = _variance_reduction_vw_random(ds, data_norm)

    _plot_vw(filename, variance_reduction, **kwargs)




def _plot_vw(filename, da, show_best=True, show_tradeoffs=False, 
    backend=_plot_vw_matplotlib, **kwargs):

    """ Plots DataArray values on vw rectangle

    .. rubric :: Keyword arguments

    ``colormap`` (`str`):
    Color palette used for plotting values 
    (choose from GMT or MTUQ built-ins)

    ``show_best`` (`bool`):
    Show where best-fitting moment tensor falls on vw rectangle

    ``title`` (`str`):
    Optional figure title

    ``backend`` (`function`):
    Choose from `_plot_vw_gmt` (default), `plot_vw_matplotlib`, 
    or user-supplied function

    """

    best_vw = None
    lune_array = None

    if show_best:
        if 'best_vw' in da.attrs:
            best_vw = da.attrs['best_vw']
        else:
            warn("Best-fitting moment tensor not given")

    if show_tradeoffs:
        if 'lune_array' in da.attrs:
            lune_array = da.attrs['lune_array']
        else:
            warn("Focal mechanism tradeoffs not given")

    backend(filename,
        da.coords['v'],
        da.coords['w'],
        da.values.transpose(),
        best_vw=best_vw,
        lune_array=lune_array,
        **kwargs)


#
# for extracting misfit, variance reduction, or likelihood from
# regularly-spaced grids
#

def _misfit_vw_regular(da, slice_magnitude=False, slice_origin=True):
    """ For each source type, extracts minimum misfit
    """
    # what are the indices of the best source?
    argmin = {}
    for dim,idx in zip(da.dims, np.unravel_index(da.argmin(), da.shape)):
        argmin[dim] = idx

    if slice_origin:
        da = da.isel(origin_idx=[argmin['origin_idx']])

    if slice_magnitude:
        da = da.isel(rho=[argmin['rho']])

    misfit = da.min(dim=('rho', 'kappa', 'sigma', 'h', 'origin_idx'))

    return misfit.assign_attrs({
        'best_mt': _min_mt(da),
        'best_vw': _min_vw(da),
        'lune_array': _lune_array(da),
        })


def _likelihoods_vw_regular(da, var):
    """ For each source type, calculates maximum likelihood value
    """
    likelihoods = da.copy()
    likelihoods.values = np.exp(-likelihoods.values/(2.*var))
    likelihoods.values /= likelihoods.values.sum()

    likelihoods = likelihoods.max(dim=('origin_idx', 'rho', 'kappa', 'sigma', 'h'))
    likelihoods.values /= likelihoods.values.sum()
    likelihoods /= vw_area

    return likelihoods.assign_attrs({
        'best_mt': _min_mt(da),
        'best_vw': _min_vw(da),
        'lune_array': _lune_array(da),
        'maximum_likelihood_estimate': dataarray_idxmax(likelihoods).values(),
        })


def _marginals_vw_regular(da, var):
    """ For each source type, calculates marginal likelihood value
    """

    likelihoods = da.copy()
    likelihoods.values = np.exp(-likelihoods.values/(2.*var))
    likelihoods.values /= likelihoods.values.sum()

    marginals = likelihoods.sum(dim=('origin_idx', 'rho', 'kappa', 'sigma', 'h'))
    marginals.values /= marginals.values.sum()
    marginals /= vw_area

    return marginals.assign_attrs({
        'best_vw': _max_vw(marginals),
        'marginal_vw': dataarray_idxmax(marginals).values(),
        })


def _magnitudes_vw_regular(da):
    """ For each source type, extracts magnitude of best-fitting moment tensor
    """
    v = da.coords['v']
    w = da.coords['w']

    nv = len(v)
    nw = len(w)

    misfit = da.min(dim=('origin_idx', 'kappa', 'sigma', 'h'))
    magnitudes = np.empty((nv,nw))

    for iv in range(nv):
        for iw in range(nw):
            sliced = misfit[:,iv,iw]
            argmin = np.argmin(sliced.values, axis=None)
            magnitudes[iv,iw] = to_Mw(da['rho'][argmin])

    magnitudes = DataArray(
        dims=('v','w'),
        coords=(v,w),
        data=magnitudes
        )

    return magnitudes.assign_attrs({
        'best_mt': _min_mt(da),
        'best_vw': _min_vw(da),
        'lune_array': _lune_array(da),
        })


def _variance_reduction_vw_regular(da, data_norm, percentage=False):
    """ For each source type, extracts maximum variance reduction
    """
    variance_reduction = 1. - da.copy()/data_norm

    variance_reduction = variance_reduction.max(
        dim=('origin_idx', 'rho', 'kappa', 'sigma', 'h'))

    if percentage:
        variance_reduction.values *= 100.

    return variance_reduction.assign_attrs({
        'best_mt': _min_mt(da),
        'best_vw': _min_vw(da),
        'lune_array': _lune_array(da),
        })


def _lune_array(da):
    """ For each source type, keeps track of best-fitting moment tensor
    """
    nv = len(da.coords['v'])
    nw = len(da.coords['w'])

    lune_array = np.empty((nv*nw,6))
    for iv in range(nv):
        for iw in range(nw):
            sliced = da[:,iv,iw,:,:,:,:]
            argmin = np.argmin(sliced.values, axis=None)
            idx = np.unravel_index(argmin, np.shape(sliced))

            lune_array[nw*iv+iw,0] = da['rho'][idx[0]]
            lune_array[nw*iv+iw,1] = da['v'][iv]
            lune_array[nw*iv+iw,2] = da['w'][iw]
            lune_array[nw*iv+iw,3] = da['kappa'][idx[1]]
            lune_array[nw*iv+iw,4] = da['sigma'][idx[2]]
            lune_array[nw*iv+iw,5] = da['h'][idx[3]]

    return lune_array


def _min_mt(da):
    """ Returns moment tensor vector corresponding to minimum DataArray value
    """
    da = dataarray_idxmin(da)
    lune_keys = ['rho', 'v', 'w', 'kappa', 'sigma', 'h']
    lune_vals = [da[key].values for key in lune_keys]
    return to_mij(*lune_vals)


def _max_mt(da):
    """ Returns moment tensor vector corresponding to maximum DataArray value
    """
    da = dataarray_idxmax(da)
    lune_keys = ['rho', 'v', 'w', 'kappa', 'sigma', 'h']
    lune_vals = [da[key].values for key in lune_keys]
    return to_mij(*lune_vals)


def _min_vw(da):
    """ Returns v,w coordinates corresponding to minimum DataArray value
    """
    da = dataarray_idxmin(da)
    lune_keys = ['v', 'w']
    lune_vals = [da[key].values for key in lune_keys]
    return lune_vals

def _max_vw(da):
    """ Returns v,w coordinates corresponding to maximum DataArray value
    """
    da = dataarray_idxmax(da)
    lune_keys = ['v', 'w']
    lune_vals = [da[key].values for key in lune_keys]
    return lune_vals


def _product_vw(*arrays, best_vw='max'):

    # evaluates product of arbitrarily many arrays
    da = product(*arrays)

    # any previous attributes no longer apply
    da = da.assign_attrs({
        'best_mt': None,
        'best_vw': None,
        'lune_array': None,
        'marginal_vw': None,
        })

    if best_vw=='min':
        return da.assign_attrs({'best_vw': _min_vw(da)})

    elif best_vw=='max':
        return da.assign_attrs({'best_vw': _max_vw(da)})

    else:
        return da



#
# for extracting misfit, variance reduction, or likelihood from
# irregularly-spaced grids
#

def _misfit_vw_random(df, **kwargs):
    df = df.copy()
    df = df.reset_index()
    da = _bin_vw_semiregular(df, lambda df: df.min(), **kwargs)

    return da.assign_attrs({
        'best_vw':  _min_vw(da),
        })


def _likelihoods_vw_random(df, var, **kwargs):
    df = df.copy()
    df = np.exp(-df/(2.*var))
    df = df.reset_index()

    da = _bin_vw_semiregular(df, lambda df: df.max(), **kwargs)
    da.values /= da.values.sum()
    da.values /= vw_area

    return da.assign_attrs({
        'best_vw': _max_vw(da),
        })


def _marginals_vw_random(df, var, **kwargs):
    df = df.copy()
    df = np.exp(-df/(2.*var))
    df = df.reset_index()

    da = _bin_vw_semiregular(df, lambda df: df.sum()/len(df))
    da.values /= da.values.sum()
    da.values /= vw_area

    return da


def _variance_reduction_vw_random(df, data_norm, **kwargs):
    """ For each source type, extracts minimum misfit
    """
    df = df.copy()
    df = 1 - df/data_norm
    df = df.reset_index()
    da = _bin_vw_semiregular(df, lambda df: df.max(), **kwargs)

    return da.assign_attrs({
        'best_vw':  _max_vw(da),
        })


#
# bins irregularly-spaced moment tensors into v,w rectangles
#

def _bin_vw_regular(df, handle, npts_v=20, npts_w=40):
    """ Bins irregularly-spaced moment tensors into square v,w cells
    """
    # define centers of cells
    centers_v = open_interval(-1./3., 1./3., npts_v)
    centers_w = open_interval(-3./8.*np.pi, 3./8.*np.pi, npts_w)

    # define corners of cells
    v = closed_interval(-1./3., 1./3., npts_v+1)
    w = closed_interval(-3./8.*np.pi, 3./8.*np.pi, npts_w+1)

    binned = np.empty((npts_w, npts_v))
    for _i in range(npts_w):
        for _j in range(npts_v):
            # which grid points lie within cell (i,j)?
            subset = df.loc[
                df['v'].between(v[_j], v[_j+1]) &
                df['w'].between(w[_i], w[_i+1])]

            binned[_i, _j] = handle(subset[0])

    return DataArray(
        dims=('v', 'w'),
        coords=(centers_v, centers_w),
        data=binned.transpose()
        )


def _bin_vw_semiregular(df, handle, npts_v=20, npts_w=40, tightness=0.6, normalize=False):
    """ Bins irregularly-spaced moment tensors into rectangular v,w cells
    """
    # at which points will we plot values?
    centers_v, centers_w = _semiregular(
        npts_v, npts_w, tightness=tightness)

    # what cell edges correspond to the above centers?
    centers_gamma = to_gamma(centers_v)
    edges_gamma = np.array(centers_gamma[:-1] + centers_gamma[1:])/2.
    edges_v = to_v(edges_gamma)

    centers_delta = to_delta(centers_w)
    edges_delta = np.array(centers_delta[:-1] + centers_delta[1:])/2.
    edges_w = to_w(edges_delta)

    edges_v = np.pad(edges_v, 1)
    edges_v[0] = -1./3.
    edges_v[-1] = +1./3.

    edges_w = np.pad(edges_w, 1)
    edges_w[0] = -3.*np.pi/8.
    edges_w[-1] = +3.*np.pi/8


    # bin grid points into cells
    binned = np.empty((npts_w, npts_v))
    binned[:] = np.nan
    for _i in range(npts_w):
        for _j in range(npts_v):
            # which grid points lie within cell (i,j)?
            subset = df.loc[
                df['v'].between(edges_v[_j], edges_v[_j+1]) &
                df['w'].between(edges_w[_i], edges_w[_i+1])]

            if len(subset)==0:
                print("Encountered empty bin")

            binned[_i, _j] = handle(subset[0])

            if normalize:
              # normalize by area of cell
              binned[_i, _j] /= edges_v[_j+1] - edges_v[_j]
              binned[_i, _j] /= edges_w[_i+1] - edges_w[_i]

    return DataArray(
        dims=('v', 'w'),
        coords=(centers_v, centers_w),
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

