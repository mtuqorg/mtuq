
# 
# graphics/uq/omega.py - uncertainty quantification over force or moment tensor
# angular distance
#

import numpy as np

from mtuq import Force, MomentTensor
from mtuq.graphics.uq._matplotlib import _plot_omega_matplotlib
from mtuq.grid_search import DataArray, DataFrame
from mtuq.util import warn
from mtuq.util.math import to_mij
from pandas import DataFrame



def plot_pdf(filename, df, var, m0=None, nbins=50, normalized=False, **kwargs):
    """ Plots probability density function over angular distance

    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``df`` (`DataFrame`):
    Data structure containing moment tensors and corresponding misfit values

    ``var`` (`float` or `array`):
    Data variance

    ``nbins`` (`int`):
    Number of angular distance bins

    ``normalized`` (`bool`):
    Normalize each angular distance bin by volume of corresponding shell?

    """
    if not isuniform(df):
        warn('plot_pdf requires randomly-drawn grid')
        return

    omega, pdf = _calculate_pdf(df, var, m0=m0, nbins=nbins, 
        normalized=normalized)

    _plot_omega(filename, omega, pdf, **kwargs)



def plot_cdf(filename, df, var, m0=None, nbins=50, normalized=False, **kwargs):
    """ Plots cumulative distribution function over angular distance

    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``df`` (`DataFrame`):
    Data structure containing moment tensors and corresponding misfit values

    ``var`` (`float` or `array`):
    Data variance

    ``nbins`` (`int`):
    Number of angular distance bins

    ``normalized`` (`bool`): 
    Normalize each angular distance bin by volume of corresponding shell?

    """
    if not isuniform(df):
        warn('plot_cdf requires randomly-drawn grid')
        return

    omega, pdf = _calculate_pdf(df, var, m0=m0, nbins=nbins, 
        normalized=normalized)

    _plot_omega(filename, omega, np.cumsum(pdf), **kwargs)



def plot_screening_curve(filename, ds, var, nbins=50, **kwargs):
    """ Plots explosion screening curve (maximum likelihood versus angular
    distance)

    ``filename`` (`str`):
    Name of output image file

    ``df`` (`DataFrame`):
    Data structure containing moment tensors and corresponding misfit values

    ``var`` (`float` or `array`):
    Data variance

    ``nbins`` (`int`):
    Number of angular distance bins

    """
    if issubclass(type(ds), DataArray):
        omega, values = _screening_curve_regular(ds, var, nbins=nbins)

    elif issubclass(type(ds), DataFrame):
        omega, values = _screening_curve_random(ds, var, nbins=nbins)

    _plot_omega(filename, omega, values, **kwargs)



def _plot_omega(filename, omega, values, backend=_plot_omega_matplotlib):
    # currently, does nothing except call backend
    backend(filename, omega, values)



#
# for extracting PDFs and explosion screening curves
#

ISO = MomentTensor([1.,1.,1.,0.,0.,0.])


def _calculate_pdf(df, var, m0=None, nbins=50, normalized=False):
    """ Calculates marginal probability density function over angular distance
    """
    likelihoods = np.exp(-df.copy()/(2.*var))

    return _map_omega(likelihoods, lambda pts: sum(pts), m0=m0, nbins=nbins,
        normalized=normalized)


def _screening_curve_random(df, var, nbins=50):
    """ Calculates explosion screening curve from randomly-drawn samples
    """
    likelihoods = np.exp(-df.copy()/(2.*var))

    return _map_omega(likelihoods, lambda pts: pts.max(), m0=ISO, nbins=nbins,
        normalized=False)


def _screening_curve_regular(da, var):
    """ Calculates explosion screening curve from regularly-spaced samples
    """
    raise NotImplementedError



#
# operations over angular distance
#

def _calculate_omega(df, m0=None):
    """ Calculates angular distances between vectors from DataFrame
    """
    # extract vectors from DataFrame
    if _type(df)=='MomentTensor':
        m = _to_array(df)

    elif _type(df)=='Force':
        raise NotImplementedError

    else:
        raise TypeError

    # extract reference vector
    if type(m0)==MomentTensor:
        # convert from lune to mij parameters
        m0 = m0.as_vector()

    elif type(m0)==Force:
        raise NotImplementedError

    elif not m0:
        # assume df holds likelihoods, try maximum likelihood estimate
        idx = _argmax(df)
        m0 = m[idx,:]

    # vectorized dot product
    dp = np.dot(m, m0)
    dp /= np.sum(m0**2)**0.5
    dp /= np.sum(m**2, axis=1)**0.5

    # return angles as NumPy array
    omega = 180./np.pi * np.arccos(dp)
    return omega



def _map_omega(df, func, m0=None, nbins=50, normalized=False):
    """ Maps function over angular distance bins
    """
    omega = _calculate_omega(df, m0=m0)

    centers, binned, counts =  _bin_omega(omega, df[0].values, nbins=nbins)

    values = np.zeros(len(binned))
    for _i in range(len(binned)):
        values[_i] = func(binned[_i])

    # normalize each bin by volume of angular distance shell?
    if normalized:
        values /= counts

    return centers, values



def _bin_omega(omega, values, nbins=50, mask_empty=True):
    """ Bins by angular distance
    """
    centers = np.linspace(0, 180., nbins+2)[1:-1]
    edges = np.linspace(0., 180., nbins+1)
    delta = np.diff(edges)

    binned = []
    counts = np.zeros(nbins)

    for _i in range(nbins):
        indices = np.argwhere(np.abs(omega - centers[_i]) < delta[_i])

        if indices.size==0:
            print('Encountered empty bin')
            binned += [[]]
            continue

        binned += [values[indices]]
        counts[_i] = len(indices)

    # exclude empty bins?
    if mask_empty:
        mask = np.argwhere(counts > 0).squeeze()
        return centers[mask], [binned[_i] for _i in mask.tolist()], counts[mask]

    else:
        return centers, binned, counts


#
# utility functions
#

def _argmax(df):

    df = df.copy()
    df = df.reset_index()
    return df[0].idxmax()


def _to_array(df):

    df = df.copy()
    df = df.reset_index()

    try:
        return np.ascontiguousarray(to_mij(
            df['rho'].to_numpy(),
            df['v'].to_numpy(),
            df['w'].to_numpy(),
            df['kappa'].to_numpy(),
            df['sigma'].to_numpy(),
            df['h'].to_numpy(),
            ))
    except:
        raise TypeError


def _type(ds):

    try:
        # DataArray
        dims = ds.dims
    except:
        # DataFrame
        ds = ds.copy()
        ds = ds.reset_index()
        dims = list(ds.columns.values)

    if 'rho' in dims\
       and 'v' in dims\
       and 'w' in dims\
       and 'kappa' in dims\
       and 'sigma' in dims\
       and 'h' in dims:
        return 'MomentTensor'

    elif 'F0' in dims\
       and 'phi' in dims\
       and 'h' in dims:
        return 'Force'


def isuniform(ds):
    if issubclass(type(ds), DataFrame):
        return True


