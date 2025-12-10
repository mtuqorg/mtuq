"""
Waveform misfit module (non-optimized pure Python version)

See ``mtuq/misfit/waveform/__init__.py`` for more information
"""

import numpy as np

from mtuq.misfit.waveform._stats import _flatten, calculate_norm_data
from mtuq.util import AttribDict
from mtuq.util.math import isclose, list_intersect_with_indices
from mtuq.util.signal import get_components


def misfit(data, greens, sources, norm, time_shift_groups,
    time_shift_min, time_shift_max, msg_handle, 
    normalize=False, set_attributes=False):
    """
    Waveform misfit function (non-optimized pure Python version)

    See ``mtuq/misfit/waveform/__init__.py`` for more information
    """
    values = np.zeros((len(sources), 1))

    if normalize:
        components = _flatten(time_shift_groups)
        norm_data = calculate_norm_data(data, norm, components)

    #
    # initialize Green's function machinery
    #
    for _j, d in enumerate(data):
        greens[_j]._set_components(get_components(d))

    #
    # iterate over sources
    #
    for _i, source in enumerate(sources):

        # optional progress message
        msg_handle()

        #
        # iterate over stations
        #
        for _j, d in enumerate(data):

            components = greens[_j].components
            if not components:
                continue

            # generate synthetics
            s = greens[_j].get_synthetics(source, inplace=True)

            # time sampling scheme
            npts = d[0].data.size
            dt = d[0].stats.delta

            padding_left = int(round(+time_shift_max/dt))
            padding_right = int(round(-time_shift_min/dt))
            npts_padding = padding_left + padding_right

            # array to hold cross correlations
            corr = np.zeros(npts_padding+1)

            for group in time_shift_groups:
                # Finds the time-shift between data and synthetics that yields
                # the maximum cross-correlation value across all components in 
                # a given group, subject to min/max constraints
                _, indices = list_intersect_with_indices(
                    components, group)

                corr[:] = 0.
                for _k in indices:
                    corr += np.correlate(d[_k].data, s[_k].data, 'valid')
                
                npts_shift = corr.argmax() - padding_right
                time_shift = npts_shift*dt

                # what start and stop indices will correctly shift synthetics
                # relative to data?
                idx_start = padding_left - npts_shift
                idx_stop = idx_start + npts

                for _k in indices:

                    # substract data from shifted synthetics
                    r = s[_k].data[idx_start:idx_stop] - d[_k].data

                    # sum the resulting residuals
                    if norm=='L1':
                        value = np.sum(abs(r))*dt

                    elif norm=='L2':
                        value = np.sum(r**2)*dt

                    elif norm=='hybrid':
                        value = np.sqrt(np.sum(r**2))*dt

                    try:
                        values[_i] += d[_k].weight * value
                    except:
                        values[_i] += value


                    if set_attributes:
                        if not hasattr(s[_k], 'attrs'):
                            s[_k].attrs = AttribDict()

                        #
                        # waveform-related attributes
                        #

                        s[_k].attrs.norm = norm

                        if normalize:
                            try:
                                s[_k].attrs.misfit = value / norm_data
                            except Exception:
                                s[_k].attrs.misfit = value
                        else:
                            s[_k].attrs.misfit = value

                        s[_k].attrs.idx_start = idx_start
                        s[_k].attrs.idx_stop = idx_stop


                        #
                        # phase-related attributes
                        #

                        s[_k].attrs.cc_max = corr.max()
                        
                        # "static_shift" is an optional user-supplied
                        # time shift applied during data processing

                        try:
                            static_shift = d[_k].attrs.static_shift
                        except:
                            static_shift = 0.

                        s[_k].attrs.static_shift = static_shift


                        # "time_shift" is the subsequent cross-correlation time shift 
                        # applied during misfit evaluation

                        s[_k].attrs.time_shift = time_shift

                        Ns = np.dot(s[_k].data,s[_k].data)**0.5
                        Nd = np.dot(d[_k].data,d[_k].data)**0.5

                        if Ns*Nd > 0:
                            max_cc = np.correlate(s[_k].data,d[_k].data,'valid').max()
                            s[_k].attrs.normalized_cc_max = max_cc/(Ns*Nd)
                        else:
                            s[_k].attrs.normalized_cc_max = np.nan

                        s[_k].attrs.time_shift_min = time_shift_min
                        s[_k].attrs.time_shift_max = time_shift_max


                        # "total_shift" is the total correction, or in other words
                        # the sum of static and cross-correlation time shifts

                        s[_k].attrs.total_shift = time_shift + static_shift


                        #
                        # amplitude-related attributes
                        #

                        s_max = s[_k].data[idx_start:idx_stop].max()
                        d_max = d[_k].data.max()

                        s[_k].attrs.amplitude_ratio = d_max/s_max
                        s[_k].attrs.log_amplitude_ratio = np.log(d_max/s_max)


    if normalize:
        values /= norm_data

    return values


