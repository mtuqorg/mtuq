
#
# graphics/waveform.py - plots of observed and synthetic waveforms
#

import numpy as np
import matplotlib.pyplot as pyplot

from collections import defaultdict
from matplotlib.font_manager import FontProperties
from mtuq.dataset import Dataset
from mtuq.event import MomentTensor, Force
from mtuq.graphics.annotations import trace_label_writer, station_label_writer,\
    _getattr
from mtuq.graphics.header import MomentTensorHeader, ForceHeader
from mtuq.util import warn
from mtuq.util.signal import get_components, m_to_deg
from obspy import Stream, Trace


#
# high-level plotting functions
#

def plot_waveforms1(filename, 
        data,
        synthetics,
        stations,
        origin,
        header=None,
        total_misfit=1., 
        normalize='maximum_amplitude',
        trace_label_writer=trace_label_writer,
        station_label_writer=station_label_writer,
        ):

    """ Creates data/synthetics comparison figure with 3 columns (Z, R, T)
    """
    if _isempty(data):
        raise Exception

    # how many stations have at least one trace?
    nstations = _count([data])

    #
    # initialize figure
    #

    fig, axes = _initialize(
       nrows=nstations,
       ncolumns=4,
       column_width_ratios=[1.,1.,1.],
       height=1.25*nstations,
       width=8.5,
       margin_right=0.5,
       header=header,
       header_height=1.5,
       station_labels=bool(station_label_writer),
       )

    _add_component_labels1(axes)

    max_amplitude = _max(data, synthetics)

    #
    # loop over stations
    #

    ir = 0

    for _i in range(len(stations)):

        # skip empty stations
        if len(data[_i])==0:
            continue

        # add station labels
        if station_label_writer is not None:
            station_label_writer(axes[ir][0], stations[_i], origin)

        #
        # plot traces
        #

        stream_dat = data[_i]
        stream_syn = synthetics[_i]

        for dat in stream_dat:
            component = dat.stats.channel[-1].upper()
            weight = _getattr(dat, 'weight', 1.)

            if not weight:
                continue

            # skip missing components
            try:
                syn = stream_syn.select(component=component)[0]
            except:
                warn('Missing component, skipping...')
                continue

            _plot_ZRT(axes[ir], 1, dat, syn, component, 
                normalize, trace_label_writer, max_amplitude, total_misfit)

        ir += 1

    _save(filename)
    pyplot.close()



def plot_waveforms2(filename, 
        data_bw,
        data_sw,
        synthetics_bw,
        synthetics_sw,
        stations,
        origin,
        header=None,
        total_misfit_bw=1., 
        total_misfit_sw=1., 
        normalize='maximum_amplitude',
        trace_label_writer=trace_label_writer,
        station_label_writer=station_label_writer,
        ):


    """ Creates data/synthetics comparison figure with 5 columns 
   (Pn Z, Pn R, Rayleigh Z, Rayleigh R, Love T)
    """
    # how many stations have at least one trace?
    nstations = _count([data_bw, data_sw])

    #
    # initialize figure
    #

    fig, axes = _initialize(
       nrows=nstations,
       ncolumns=6,
       column_width_ratios=[0.5,0.5,1.,1.,1.],
       height=1.25*nstations,
       width=10.,
       header=header,
       header_height=2.,
       station_labels=bool(station_label_writer),
       )

    _add_component_labels2(axes)

    # determine maximum trace amplitudes
    max_amplitude_bw = _max(data_bw, synthetics_bw)
    max_amplitude_sw = _max(data_sw, synthetics_sw)


    #
    # loop over stations
    #

    ir = 0

    for _i in range(len(stations)):

        # skip empty stations
        if len(data_bw[_i])==len(data_sw[_i])==0:
            continue

        # add station labels
        if station_label_writer is not None:
            station_label_writer(axes[ir][0], stations[_i], origin)

        #
        # plot body wave traces
        #

        stream_dat = data_bw[_i]
        stream_syn = synthetics_bw[_i]

        for dat in stream_dat:
            component = dat.stats.channel[-1].upper()
            weight = _getattr(dat, 'weight', 1.)

            if not weight:
                continue

            # skip missing components
            try:
                syn = stream_syn.select(component=component)[0]
            except:
                warn('Missing component, skipping...')
                continue

            _plot_ZR(axes[ir], 1, dat, syn, component, 
                normalize, trace_label_writer, max_amplitude_bw, total_misfit_bw)


        #
        # plot surface wave traces
        #

        stream_dat = data_sw[_i]
        stream_syn = synthetics_sw[_i]

        for dat in stream_dat:
            component = dat.stats.channel[-1].upper()
            weight = _getattr(dat, 'weight', 1.)

            if not weight:
                continue

            # skip missing components
            try:
                syn = stream_syn.select(component=component)[0]
            except:
                warn('Missing component, skipping...')
                continue

            _plot_ZRT(axes[ir], 3, dat, syn, component,
                normalize, trace_label_writer, max_amplitude_sw, total_misfit_sw)


        ir += 1

    _save(filename)
    pyplot.close()


def plot_data_greens1(filename,
        data,
        greens,
        process_data,
        misfit,
        stations,
        origin,
        source,
        source_dict,
        **kwargs):

    """ Creates data/synthetics comparison figure with 3 columns (Z, R, T)

    Different input arguments, same result as plot_waveforms1
    """

    # collect synthetic waveforms with misfit attributes attached
    synthetics = misfit.collect_synthetics(data, greens.select(origin), source)

    # calculate total misfit for display in figure header
    total_misfit = misfit(data, greens.select(origin), source, optimization_level=0)

    # prepare figure header
    if 'header' in kwargs:
        header = kwargs.pop('header')

    else:
        model = _get_tag(greens[0].tags, 'model')
        solver = _get_tag(greens[0].tags, 'solver')

        header = _prepare_header(
            model, solver, source, source_dict, origin,
            process_data, misfit, total_misfit)

    plot_waveforms1(filename,
        data, synthetics, stations, origin,
        header=header, total_misfit=total_misfit, **kwargs)


def plot_data_greens2(filename,
        data_bw,
        data_sw,
        greens_bw,
        greens_sw,
        process_data_bw,
        process_data_sw,
        misfit_bw,
        misfit_sw,
        stations,
        origin,
        source,
        source_dict,
        **kwargs):

    """ Creates data/synthetics comparison figure with 5 columns 
    (Pn Z, Pn R, Rayleigh Z, Rayleigh R, Love T)

    Different input arguments, same result as plot_waveforms2
    """

    # collect synthetic waveforms with misfit attributes attached
    synthetics_bw = misfit_bw.collect_synthetics(
        data_bw, greens_bw.select(origin), source)

    synthetics_sw = misfit_sw.collect_synthetics(
        data_sw, greens_sw.select(origin), source)


    # calculate total misfit for display in figure header
    total_misfit_bw = misfit_bw(
        data_bw, greens_bw.select(origin), source, optimization_level=0)

    total_misfit_sw = misfit_sw(
        data_sw, greens_sw.select(origin), source, optimization_level=0) 


    # prepare figure header
    if 'header' in kwargs:
        header = kwargs.pop('header')

    else:
        model = _get_tag(greens_sw[0].tags, 'model')
        solver = _get_tag(greens_sw[0].tags, 'solver')

        header = _prepare_header(
            model, solver, source, source_dict, origin,
            process_data_bw, process_data_sw,
            misfit_bw, misfit_sw, total_misfit_bw, total_misfit_sw)

    plot_waveforms2(filename,
        data_bw, data_sw, synthetics_bw, synthetics_sw, stations, origin,
        total_misfit_bw=total_misfit_bw, total_misfit_sw=total_misfit_sw,
        header=header, **kwargs)



#
# low-level plotting utilities
#


def _initialize(nrows=None, ncolumns=None, column_width_ratios=None, 
    header=None, height=None, width=None, margin_top=0.25, margin_bottom=0.25,
    margin_left=0.25, margin_right=0.25, header_height=1.5, 
    station_labels=True, station_label_width=0.5):

    if header:
        height += header_height

    if not station_labels:
        station_label_width = 0.

    height += margin_bottom
    height += margin_top
    width += margin_left
    width += margin_right

    fig, axes = pyplot.subplots(nrows, ncolumns,
        figsize=(width, height),
        subplot_kw=dict(clip_on=False),
        gridspec_kw=dict(width_ratios=[station_label_width]+column_width_ratios)
        )

    pyplot.subplots_adjust(
        left=margin_left/width,
        right=1.-margin_right/width,
        bottom=margin_bottom/height,
        top=1.-(header_height+margin_top)/height,
        wspace=0.,
        hspace=0.,
        )

    _hide_axes(axes)

    if header:
        header.write(
            header_height, width,
            margin_left, margin_top)

    # single station plotting workaround
    if nrows==1:
        axes = [axes]

    return fig, axes


def _plot_ZRT(axes, ic, dat, syn, component, 
    normalize='maximum_amplitude', trace_label_writer=None,
    max_amplitude=1., total_misfit=1.):

    # plot traces
    if component=='Z':
        axis = axes[ic+0]
    elif component=='R':
        axis = axes[ic+1]
    elif component=='T':
        axis = axes[ic+2]
    else:
        return

    _plot(axis, dat, syn)

    # normalize amplitude
    if normalize=='trace_amplitude':
        max_trace = _max(dat, syn)
        ylim = [-1.5*max_trace, +1.5*max_trace]
        axis.set_ylim(*ylim)
    elif normalize=='station_amplitude':
        max_stream = _max(stream_dat, stream_syn)
        ylim = [-1.5*max_stream, +1.5*max_stream]
        axis.set_ylim(*ylim)
    elif normalize=='maximum_amplitude':
        ylim = [-0.75*max_amplitude, +0.75*max_amplitude]
        axis.set_ylim(*ylim)

    if trace_label_writer is not None:
        trace_label_writer(axis, dat, syn, total_misfit)


def _plot_ZR(axes, ic, dat, syn, component, 
    normalize='maximum_amplitude', trace_label_writer=None,
    max_amplitude=1., total_misfit=1.):

    # plot traces
    if component=='Z':
        axis = axes[ic+0]
    elif component=='R':
        axis = axes[ic+1]
    else:
        return

    _plot(axis, dat, syn)

    # normalize amplitude
    if normalize=='trace_amplitude':
        max_trace = _max(dat, syn)
        ylim = [-1.5*max_trace, +1.5*max_trace]
        axis.set_ylim(*ylim)
    elif normalize=='station_amplitude':
        max_stream = _max(stream_dat, stream_syn)
        ylim = [-1.5*max_stream, +1.5*max_stream]
        axis.set_ylim(*ylim)
    elif normalize=='maximum_amplitude':
        ylim = [-0.75*max_amplitude, +0.75*max_amplitude]
        axis.set_ylim(*ylim)


    if trace_label_writer is not None:
        trace_label_writer(axis, dat, syn, total_misfit)


def _plot(axis, dat, syn, label=None):
    """ Plots data and synthetics time series on current axes
    """
    t1,t2,nt,dt = _time_stats(dat)

    start = _getattr(syn, 'start', 0)
    stop = _getattr(syn, 'stop', len(syn.data))

    t = np.linspace(0,t2-t1,nt,dt)
    d = dat.data
    s = syn.data

    axis.plot(t, d, 'k', linewidth=1.5,
        clip_on=False, zorder=10)
    axis.plot(t, s[start:stop], 'r', linewidth=1.25, 
        clip_on=False, zorder=10)


def _add_component_labels1(axes, body_wave_labels=True, surface_wave_labels=True):
    """ Displays component name above each column
    """
    font = FontProperties()
    font.set_weight('bold')

    ax = axes[0][1]
    pyplot.text(0.,0.70, 'Z', fontproperties=font, fontsize=16,
        transform=ax.transAxes)

    ax = axes[0][2]
    pyplot.text(0.,0.70, 'R', fontproperties=font, fontsize=16,
        transform=ax.transAxes)

    ax = axes[0][3]
    pyplot.text(0.,0.70, 'T', fontproperties=font, fontsize=16,
        transform=ax.transAxes)


def _add_component_labels2(axes, body_wave_labels=True, surface_wave_labels=True):
    """ Displays component name above each column
    """
    font = FontProperties()
    font.set_weight('bold')

    ax = axes[0][1]
    pyplot.text(0.,0.70, 'Z', fontproperties=font, fontsize=16,
        transform=ax.transAxes)

    ax = axes[0][2]
    pyplot.text(0.,0.70, 'R', fontproperties=font, fontsize=16,
        transform=ax.transAxes)

    ax = axes[0][3]
    pyplot.text(0.,0.70, 'Z', fontproperties=font, fontsize=16,
        transform=ax.transAxes)

    ax = axes[0][4]
    pyplot.text(0.,0.70, 'R', fontproperties=font, fontsize=16,
        transform=ax.transAxes)

    ax = axes[0][5]
    pyplot.text(0.,0.70, 'T', fontproperties=font, fontsize=16,
        transform=ax.transAxes)


#
# utility functions
#

def _time_stats(trace):
    # returns time scheme
    return (
        float(trace.stats.starttime),
        float(trace.stats.endtime),
        trace.stats.npts,
        trace.stats.delta,
        )


def _count(datasets):
    # counts number of nonempty streams in dataset(s)
    count = 0
    for streams in zip(*datasets):
        for stream in streams:
            if len(stream) > 0:
                count += 1
                break
    return count


def _isempty(dataset):
    if not dataset:
        return True
    else:
        return bool(_count([dataset])==0)


def _max(dat, syn):
    if type(dat)==type(syn)==Trace:
        return max(
            abs(dat.max()),
            abs(syn.max()))

    elif type(dat)==type(syn)==Stream:
        return max(
            max(map(abs, dat.max())),
            max(map(abs, syn.max())))

    elif type(dat)==type(syn)==Dataset:
        return max(
            abs(dat.max()),
            abs(syn.max()))

    else:
        raise TypeError



def _hide_axes(axes):
    # hides axes lines, ticks, and labels
    for row in axes:
        for col in row:
            col.spines['top'].set_visible(False)
            col.spines['right'].set_visible(False)
            col.spines['bottom'].set_visible(False)
            col.spines['left'].set_visible(False)
            col.get_xaxis().set_ticks([])
            col.get_yaxis().set_ticks([])
            col.patch.set_visible(False)


def _prepare_header(model, solver, source, source_dict, origin, *args):
    # prepares figure header

    if len(args)==3:
        args = [None, args[0], None, args[1], 0., args[2]]

    if type(source)==MomentTensor:
        return MomentTensorHeader(
            *args, model, solver, source, source_dict, origin)

    elif type(source)==Force:
        return ForceHeader(
            *args, model, solver, source, source_dict, origin)

    else:
        raise TypeError


def _save(filename):
    pyplot.savefig(filename)


def _get_tag(tags, pattern):
    for tag in tags:
        parts = tag.split(':')
        if parts[0]==pattern:
            return parts[1]
    else:
        return None


