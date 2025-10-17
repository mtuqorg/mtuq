
#
# graphics/waveform.py - plots of observed and synthetic waveforms
#

import numpy as np
import matplotlib.pyplot as pyplot

from matplotlib.font_manager import FontProperties
from mtuq.dataset import Dataset
from mtuq.event import MomentTensor, Force
from mtuq.graphics.annotations import trace_label_writer, station_label_writer,\
    _getattr
from mtuq.graphics.header import MomentTensorHeader, ForceHeader
from mtuq.util import Null, warn
from mtuq.util.signal import get_components, m_to_deg
from obspy import Stream, Trace


#
# high-level plotting functions
#

def plot_waveforms1(
        filename, 
        data,
        synthetics,
        stations,
        origin,
        header=None,
        total_misfit=1., 
        normalize='maximum_amplitude',
        trace_label_writer=trace_label_writer,
        station_label_writer=station_label_writer,
        dat_kwargs={},     
        syn_kwargs={},
        ):

    """ Creates data/synthetics comparison figure with 3 columns (Z, R, T)
    """
    if data is None:
        data = Null()

    if synthetics is None:
        synthetics = Null()

    # how many stations have at least one trace?
    nrows = _count(data, synthetics)

    # intialize figure
    fig, axes = _initialize1(nrows, header)

    _add_component_labels1(axes)

    # optional normalization
    if normalize=='maximum_amplitude':
        factor = _max(data, synthetics)
    elif normalize=='median_amplitude':
        factor = 2.*_median(data, synthetics)
    else:
        factor = None


    #
    # loop over stations
    #

    ir = 0

    for _i in range(len(stations)):

        # skip empty stations
        if len(data[_i])==0:
            continue

        # add station labels
        try:
            station_label_writer(axes[ir][0], stations[_i], origin)
        except:
            pass

        # plot traces
        _plot_stream(axes[ir], [1,2,3], ['Z','R','T'], 
                  data[_i], synthetics[_i],
                  normalize, factor, trace_label_writer, total_misfit,
                  dat_kwargs=dat_kwargs, syn_kwargs=syn_kwargs)

        ir += 1

    _save(filename)
    pyplot.close()



def plot_waveforms2(
        filename, 
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
        dat_kwargs={},
        syn_kwargs={},
        ):


    """ Creates data/synthetics comparison figure with 5 columns 
   (Pn Z, Pn R, Rayleigh Z, Rayleigh R, Love T)
    """
    if data_bw is None:
        data_bw = Null()

    if data_sw is None:
        data_sw = Null()

    if synthetics_bw is None:
        synthetics_bw = Null()

    if synthetics_sw is None:
        synthetics_sw = Null()

    # how many stations have at least one trace?
    nrows = _count(data_bw, data_sw, synthetics_bw, synthetics_bw)

    # intialize figure
    fig, axes = _initialize2(nrows, header)

    _add_component_labels2(axes)

    # optional normalization
    if normalize=='maximum_amplitude':
        factor_bw = _max(data_bw, synthetics_bw)
        factor_sw = _max(data_sw, synthetics_sw)
    elif normalize=='median_amplitude':
        factor_bw = 2.*_median(data_bw, synthetics_bw)
        factor_sw = 2.*_median(data_sw, synthetics_sw)
    else:
        factor_bw = None
        factor_sw = None

    #
    # loop over stations
    #

    ir = 0

    for _i in range(len(stations)):

        # skip empty stations
        if len(data_bw[_i])==len(data_sw[_i])==0:
            continue

        # add station labels
        try:
            station_label_writer(axes[ir][0], stations[_i], origin)
        except:
            pass

        # plot body wave traces
        _plot_stream(axes[ir], [1,2], ['Z','R'],
                     data_bw[_i], synthetics_bw[_i],
                     normalize, factor_bw, trace_label_writer, total_misfit_bw,
                     dat_kwargs=dat_kwargs, syn_kwargs=syn_kwargs)

        # plot surface wave traces
        _plot_stream(axes[ir], [3,4,5], ['Z','R','T'],
                     data_sw[_i], synthetics_sw[_i],
                     normalize, factor_sw, trace_label_writer, total_misfit_sw,
                     dat_kwargs=dat_kwargs, syn_kwargs=syn_kwargs)

        ir += 1

    _save(filename)
    pyplot.close()


def plot_waveforms3(
        filename, 
        data_bw,
        data_rayl,
        data_love,
        synthetics_bw,
        synthetics_rayl,
        synthetics_love,
        stations,
        origin,
        header=None,
        total_misfit_bw=1., 
        total_misfit_rayl=1., 
        total_misfit_love=1.,
        normalize='maximum_amplitude',
        trace_label_writer=trace_label_writer,
        station_label_writer=station_label_writer,
        dat_kwargs={},
        syn_kwargs={},
        ):

    """ Creates data/synthetics comparison figure with 5 columns 
    (Pn Z, Pn R, Rayleigh Z, Rayleigh R, Love T)
    """
    if data_bw is None: 
        data_bw = Null()

    if data_rayl is None: 
        data_rayl = Null()

    if data_love is None:
        data_love = Null()

    if synthetics_bw is None:
        synthetics_bw = Null()

    if synthetics_rayl is None:
        synthetics_rayl = Null()

    if synthetics_love is None:
        synthetics_love = Null()

    # how many stations have at least one trace?
    nrows = _count(data_bw, data_rayl, data_love)

    # intialize figure
    fig, axes = _initialize2(nrows, header)

    _add_component_labels2(axes)


    # optional normalization
    if normalize=='maximum_amplitude':
        factor_bw = _max(data_bw, synthetics_bw)
        factor_rayl = _max(data_rayl, synthetics_rayl)
        factor_love = _max(data_love, synthetics_love)

    elif normalize=='median_amplitude':
        factor_bw = 2.*_median(data_bw, synthetics_bw)
        factor_rayl = 2.*_median(data_rayl, synthetics_rayl)
        factor_love = 2.*_median(data_love, synthetics_love)

    else:
        factor_bw = None
        factor_rayeligh = None
        factor_love = None


    #
    # loop over stations
    #

    ir = 0

    for _i in range(len(stations)):

        # skip empty stations
        if len(data_bw[_i]) == len(data_rayl[_i]) == len(data_love[_i]) == 0:
            continue

        # add station labels
        try:
            station_label_writer(axes[ir][0], stations[_i], origin)
        except:
            pass

        # plot body waves
        _plot_stream(axes[ir], [1,2], ['Z','R'], 
                     data_bw[_i], synthetics_bw[_i],
                     normalize, factor_bw, trace_label_writer, total_misfit_bw,
                     dat_kwargs=dat_kwargs, syn_kwargs=syn_kwargs)
        
        # plot Rayleigh waves
        _plot_stream(axes[ir], [3,4], ['Z','R'],
                     data_rayl[_i], synthetics_rayl[_i],
                     normalize, factor_rayl, trace_label_writer, total_misfit_rayl,
                     dat_kwargs=dat_kwargs, syn_kwargs=syn_kwargs)

        # plot Love waves
        _plot_stream(axes[ir], [5], ['T'],
                     data_love[_i], synthetics_love[_i],
                     normalize, factor_love, trace_label_writer, total_misfit_love,
                     dat_kwargs=dat_kwargs, syn_kwargs=syn_kwargs)

        ir += 1

    _save(filename)
    pyplot.close()


def plot_data_greens1(
        filename,
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
    total_misfit = misfit(data, greens.select(origin), source, level=0)

    # prepare figure header
    if 'header' in kwargs:
        header = kwargs.pop('header')

    else:
        model = _get_tag(greens[0].tags, 'model')
        solver = _get_tag(greens[0].tags, 'solver')

        header = _prepare_header(
            model, solver, source, source_dict, origin,
            process_data, misfit, total_misfit, data_sw=data)

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
        data_bw, greens_bw.select(origin), source, level=0)

    total_misfit_sw = misfit_sw(
        data_sw, greens_sw.select(origin), source, level=0) 


    # prepare figure header
    if 'header' in kwargs:
        header = kwargs.pop('header')

    else:
        model = _get_tag(greens_sw[0].tags, 'model')
        solver = _get_tag(greens_sw[0].tags, 'solver')

        header = _prepare_header(
            model, solver, source, source_dict, origin,
            process_data_bw, process_data_sw,
            misfit_bw, misfit_sw, total_misfit_bw, total_misfit_sw,
            data_bw=data_bw, data_sw=data_sw)

    plot_waveforms2(filename,
        data_bw, data_sw, synthetics_bw, synthetics_sw, stations, origin,
        total_misfit_bw=total_misfit_bw, total_misfit_sw=total_misfit_sw,
        header=header, **kwargs)


def plot_data_greens3(
        filename,
        data_bw,
        data_rayl,
        data_love,
        greens_bw,
        greens_rayl,
        greens_love,
        process_data_bw,
        process_data_rayl,
        process_data_love,
        misfit_bw,
        misfit_rayl,
        misfit_love,
        stations,
        origin,
        source,
        source_dict,
        **kwargs):

    """ Creates data/synthetics comparison figure with 5 columns 
    (Pn Z, Pn R, Rayleigh Z, Rayleigh R, Love T)

    Different input arguments, same result as plot_waveforms3
    """

    # collect synthetic waveforms with misfit attributes attached
    synthetics_bw = misfit_bw.collect_synthetics(
        data_bw, greens_bw.select(origin), source)

    synthetics_rayl = misfit_rayl.collect_synthetics(
        data_rayl, greens_rayl.select(origin), source)

    synthetics_love = misfit_love.collect_synthetics(
        data_love, greens_love.select(origin), source)
    
    # calculate total misfit for display in figure header
    total_misfit_bw = misfit_bw(
        data_bw, greens_bw.select(origin), source, level=0)

    total_misfit_rayl = misfit_rayl(
        data_rayl, greens_rayl.select(origin), source, level=0) 

    total_misfit_love = misfit_love(
        data_love, greens_love.select(origin), source, level=0)

    # prepare figure header
    if 'header' in kwargs:
        header = kwargs.pop('header')
    else:
        model = _get_tag(greens_bw[0].tags, 'model')
        solver = _get_tag(greens_bw[0].tags, 'solver')

        header = _prepare_header(
            model, solver, source, source_dict, origin,
            process_data_bw, process_data_rayl, misfit_bw, misfit_rayl,
            total_misfit_bw, total_misfit_rayl, best_misfit_sw_supp=total_misfit_love,
            misfit_sw_supp = misfit_love, data_bw=data_bw, data_sw=data_rayl,
            data_sw_supp=data_love, process_sw_supp=process_data_love)

    plot_waveforms3(filename,
        data_bw, data_rayl, data_love,
        synthetics_bw, synthetics_rayl, synthetics_love,
        stations, origin,
        total_misfit_bw=total_misfit_bw, 
        total_misfit_rayl=total_misfit_rayl,
        total_misfit_love=total_misfit_love,
        header=header, **kwargs)


#
# low-level plotting utilities
#

def _initialize1(nrows, header):
    return _initialize(
       nrows=nrows,
       ncolumns=4,
       column_width_ratios=[1.,1.,1.],
       height=1.25*nrows,
       width=8.5,
       margin_right=0.5,
       header=header,
       header_height=1.5,
       station_labels=True,
       )


def _initialize2(nrows, header):
    return _initialize(
       nrows=nrows,
       ncolumns=6,
       column_width_ratios=[0.5,0.5,1.,1.,1.],
       height=1.25*nrows,
       width=10.,
       header=header,
       header_height=2.,
       station_labels=True,
       )


def _initialize(
    nrows=None, ncolumns=None, column_width_ratios=None, 
    header=None, height=None, width=None, margin_top=0.25, margin_bottom=0.25,
    margin_left=0.25, margin_right=0.25, header_height=1.5, 
    station_labels=True, station_label_width=0.5):

    if not header:
        header_height = 0.

    if not station_labels:
        station_label_width = 0.

    height += header_height

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

    if header:
        header.write(
            header_height, width,
            margin_left, margin_top)

    # single station plotting workaround
    if nrows==1:
        axes = [axes]

    _hide_axes(axes)

    return fig, axes


def _plot_stream(
    axes,
    column_indices,
    components,
    stream_dat,
    stream_syn,
    normalize='maximum_amplitude',
    amplitude_factor=None,
    trace_label_writer=None,
    total_misfit=1.,
    dat_kwargs={},
    syn_kwargs={},
    ):

    if normalize in [
        'maximum_amplitude',
        'median_amplitude',
        'station_amplitude',
        'trace_amplitude',
        ]:
        pass
    elif not normalize:
        pass
    else:
        warn(f"Unrecognized option: normalize={normalize}.\n\n"
             "Setting normalize='station_amplitude'")

        normalize='station_amplitude'

    for _i, component, in enumerate(components):
        axis = axes[column_indices[_i]]

        try:
            dat = stream_dat.select(component=component)[0]
        except:
            dat = None

        try:
            syn = stream_syn.select(component=component)[0]
        except:
            syn = None

        weight = _getattr(dat, 'weight', 1.)
        if not weight:
            continue

        if normalize=='trace_amplitude':
            max_trace = _max(dat, syn)
            ylim = [-1.5*max_trace, +1.5*max_trace]

        elif normalize=='station_amplitude':
            max_station = _max(stream_dat, stream_syn)
            ylim = [-1.25*max_station, +1.25*max_station]

        elif amplitude_factor:
            ylim = [-amplitude_factor, +amplitude_factor]

        _plot_trace(axis, dat, syn, dat_update=dat_kwargs, syn_update=syn_kwargs)

        try:
            axis.set_ylim(*ylim)
        except (UnboundLocalError, ValueError):
            pass

        try:
            trace_label_writer(axis, dat, syn, total_misfit)
        except:
            pass


def _plot_trace(axis, dat, syn, dat_update={}, syn_update={}):
    """ Plots data and synthetics time series on current axes
    """
    if dat is None and syn is None:
        # nothing to plot
        return

    dat_kwargs = {
        'color': 'black',
        'linewidth': 1.5,
        'clip_on': True,
        'zorder': 10,
        }

    syn_kwargs = {
        'color': 'red',
        'linewidth': 1.25,
        'clip_on': True,
        'zorder': 10,
        }

    dat_kwargs.update(dat_update)
    syn_kwargs.update(syn_update)


    if dat:
        t,d = _time_series(dat)

        axis.plot(t, d, **dat_kwargs)

    if syn:
        # which start and stop indices will correctly align synthetics?
        start = _getattr(syn, 'idx_start', 0)
        stop = _getattr(syn, 'idx_stop', len(syn.data))

        t,s = _time_series(syn)

        axis.plot(t[:stop-start], s[start:stop], **syn_kwargs)


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

def _time_series(trace):
    t1 = float(trace.stats.starttime)
    t2 = float(trace.stats.endtime)
    nt = trace.stats.npts
    dt = trace.stats.delta
    return np.linspace(0,t2-t1,nt,dt), trace.data



def _count(*datasets):
    datasets = filter(lambda x: x is not None, datasets)

    # counts number of nonempty streams in dataset(s)
    count = 0
    for streams in zip(*datasets):
        for stream in streams:
            if stream is None:
                continue
            if len(stream) > 0:
                count += 1
                break
    return count


def _isempty(dataset):
    if not dataset:
        return True
    else:
        return bool(_count(dataset)==0)


def _max(*datasets):
    # returns maximum amplitude over traces, streams, or datasets

    maxval = -np.inf
    maxall = -np.inf

    for ds in datasets:
        if type(ds) not in [Dataset, Stream, Trace, Null]:
            print('Expected a type Dataset, Stream, Trace, Null '
                  'but received a type f{type(ds)}.\nSkipping...')
            continue

        if type(ds)==Trace:
            maxval = abs(ds.max())

        elif type(ds)==Stream:
            if len(ds) > 0:
                maxval = max(map(abs, ds.max()))

        elif type(ds)==Dataset:
            maxval = abs(ds.max())

        elif type(ds)==Null:
            continue

        if maxval > maxall:
            maxall = maxval

    if maxall==-np.inf:
        print('Warning: maximum value undefined')

    return maxall


def _median(*datasets):
    # returns median Linf amplitude over traces in dataset

    _list = []
    for ds in datasets:
        assert type(ds)==Dataset
        for stream in ds:
            for trace in stream:
                trace_max = abs(trace.max())
                if trace_max > 0.:
                    _list.append(trace_max)
    return np.median(np.array(_list))


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


def _prepare_header(model, solver, source, source_dict, origin, *args, **kwargs):
    # prepares figure header

    if len(args)==3:
        args = [None, args[0], None, args[1], 0., args[2]]

    if type(source)==MomentTensor:
        return MomentTensorHeader(
            *args, model, solver, source, source_dict, origin, **kwargs)

    elif type(source)==Force:
        return ForceHeader(
            *args, model, solver, source, source_dict, origin, **kwargs)

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

