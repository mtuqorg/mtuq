#!/usr/bin/env python

import os
import numpy as np

from mtuq import read, open_db, download_greens_tensors
from mtuq.event import Origin
from mtuq.graphics import plot_data_greens2, plot_beachball, plot_polarities, plot_misfit_lune
from mtuq.grid import FullMomentTensorGridSemiregular
from mtuq.grid_search import grid_search
from mtuq.misfit import WaveformMisfit, PolarityMisfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath, merge_dicts, save_json
from mtuq.util.cap import parse_station_codes, Trapezoid



if __name__=='__main__':
    #
    # Joint waveform and polarity grid search over all moment tensor parameters
    #
    # USAGE
    #   mpirun -n <NPROC> python Waveforms+Polarities.py
    #   
    # For a simpler example, see SerialGridSearch.DoubleCouple.py
    #


    path_data=    fullpath('data/examples/20090407201255351/*.[zrt]')
    path_weights= fullpath('data/examples/20090407201255351/weights.dat')
    event_id=     '20090407201255351'
    model=        'ak135'


    #
    # Body and surface wave measurements will be made separately
    #

    process_bw = ProcessData(
        filter_type='Bandpass',
        freq_min= 0.1,
        freq_max= 0.333,
        pick_type='taup',
        taup_model=model,
        window_type='body_wave',
        window_length=15.,
        capuaf_file=path_weights,
        )

    process_sw = ProcessData(
        filter_type='Bandpass',
        freq_min=0.025,
        freq_max=0.0625,
        pick_type='taup',
        taup_model=model,
        window_type='surface_wave',
        window_length=150.,
        capuaf_file=path_weights,
        )


    #
    # We will jointly evaluate waveform differences and polarities
    #

    misfit_bw = WaveformMisfit(
        norm='L2',
        time_shift_min=-2.,
        time_shift_max=+2.,
        time_shift_groups=['ZR'],
        )

    misfit_sw = WaveformMisfit(
        norm='L2',
        time_shift_min=-10.,
        time_shift_max=+10.,
        time_shift_groups=['ZR','T'],
        )

    polarity_misfit = PolarityMisfit(
        taup_model=model)


    #
    # Observed polarities can be attached to the data or passed through a 
    # user-supplied dictionary or list in which +1 corresopnds to positive 
    # first motion, -1 to negative first moation, and 0 to indeterminate or
    # unpicked
    #

    #polarities = np.array([-1, -1, -1, 1, 1, 0, 1, 1, -1, 1, 1, 1, 0, 1, 1, 1, -1, 1, 1, 0])
    pol_data = np.genfromtxt('../data/examples/20090407201255351/weights.dat')
    polarities = pol_data[:,13]



    #
    # User-supplied weights control how much each station contributes to the
    # objective function
    #

    station_id_list = parse_station_codes(path_weights)


    #
    # Next, we specify the moment tensor grid and source-time function
    #

    grid = FullMomentTensorGridSemiregular(
        npts_per_axis=10,
        magnitudes=[4.4, 4.5, 4.6, 4.7])

    wavelet = Trapezoid(
        magnitude=4.5)


    #
    # Origin time and location will be fixed. For an example in which they 
    # vary, see examples/GridSearch.DoubleCouple+Magnitude+Depth.py
    #
    # See also Dataset.get_origins(), which attempts to create Origin objects
    # from waveform metadata
    #

    origin = Origin({
        'time': '2009-04-07T20:12:55.000000Z',
        'latitude': 61.454200744628906,
        'longitude': -149.7427978515625,
        'depth_in_m': 33033.599853515625,
        })


    from mpi4py import MPI
    comm = MPI.COMM_WORLD


    #
    # The main I/O work starts now
    #

    if comm.rank==0:
        print('Reading data...\n')
        data = read(path_data, format='sac', 
            event_id=event_id,
            station_id_list=station_id_list,
            tags=['units:m', 'type:velocity']) 


        data.sort_by_distance()
        stations = data.get_stations()


        print('Processing data...\n')
        data_bw = data.map(process_bw)
        data_sw = data.map(process_sw)


        print('Reading Greens functions...\n')
        greens = download_greens_tensors(stations, origin, model)

        print('Processing Greens functions...\n')
        greens.convolve(wavelet)
        greens_bw = greens.map(process_bw)
        greens_sw = greens.map(process_sw)


    else:
        stations = None
        data_bw = None
        data_sw = None
        greens_bw = None
        greens_sw = None


    stations = comm.bcast(stations, root=0)
    data_bw = comm.bcast(data_bw, root=0)
    data_sw = comm.bcast(data_sw, root=0)
    greens_bw = comm.bcast(greens_bw, root=0)
    greens_sw = comm.bcast(greens_sw, root=0)


    #
    # The main computational work starts now
    #

    if comm.rank==0:
        print('Evaluating body wave misfit...\n')

    results_bw = grid_search(
        data_bw, greens_bw, misfit_bw, origin, grid)

    if comm.rank==0:
        print('Evaluating surface wave misfit...\n')

    results_sw = grid_search(
        data_sw, greens_sw, misfit_sw, origin, grid)


    if comm.rank==0:
        print('Evaluating polarity misfit...\n')

    results_polarity = grid_search(
        polarities, greens_bw, polarity_misfit, origin, grid)


    if comm.rank==0:

        m = 0.1
        results = m*results_polarity/sum(abs(polarities)) + (1-m)*(results_bw+ results_sw)

        # `grid` index corresponding to minimum misfit
        idx = results.source_idxmin()

        best_mt = grid.get(idx)
        lune_dict = grid.get_dict(idx)
        mt_dict = best_mt.as_dict()


        #
        # Generate figures and save results
        #

        print('Generating figures...\n')

        plot_data_greens2(event_id+'FMT_waveforms.png',
            data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw,
            misfit_bw, misfit_sw, stations, origin, best_mt, lune_dict)

        plot_beachball(event_id+'FMT_beachball.png',
            best_mt, stations, origin)

        plot_misfit_lune(event_id+'FMT_misfit.png', results,
            title='Waveform Misfit')

        # generate polarity figures

        plot_misfit_lune(event_id+'FMT_misfit_polarity.png', results_polarity,
            show_best=False, title='Polarity Misfit', plot_type='scatter')

        # predicted polarities
        predicted = polarity_misfit.get_predicted(greens, best_mt)

        # station attributes
        attrs = polarity_misfit.collect_attributes(polarities, greens)

        plot_polarities(event_id+'FMT_beachball_polarity.png',
            polarities, predicted, attrs, origin, best_mt)

        print('\nFinished\n')

