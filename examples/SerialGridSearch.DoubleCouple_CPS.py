#!/usr/bin/env python

import os
import numpy as np

from mtuq import read, open_db, download_greens_tensors
from mtuq.event import Origin
from mtuq.graphics import plot_data_greens2, plot_beachball, plot_misfit_dc
from mtuq.grid import DoubleCoupleGridRegular
from mtuq.grid_search import grid_search
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath, merge_dicts, save_json
from mtuq.util.cap import parse_station_codes, Trapezoid



if __name__=='__main__':
    #
    # Carries out grid search over 64,000 double couple moment tensors
    #
    # USAGE
    #   python SerialGridSearch.DoubleCouple_CPS.py
    #
    # A typical runtime is about 60 seconds. For faster results try 
    # GridSearch.DoubleCouple.py, which runs the same inversion in parallel
    #


    #
    # We will investigate the source process of an Mw~4 earthquake using data
    # from a regional seismic array
    #

    # path_data=    fullpath('data/examples/20090407201255351/*.[zrt]')
    # path_weights= fullpath('data/examples/20090407201255351/weights.dat')
    # path_greens=  fullpath('data/tests/benchmark_cps/greens/scak')

    path_greens=  '/home/dockimble/src/mtuq/data/tests/benchmark_cap/greens_cps/scak'
    path_data=    '/home/dockimble/src/mtuq/data/examples/20090407201255351/*.[zrt]'
    path_weights= '/home/dockimble/src/mtuq/data/examples/20090407201255351/weights.dat'
    

    path_data=    '/home/dockimble/Projects/mtuq/data/examples/20090407201255351/*.[zrt]' # fullpath('data/examples/20080418093700/*.[zrt]')
    path_weights= '/home/dockimble/Projects/mtuq/data/examples/20090407201255351/weights.dat' # fullpath('data/examples/20080418093700/weights.dat')
    path_greens=  '/home/dockimble/Projects/mtuq/data/tests/benchmark_cps/greens/scak' #fullpath('data/tests/benchmark_cps/greens/cus')    
      

    event_id=     '20090407201255351'
    model=        'scak'


    #
    # Body and surface wave measurements will be made separately
    #

    process_bw = ProcessData(
        filter_type='Bandpass',
        freq_min= 0.1,
        freq_max= 0.333,
        pick_type='CPS_metadata',
        CPS_database=path_greens,
        window_type='body_wave',
        window_length=15.,
        capuaf_file=path_weights,
        )

    process_sw = ProcessData(
        filter_type='Bandpass',
        freq_min=0.025,
        freq_max=0.0625,
        pick_type='CPS_metadata',
        CPS_database=path_greens,
        window_type='surface_wave',
        window_length=150.,
        capuaf_file=path_weights,
        )


    #
    # For our objective function, we will use a sum of body and surface wave
    # contributions
    #

    misfit_bw = Misfit(
        norm='L2',
        time_shift_min=-2.,
        time_shift_max=+2.,
        time_shift_groups=['ZR'],
        )

    misfit_sw = Misfit(
        norm='L2',
        time_shift_min=-10.,
        time_shift_max=+10.,
        time_shift_groups=['ZR','T'],
        )


    #
    # User-supplied weights control how much each station contributes to the
    # objective function
    #

    station_id_list = parse_station_codes(path_weights)


    #
    # Next, we specify the moment tensor grid and source-time function
    #

    grid = DoubleCoupleGridRegular(
        npts_per_axis=40,
        magnitudes=[4.5])

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


    #
    # The main I/O work starts now
    #

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
    db = open_db(path_greens, format='CPS', model=model)
    greens = db.get_greens_tensors(stations, origin)


    print('Processing Greens functions...\n')
    greens.convolve(wavelet)
    greens_bw = greens.map(process_bw)
    greens_sw = greens.map(process_sw)


    #
    # The main computational work starts now
    #

    print('Evaluating body wave misfit...\n')
    results_bw = grid_search(data_bw, greens_bw, misfit_bw, origin, grid)

    print('Evaluating surface wave misfit...\n')
    results_sw = grid_search(data_sw, greens_sw, misfit_sw, origin, grid)



    results = results_bw + results_sw

    #
    # Collect information about best-fitting source
    #

    # index of best-fitting moment tensor
    idx = results.source_idxmin()

    # MomentTensor object
    best_mt = grid.get(idx)

    # dictionary of lune parameters
    lune_dict = grid.get_dict(idx)

    # dictionary of Mij parameters
    mt_dict = best_mt.as_dict()

    merged_dict = merge_dicts(
        mt_dict, lune_dict, {'M0': best_mt.moment()},
        {'Mw': best_mt.magnitude()}, origin)

    #
    # Generate figures and save results
    #

    print('Generating figures...\n')

    plot_data_greens2(event_id+'DC_waveforms.png',
        data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw, 
        misfit_bw, misfit_sw, stations, origin, best_mt, lune_dict)


    plot_beachball(event_id+'DC_beachball.png',
        best_mt, stations, origin)


    plot_misfit_dc(event_id+'DC_misfit.png', results)


    print('Saving results...\n')

    # save best-fitting source
    save_json(event_id+'DC_solution.json', merged_dict)


    # save misfit surface
    results.save(event_id+'DC_misfit.nc')


    print('\nFinished\n')