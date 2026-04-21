#!/usr/bin/env python

import os
import numpy as np

from mtuq import read, download_greens
from mtuq.event import MomentTensor, Origin
from mtuq.graphics import plot_data_greens2, plot_beachball
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath, merge_dicts, save_json
from mtuq.util.cap import parse_station_codes, Trapezoid
from mtuq.util.math import from_mij, to_mij, to_rho


if __name__=='__main__':
    #
    # Helper code for evaluating synthetic waveforms for a single fixed source.
    #
    # USAGE
    #   python DirectEvaluation.py
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
    # For our objective function, we will use a sum of body and surface wave
    # contributions
    #

    misfit_bw = Misfit(
        norm='L2',
        time_shift_min=-2.,
        time_shift_max=+2.,
        time_shift_groups=['ZR'],
        normalize=True,
        )

    misfit_sw = Misfit(
        norm='L2',
        time_shift_min=-10.,
        time_shift_max=+10.,
        time_shift_groups=['ZR','T'],
        normalize=True,
        )


    #
    # User-supplied weights control how much each station contributes to the
    # objective function
    #

    station_id_list = parse_station_codes(path_weights)


    #
    # Define the source directly from moment tensor components (Mrr, Mtt, Mpp, Mrt, Mrp, Mtp).
    # Alternatively, fault geometry can be converted using to_mij (see commented block below).
    #

    mt_dict = {
        'Mrr': -5800435174432632.0,
        'Mtt':   789342714264084.5,
        'Mpp':  5011092460168549.0,
        'Mrt':  3034156721905327.0,
        'Mrp':  2110077787951300.2,
        'Mtp':  2602039428905879.0,
        }

    mt_array = list(mt_dict.values())

    # To define the source from fault geometry instead:
    # Mw, strike, dip, rake = 4.5, 230., 40., -10.
    # mt_array = to_mij(to_rho(Mw), 0., 0., strike, rake, np.cos(np.deg2rad(dip)))

    lune_dict = dict(zip(('rho', 'v', 'w', 'kappa', 'sigma', 'h'), from_mij(mt_array)))

    mt = MomentTensor(mt_array)

    wavelet = Trapezoid(magnitude=4.5)


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
    greens = download_greens(stations, origin, model)

    print('Processing Greens functions...\n')
    greens.convolve(wavelet)
    greens_bw = greens.map(process_bw)
    greens_sw = greens.map(process_sw)


    #
    # Generating synthetics
    #

    syn_bw = greens_bw.get_synthetics(mt, components=['Z', 'R'])
    syn_sw = greens_sw.get_synthetics(mt, components=['Z', 'R', 'T'])

    syn_bw.write(event_id+'DC_synthetics_bw.sac', format='sac')
    syn_sw.write(event_id+'DC_synthetics_sw.sac', format='sac')


    print('Generating figures...\n')

    plot_data_greens2(event_id+'DC_waveforms.png',
        data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw,
        misfit_bw, misfit_sw, stations, origin, mt, lune_dict)

    plot_beachball(event_id+'DC_beachball.png',
        mt, stations, origin)


    print('Saving results...\n')

    mt_dict.update(lune_dict)
    mt_dict.update({'Mw': mt.magnitude(), 'M0': mt.moment()})
    save_json(event_id+'DC_solution.json', mt_dict)

    print('\nFinished\n')
