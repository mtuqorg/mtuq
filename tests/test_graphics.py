
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
    # Tests data, synthetics and beachball plotting utilities
    #
    # Note that in the figures created by this script, the data and synthetics 
    # are not expected to fit epsecially well; currently, the only requirement 
    # is that the script runs without errors
    #

    import matplotlib
    matplotlib.use('Agg', warn=False, force=True)
    import matplotlib

    path_greens=  fullpath('data/tests/benchmark_cap/greens/scak')
    path_data=    fullpath('data/examples/20090407201255351/*.[zrt]')
    path_weights= fullpath('data/examples/20090407201255351/weights.dat')
    event_id=     '20090407201255351'
    model=        'scak'


    process_bw = ProcessData(
        filter_type='Bandpass',
        freq_min= 0.1,
        freq_max= 0.333,
        pick_type='FK_metadata',
        FK_database=path_greens,
        window_type='body_wave',
        window_length=15.,
        capuaf_file=path_weights,
        )

    process_sw = ProcessData(
        filter_type='Bandpass',
        freq_min=0.025,
        freq_max=0.0625,
        pick_type='FK_metadata',
        FK_database=path_greens,
        window_type='surface_wave',
        window_length=150.,
        capuaf_file=path_weights,
        )


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


    mt = np.sqrt(1./3.)*np.array([1., 1., 1., 0., 0., 0.]) # explosion
    mt *= 1.e16

    wavelet = Trapezoid(
        magnitude=4.5)


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
    db = open_db(path_greens, format='FK', model=model)
    greens = db.get_greens_tensors(stations, origin)

    print('Processing Greens functions...\n')
    greens.convolve(wavelet)
    greens_bw = greens.map(process_bw)
    greens_sw = greens.map(process_sw)


    #
    # Generate figures
    #

    print('Figure 1 of 3\n')

    plot_data_greens2('graphics_test_1.png',
        data_bw, data_sw, greens_bw, greens_sw,
        process_bw, process_sw, misfit_bw, misfit_sw,
        stations, origin, mt, header=False)

    print('Figure 2 of 3\n')

    plot_data_greens2('graphics_test_2.png',
        data_bw, data_sw, greens_bw, greens_sw,
        process_bw, process_sw, misfit_bw, misfit_sw,
        stations, origin, mt, header=False)

    print('Figure 3 of 3\n')

    plot_beachball('graphics_test_3.png', 
        mt, None, None)

    print('\nFinished\n')
