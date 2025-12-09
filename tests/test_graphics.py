
import os
import numpy as np

from mtuq import read, open_db, download_greens
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

    #
    # The idea is for a test that runs very quickly, suitable for CI testing;
    # eventually we may more detailed tests to tests/graphics
    #

    import matplotlib
    matplotlib.use('Agg', force=True)
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
        normalize=True,
        )

    misfit_sw = Misfit(
        norm='L2',
        time_shift_min=-10.,
        time_shift_max=+10.,
        time_shift_groups=['ZR','T'],
        normalize=True,
        )


    origin = Origin({
        'time': '2009-04-07T20:12:55.000000Z',
        'latitude': 61.454200744628906,
        'longitude': -149.7427978515625,
        'depth_in_m': 33033.599853515625,
        })


    from mtuq import MomentTensor

    mt = MomentTensor(
        1.e16 * np.sqrt(1./3.)*np.array([1., 1., 1., 0., 0., 0.])) # explosion

    mt_dict = {
       'rho':1.,'v':0.,'w':3/8*np.pi,'kappa':0.,'sigma':0.,'h':0.}

    wavelet = Trapezoid(
        magnitude=4.5)


    print('Reading data...\n')
    data = read(path_data, format='sac',
        event_id=event_id,
        #station_id_list=station_id_list,
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

    print('Plot data (1 of 6)\n')

    from mtuq.graphics import plot_waveforms2
    from mtuq.util import Null

    plot_waveforms2('graphics_test_1.png',
        data_bw, data_sw, Null(), Null(),
        stations, origin, header=False)


    print('Plot synthetics (2 of 6)\n')

    synthetics_bw = greens_bw.get_synthetics(mt, components=['Z','R'])
    synthetics_sw = greens_sw.get_synthetics(mt, components=['Z','R','T'])


    plot_waveforms2('graphics_test_2.png',
        synthetics_bw, synthetics_sw, Null(), Null(),
        stations, origin, header=False)


    print('Plot synthetics (3 of 6)\n')

    synthetics_bw = misfit_bw.collect_synthetics(data_bw, greens_bw, mt)
    synthetics_sw = misfit_sw.collect_synthetics(data_sw, greens_sw, mt)


    plot_waveforms2('graphics_test_3.png',
        synthetics_bw, synthetics_sw, Null(), Null(),
        stations, origin, header=False)


    print('Plot data and synthetics without header (4 of 6)\n')

    plot_waveforms2('graphics_test_4.png',
        data_bw, data_sw, synthetics_bw, synthetics_sw,
        stations, origin, header=None)


    print('Plot data and synthetics with header (5 of 6)\n')

    plot_data_greens2('graphics_test_5.png',
        data_bw, data_sw, greens_bw, greens_sw,
        process_bw, process_sw, misfit_bw, misfit_sw,
        stations, origin, mt, mt_dict)


    print('Plot explosion bechball (6 of 6)\n')

    plot_beachball('graphics_test_6.png',
        mt, None, None)

    print('\nFinished\n')
