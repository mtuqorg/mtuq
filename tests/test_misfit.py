
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
    # Checks the correctness of the fast (optimized) misfit function
    # implementations against a simple pure Python implementation.
    # These implementations correspond to:
    #
    #   level=0: simple pure Python
    #   level=1: fast pure Python
    #   level=2: fast Python/C
    #
    # In running the test in our environment, we observe that the two pure 
    # Python implementations agree almost exactly.  On the other hand, the
    # pure Python and Python/C results differ by as much as 0.1 percent, 
    # presumably as a result of differences in the way that floating-point
    # error accumulates in the sum over residuals. Further work is required to 
    # understand this better
    #
    # Possibly relevant is the fact that C extensions are compiled with
    # `-Ofast` flag, as specified in `setup.py`.
    #
    # Note that the `level` keyword argument does not correspond
    # at all to C compiler optimization flags.  For example, the NumPy binaries
    # called by the simple pure Python misfit function are probably compiled 
    # using a nonzero optimization level?
    #



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


    #
    # User-supplied weights control how much each station contributes to the
    # objective function
    #

    station_id_list = parse_station_codes(path_weights)


    #
    # Next, we specify the moment tensor grid and source-time function
    #

    grid = DoubleCoupleGridRegular(
        npts_per_axis=5,
        magnitudes=[4.5])

    wavelet = Trapezoid(
        magnitude=4.5)


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
    db = open_db(path_greens, format='FK', model=model)
    greens = db.get_greens_tensors(stations, origin)


    print('Processing Greens functions...\n')
    greens.convolve(wavelet)
    greens_bw = greens.map(process_bw)
    greens_sw = greens.map(process_sw)


    #
    # The main computational work starts now
    #

    print('Evaluating body wave misfit...\n')

    results_0 = misfit_bw(
        data_bw, greens_bw, grid, level=0)

    results_1 = misfit_bw(
        data_bw, greens_bw, grid, level=1)

    results_2 = misfit_bw(
        data_bw, greens_bw, grid, level=2)

    print('  optimization level:  0\n', 
          '  argmin:  %d\n' % results_0.argmin(), 
          '  min:     %e\n\n' % results_0.min())

    print('  optimization level:  1\n', 
          '  argmin:  %d\n' % results_1.argmin(), 
          '  min:     %e\n\n' % results_1.min())

    print('  optimization level:  2\n', 
          '  argmin:  %d\n' % results_2.argmin(), 
          '  min:     %e\n\n' % results_2.min())

    print('')

    assert results_0.argmin()==results_1.argmin()==results_2.argmin()


    print('Evaluating surface wave misfit...\n')

    results_0 = misfit_sw(
        data_sw, greens_sw, grid, level=0)

    results_1 = misfit_sw(
        data_sw, greens_sw, grid, level=1)

    results_2 = misfit_sw(
        data_sw, greens_sw, grid, level=2)

    print('  optimization level:  0\n', 
          '  argmin:  %d\n' % results_0.argmin(), 
          '  min:     %e\n\n' % results_0.min())

    print('  optimization level:  1\n', 
          '  argmin:  %d\n' % results_1.argmin(), 
          '  min:     %e\n\n' % results_1.min())

    print('  optimization level:  2\n', 
          '  argmin:  %d\n' % results_2.argmin(), 
          '  min:     %e\n\n' % results_2.min())

    assert results_0.argmin()==results_1.argmin()==results_2.argmin()


