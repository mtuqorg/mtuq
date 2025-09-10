
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
    # Grid search integration test
    #
    # This script is similar to examples/SerialGridSearch.DoubleCouple.py,
    # except here we use a coarser grid, and at the end we assert that the test
    # result equals the expected result
    #
    # The compare against CAP/FK:
    #
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.1/0.333/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.5 -I1/1/10/10/10 -R0/0/0/0/0/360/0/90/-180/180 20090407201255351
    #
    # Note however that CAP uses a different method for defining regular grids
    #


    # by default, the script runs with figure generation and error checking
    # turned on
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_checks', action='store_true')
    parser.add_argument('--no_figures', action='store_true')
    args = parser.parse_args()
    run_checks = (not args.no_checks)
    run_figures = (not args.no_figures)


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


    #
    # Next, we specify the moment tensor grid and source-time function
    #

    grid = DoubleCoupleGridRegular(
        npts_per_axis=5,
        magnitudes=[4.5])

    wavelet = Trapezoid(
        magnitude=4.5)


    station_id_list = parse_station_codes(path_weights)


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
    results_bw = grid_search(data_bw, greens_bw, misfit_bw, origin, grid, 0)

    print('Evaluating surface wave misfit...\n')
    results_sw = grid_search(data_sw, greens_sw, misfit_sw, origin, grid, 0)



    results = results_bw + results_sw

    # source corresponding to minimum misfit
    idx = results.source_idxmin()
    best_mt = grid.get(idx)
    lune_dict = grid.get_dict(idx)

    if run_figures:

        plot_data_greens2(event_id+'DC_waveforms.png',
            data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw, 
            misfit_bw, misfit_sw, stations, origin, best_mt, lune_dict)

        plot_beachball(event_id+'DC_beachball.png',
            best_mt, None, None)


    if run_checks:
        def isclose(a, b, atol=1.e6, rtol=1.e-6):
            # the default absolute tolerance (1.e6) is several orders of 
            # magnitude less than the moment of an Mw=0 event

            for _a, _b, _bool in zip(
                a, b, np.isclose(a, b, atol=atol, rtol=rtol)):

                print('%s:  %.e <= %.1e + %.1e * %.1e' %\
                    ('passed' if _bool else 'failed', abs(_a-_b), atol, rtol, abs(_b)))

            print('')

            return np.all(
                np.isclose(a, b, atol=atol, rtol=rtol))

        if not isclose(best_mt.as_vector(),
            np.array([
                 -6.731618e+15,
                  8.398708e+14,
                  5.891747e+15,
                 -1.318056e+15,
                  7.911756e+14,
                  2.718294e+15,
                 ])
            ):
            raise Exception(
                "Grid search result differs from previous mtuq result")

        print('SUCCESS\n')
