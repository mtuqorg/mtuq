"""Execution and command-line interface for one MTUQ workflow recipe.

Execution deliberately remains a straight-line MTUQ program: normalize the
recipe, construct native objects, read and process observations and Green's
functions, evaluate each measurement with ``grid_search()``, combine the
result surfaces, select the best source, and write native MTUQ outputs.
"""

import argparse
import os
from pathlib import Path

from mtuq import read
from mtuq.grid_search import grid_search
from mtuq.util import merge_dicts, save_json
from mtuq.util.cap import parse_station_codes

from .build import prepare_workflow
from .config import validate_config
from .io import (
    _open_greens_database,
    _print_grid_summary,
    _resolve_path,
    _save_native_results,
    _write_input_config,
    _write_yaml,
)
from .plots import generate_plots


def run(config, output=None):
    """ Runs one MTUQ inversion from a YAML recipe

    .. rubric :: Input arguments

    ``config`` (`str` or `os.PathLike`)
        Path to an abridged or complete YAML recipe.

    ``output`` (`str` or `os.PathLike`)
        Optional output-directory override.

    .. rubric :: Returns

    Dictionary containing the resolved configuration and MTUQ objects/results
    on rank 0. Under MPI, other ranks return ``None``.
    """
    # Normalized configuration used to construct MTUQ objects
    normalized = validate_config(config)

    if output is not None:
        if not isinstance(output, (str, os.PathLike)):
            raise TypeError('output must be a path string')
        normalized['output'] = _resolve_path(output, Path.cwd())

    #
    # Set up source grid, processing, and misfit
    #
    comm = _mpi_comm()
    rank = 0 if comm is None else comm.rank

    prepared = prepare_workflow(normalized)
    catalog_origin = prepared.catalog_origin
    origins = prepared.origins
    origin_arg = origins if origins is not None else catalog_origin
    grid = prepared.grid
    wavelet = prepared.wavelet
    processors = prepared.processors
    misfits = prepared.misfits
    resolved = prepared.resolved

    output_dir = Path(normalized['output'])

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_input_config(config, output_dir)
        _write_yaml(output_dir / 'config.resolved.yaml', resolved)
        _print_grid_summary(resolved)

    #
    # Read observations and Green's functions, then process each measurement
    #
    processed_data = {name: None for name in normalized['measurements']}
    processed_greens = {name: None for name in normalized['measurements']}

    if rank == 0:
        print('Reading data...\n')
        data_cfg = normalized['data']
        station_id_list = parse_station_codes(data_cfg['weights'])
        data = read(
            data_cfg['path'],
            format=data_cfg['format'],
            event_id=normalized['event']['id'],
            station_id_list=station_id_list,
            tags=data_cfg['tags'],
        )
        data.sort_by_distance()
        stations = data.get_stations()

        print('Reading Greens functions...\n')
        db = _open_greens_database(normalized['greens'])
        greens = db.get_greens_tensors(stations, origin_arg)

        print('Convolving Greens functions...\n')
        greens.convolve(wavelet)

        for name in normalized['measurements']:
            print('Processing measurement %r...\n' % name)
            processor = processors[name]
            processed_data[name] = data.map(processor)
            processed_greens[name] = greens.map(processor)

    if comm is not None:
        for name in normalized['measurements']:
            processed_data[name] = comm.bcast(processed_data[name], root=0)
            processed_greens[name] = comm.bcast(
                processed_greens[name], root=0
            )

    #
    # Evaluate each misfit term
    #
    term_results = {}
    for name in normalized['measurements']:
        if rank == 0:
            print('Evaluating measurement %r...\n' % name)

        term_results[name] = grid_search(
            processed_data[name],
            processed_greens[name],
            misfits[name],
            origin_arg,
            grid,
        )

    if rank != 0:
        return None

    #
    # Combine misfit terms and find the best-fitting source
    #
    if 'objective' in normalized:
        coefficients = normalized['objective']['coefficients']
    else:
        coefficients = {
            name: 1.0 for name in normalized['measurements']
        }

    total_results = None
    for name in normalized['measurements']:
        coefficient = coefficients[name]
        weighted = term_results[name]
        if coefficient != 1.0:
            weighted = weighted * coefficient
        if total_results is None:
            total_results = weighted
        else:
            total_results = total_results + weighted

    source_idx = total_results.source_idxmin()
    best_source = grid.get(source_idx)
    source_coordinates = grid.get_dict(source_idx)

    if origins is None:
        best_origin = catalog_origin
    else:
        origin_idx = total_results.origin_idxmin()
        best_origin = origins[origin_idx]

    solution = merge_dicts(
        best_source.as_dict(),
        source_coordinates,
        {'M0': best_source.moment()},
        {'Mw': best_source.magnitude()},
        best_origin,
    )
    if origins is not None:
        solution['reference_origin'] = catalog_origin.as_dict().copy()

    #
    # Save results
    #
    print('Saving results...\n')
    _save_native_results(output_dir, total_results, term_results)
    if origins is not None:
        save_json(
            output_dir / 'origins.json',
            {index: origin for index, origin in enumerate(origins)},
        )
    save_json(output_dir / 'solution.json', solution)

    if normalized.get('plots'):
        print('Generating plots...\n')
        try:
            generate_plots(
                normalized,
                output_dir=output_dir,
                results=total_results,
                data=processed_data,
                greens=processed_greens,
                processors=processors,
                misfits=misfits,
                stations=stations,
                origin=best_origin,
                source=best_source,
                source_dict=source_coordinates,
                origins=origins,
            )
        except Exception as exc:
            detail = str(exc).strip() or type(exc).__name__
            print('  Plot generation failed: %s' % detail)

    result = {
        'config': resolved,
        'origin': best_origin,
        'reference_origin': catalog_origin,
        'grid': grid,
        'source': best_source,
        'results': total_results,
        'terms': term_results,
        'data': processed_data,
        'greens': processed_greens,
        'misfits': misfits,
        'processors': processors,
    }
    if origins is not None:
        result['origins'] = origins
    return result


def main(argv=None):
    """ Command-line entry point
    """
    parser = argparse.ArgumentParser(
        prog='mtuq-run',
        description='Run one MTUQ inversion from a YAML recipe.',
    )
    parser.add_argument('config', help='workflow YAML file')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='validate configuration and construct native MTUQ objects only',
    )
    parser.add_argument(
        '--output',
        help='override output directory',
    )
    args = parser.parse_args(argv)

    if args.validate:
        normalized = validate_config(args.config)
        if args.output is not None:
            normalized['output'] = _resolve_path(args.output, Path.cwd())
        prepared = prepare_workflow(normalized)
        origin = prepared.catalog_origin
        resolved = prepared.resolved

        print('Configuration is valid.\n')
        _print_grid_summary(resolved)
        print('Wavelet')
        wavelet = resolved['wavelet']
        print('  function: %s' % wavelet['function'])
        for key, value in wavelet.items():
            if key in {'type', 'function'}:
                continue
            print('  %s: %s' % (key, value))
        print('\nOrigin')
        print('  time: %s' % origin.time)
        print('  latitude: %g' % origin.latitude)
        print('  longitude: %g' % origin.longitude)
        print('  depth_in_m: %g' % origin.depth_in_m)
        if prepared.origins is not None:
            print('\nOrigin search')
            print('  number of origins: %d' % len(prepared.origins))
        return 0

    run(args.config, output=args.output)
    return 0


def _mpi_comm():
    try:
        from mpi4py import MPI
    except ImportError:
        return None
    if MPI.COMM_WORLD.Get_size() <= 1:
        return None
    return MPI.COMM_WORLD


if __name__ == '__main__':
    raise SystemExit(main())
