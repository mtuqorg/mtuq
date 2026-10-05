"""Optional DetailedAnalysis-style outputs for MTUQ workflow runs."""

from pathlib import Path

from mtuq.graphics import likelihood_analysis
from mtuq.util import save_json

from .plots import (
    _measurement_attributes,
    _measurement_variance,
    _raw_data_norm,
)


def save_requested_outputs(
    config,
    output_dir,
    term_results,
    data,
    greens,
    misfits,
    stations,
    catalog_origin,
    origin,
    source,
    solution,
    origins=None,
    cache=None,
):
    """Writes optional products requested through the recipe ``save`` list."""
    requested = config.get('save') or []
    if not requested:
        return

    if cache is None:
        cache = {
            'variance': {},
            'data_norm': {},
            'attributes': {},
        }

    output_dir = Path(output_dir)

    if 'stations' in requested:
        save_json(
            output_dir / 'stations.json',
            {station.id: station for station in stations},
        )

    if 'origins' in requested:
        origin_values = origins if origins is not None else [catalog_origin]
        save_json(
            output_dir / 'origins.json',
            {index: value for index, value in enumerate(origin_values)},
        )

    if 'attributes' in requested:
        _save_attributes(
            output_dir,
            config,
            data,
            greens,
            misfits,
            stations,
            origin,
            source,
            cache,
        )

    if 'statistics' in requested:
        _save_statistics(
            output_dir,
            config,
            data,
            greens,
            misfits,
            origin,
            source,
            cache,
        )

    if 'solutions' in requested:
        _save_solutions(
            output_dir,
            config,
            term_results,
            data,
            greens,
            misfits,
            origin,
            source,
            solution,
            cache,
        )

    if 'waveforms' in requested:
        _save_waveforms(
            output_dir,
            config,
            data,
            greens,
            origin,
            source,
        )


def _save_attributes(
    output_dir,
    config,
    data,
    greens,
    misfits,
    stations,
    origin,
    source,
    cache,
):
    directory = Path(output_dir) / 'attributes'
    directory.mkdir(parents=True, exist_ok=True)

    for name in config['measurements']:
        attrs = _measurement_attributes(
            name, data, greens, misfits, origin, source, cache
        )
        if len(attrs) != len(stations):
            raise ValueError(
                "measurement %r returned %d station-attribute entries for "
                "%d stations" % (name, len(attrs), len(stations))
            )

        save_json(
            directory / ('%s.json' % name),
            {
                station.id: attrs[index]
                for index, station in enumerate(stations)
            },
        )


def _save_statistics(
    output_dir,
    config,
    data,
    greens,
    misfits,
    origin,
    source,
    cache,
):
    directory = Path(output_dir) / 'statistics'
    directory.mkdir(parents=True, exist_ok=True)

    variances = {}
    norms = {}
    for name in config['measurements']:
        effective_variance = _measurement_variance(
            name, data, greens, misfits, origin, source, cache
        )
        data_norm = _raw_data_norm(name, data, misfits, cache)

        # MTUQ stores normalized misfit as J / ||d||.  The plotting likelihood
        # therefore uses sigma^2 / ||d||, while this file records the physical
        # data variance sigma^2 itself.
        if misfits[name].normalize:
            variances[name] = effective_variance * data_norm
        else:
            variances[name] = effective_variance
        norms[name] = data_norm

    save_json(directory / 'data_variance.json', variances)
    save_json(directory / 'data_norm.json', norms)


def _save_solutions(
    output_dir,
    config,
    term_results,
    data,
    greens,
    misfits,
    origin,
    source,
    solution,
    cache,
):
    directory = Path(output_dir) / 'solutions'
    directory.mkdir(parents=True, exist_ok=True)

    save_json(directory / 'minimum_misfit.json', solution)

    likelihood_terms = []
    for name in config['measurements']:
        variance = _measurement_variance(
            name, data, greens, misfits, origin, source, cache
        )
        likelihood_terms.append((term_results[name], variance))

    _, maximum_likelihood, marginal_likelihood = likelihood_analysis(
        *likelihood_terms
    )
    save_json(
        directory / 'maximum_likelihood.json', maximum_likelihood
    )
    save_json(
        directory / 'marginal_likelihood.json', marginal_likelihood
    )


def _save_waveforms(
    output_dir,
    config,
    data,
    greens,
    origin,
    source,
):
    directory = Path(output_dir) / 'waveforms'

    for name in config['measurements']:
        measurement_dir = directory / name
        data[name].write(measurement_dir / 'data')

        selected_greens = greens[name].select(origin)
        synthetics = selected_greens.get_synthetics(
            source,
            components=data[name].get_components(),
            mode='map',
        )
        synthetics.write(measurement_dir / 'synthetics')
