"""Standard plots for completed MTUQ workflow inversions."""

from pathlib import Path

import numpy as np

from mtuq.graphics import (
    _likelihoods_vw_regular,
    _marginals_vw_regular,
    _plot_lune,
    _plot_vw,
    _product_vw,
    plot_amplitude_ratios,
    plot_beachball,
    plot_confidence_curve,
    plot_data_greens1,
    plot_data_greens2,
    plot_data_greens3,
    plot_likelihood_lune,
    plot_magnitude_tradeoffs_lune,
    plot_marginal_vw,
    plot_misfit_dc,
    plot_misfit_depth,
    plot_misfit_force,
    plot_misfit_latlon,
    plot_misfit_lune,
    plot_time_shifts,
    plot_variance_reduction_lune,
)
from mtuq.misfit.waveform import calculate_norm_data, estimate_sigma


_SOURCE_MISFIT_PLOTS = {
    'dc': plot_misfit_dc,
    'dev': plot_misfit_lune,
    'fmt': plot_misfit_lune,
    'force': plot_misfit_force,
}


_DETAILED_PLOTS = {
    'likelihood',
    'marginal',
    'variance_reduction',
    'orientation_tradeoffs',
    'magnitude_tradeoffs',
    'time_shifts',
    'amplitude_ratios',
}


def generate_plots(
    config,
    output_dir,
    results,
    data,
    greens,
    processors,
    misfits,
    stations,
    origin,
    source,
    source_dict,
    origins=None,
    term_results=None,
    origin_idx=0,
    cache=None,
):
    """Generates requested MTUQ plots after results are saved."""
    plot_dir = Path(output_dir) / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    event_id = config['event']['id']
    if cache is None:
        cache = {
            'variance': {},
            'data_norm': {},
            'attributes': {},
        }

    for name in config['plots']:
        if name == 'waveform':
            _plot_waveform(
                plot_dir,
                event_id,
                config['measurements'],
                data,
                greens,
                processors,
                misfits,
                stations,
                origin,
                source,
                source_dict,
            )
        elif name == 'beachball':
            _call_plot(
                'beachball',
                plot_beachball,
                _plot_filename(plot_dir, event_id, 'beachball'),
                source,
                stations,
                origin,
            )
        elif name == 'misfit':
            _plot_misfit(
                plot_dir,
                event_id,
                config,
                results,
                origins,
            )
        elif name == 'confidence':
            try:
                _plot_confidence(
                    plot_dir,
                    event_id,
                    config,
                    term_results,
                    data,
                    greens,
                    misfits,
                    origin,
                    origin_idx,
                    source,
                    source_dict,
                    cache,
                )
            except Exception as exc:
                detail = str(exc).strip() or type(exc).__name__
                print("  Plot 'confidence' failed: %s" % detail)
        elif name in _DETAILED_PLOTS:
            try:
                _plot_detailed(
                    name,
                    plot_dir,
                    config,
                    results,
                    term_results,
                    data,
                    greens,
                    misfits,
                    stations,
                    origin,
                    origin_idx,
                    source,
                    cache,
                )
            except Exception as exc:
                detail = str(exc).strip() or type(exc).__name__
                print("  Plot %r failed: %s" % (name, detail))


def _plot_waveform(
    plot_dir,
    event_id,
    measurements,
    data,
    greens,
    processors,
    misfits,
    stations,
    origin,
    source,
    source_dict,
):
    names = list(measurements)

    if len(names) == 1:
        name = names[0]
        _plot_waveform_term(
            plot_dir, event_id, 'waveform', name,
            data, greens, processors, misfits,
            stations, origin, source, source_dict,
        )
        return

    roles = {
        measurement['role']: name
        for name, measurement in measurements.items()
        if 'role' in measurement
    }

    surface_role = next(
        (
            role
            for role in ('surface', 'rayleigh', 'love')
            if role in roles
        ),
        None,
    )
    if (
        len(names) == 2
        and surface_role is not None
        and set(roles) == {'body', surface_role}
    ):
        body = roles['body']
        surface = roles[surface_role]
        _call_plot(
            'waveform',
            plot_data_greens2,
            _plot_filename(plot_dir, event_id, 'waveform'),
            data[body],
            data[surface],
            greens[body],
            greens[surface],
            processors[body],
            processors[surface],
            misfits[body],
            misfits[surface],
            stations,
            origin,
            source,
            source_dict,
        )
        return

    if len(names) == 3 and set(roles) == {'body', 'rayleigh', 'love'}:
        body = roles['body']
        rayleigh = roles['rayleigh']
        love = roles['love']
        _call_plot(
            'waveform',
            plot_data_greens3,
            _plot_filename(plot_dir, event_id, 'waveform'),
            data[body],
            data[rayleigh],
            data[love],
            greens[body],
            greens[rayleigh],
            greens[love],
            processors[body],
            processors[rayleigh],
            processors[love],
            misfits[body],
            misfits[rayleigh],
            misfits[love],
            stations,
            origin,
            source,
            source_dict,
        )
        return

    for name in names:
        _plot_waveform_term(
            plot_dir, event_id, 'waveform_%s' % name, name,
            data, greens, processors, misfits,
            stations, origin, source, source_dict,
        )


def _plot_waveform_term(
    plot_dir,
    event_id,
    plot_type,
    name,
    data,
    greens,
    processors,
    misfits,
    stations,
    origin,
    source,
    source_dict,
):
    _call_plot(
        plot_type,
        plot_data_greens1,
        _plot_filename(plot_dir, event_id, plot_type),
        data[name],
        greens[name],
        processors[name],
        misfits[name],
        stations,
        origin,
        source,
        source_dict,
    )


def _plot_misfit(plot_dir, event_id, config, results, origins):
    source_type = config['source']['type']
    _call_plot(
        'misfit',
        _SOURCE_MISFIT_PLOTS[source_type],
        _plot_filename(plot_dir, event_id, 'misfit'),
        results,
    )
    if source_type in {'dev', 'fmt'}:
        _call_plot(
            'misfit_dc',
            plot_misfit_dc,
            _plot_filename(plot_dir, event_id, 'misfit_dc'),
            results,
        )

    search = config.get('origin_search')
    if search is None:
        return

    if source_type == 'force':
        _print_skip(
            'origin misfit',
            'native origin-search plots are moment-tensor-specific',
        )
        return

    if config['source']['grid']['type'] != 'regular':
        _print_skip(
            'origin misfit',
            'native origin-search plots require a regular source grid',
        )
        return

    depths = {origin.depth_in_m for origin in origins}
    locations = {
        (origin.latitude, origin.longitude) for origin in origins
    }
    varies_in_depth = len(depths) > 1
    varies_horizontally = len(locations) > 1

    if varies_in_depth and varies_horizontally:
        _print_skip(
            'origin misfit',
            'combined depth and hypocenter searches have no native 2-D plot',
        )
        return

    if varies_in_depth:
        _call_plot(
            'misfit_depth',
            plot_misfit_depth,
            _plot_filename(plot_dir, event_id, 'misfit_depth'),
            results,
            origins,
            show_tradeoffs=True,
            show_magnitudes=True,
            title=event_id,
        )
    elif varies_horizontally:
        _call_plot(
            'misfit_latlon',
            plot_misfit_latlon,
            _plot_filename(plot_dir, event_id, 'misfit_latlon'),
            results,
            origins,
            show_tradeoffs=True,
        )
    else:
        _print_skip(
            'origin misfit',
            'searched origins do not vary in depth or hypocenter',
        )


def _plot_detailed(
    name,
    plot_dir,
    config,
    results,
    term_results,
    data,
    greens,
    misfits,
    stations,
    origin,
    origin_idx,
    source,
    cache,
):
    """Generates DetailedAnalysis-style products requested by plot name."""
    if term_results is None:
        raise ValueError('measurement result surfaces are unavailable')

    if name == 'likelihood':
        _plot_likelihood(
            plot_dir, config, term_results, data, greens,
            misfits, origin, origin_idx, source, cache,
        )
    elif name == 'marginal':
        _plot_marginal(
            plot_dir, config, term_results, data, greens,
            misfits, origin, origin_idx, source, cache,
        )
    elif name == 'variance_reduction':
        _plot_variance_reduction(
            plot_dir, config, term_results, data, misfits,
            origin_idx, cache,
        )
    elif name == 'orientation_tradeoffs':
        conditioned = _condition_regular_results(results, origin_idx)
        tradeoff_dir = Path(plot_dir) / 'tradeoffs'
        tradeoff_dir.mkdir(parents=True, exist_ok=True)
        _call_plot(
            'orientation_tradeoffs',
            plot_misfit_lune,
            tradeoff_dir / 'orientation.png',
            conditioned,
            show_tradeoffs=True,
            title='Orientation tradeoffs',
        )
    elif name == 'magnitude_tradeoffs':
        conditioned = _condition_regular_results(results, origin_idx)
        tradeoff_dir = Path(plot_dir) / 'tradeoffs'
        tradeoff_dir.mkdir(parents=True, exist_ok=True)
        _call_plot(
            'magnitude_tradeoffs',
            plot_magnitude_tradeoffs_lune,
            tradeoff_dir / 'magnitude.png',
            conditioned,
            title='Magnitude tradeoffs',
            colorbar_label='Mw',
        )
    elif name == 'time_shifts':
        _plot_trace_attributes(
            plot_dir,
            'time_shifts',
            plot_time_shifts,
            config,
            data,
            greens,
            misfits,
            stations,
            origin,
            source,
            cache,
        )
    elif name == 'amplitude_ratios':
        _plot_trace_attributes(
            plot_dir,
            'amplitude_ratios',
            plot_amplitude_ratios,
            config,
            data,
            greens,
            misfits,
            stations,
            origin,
            source,
            cache,
        )
    else:
        raise ValueError('unsupported detailed plot %r' % name)


def _plot_likelihood(
    plot_dir,
    config,
    term_results,
    data,
    greens,
    misfits,
    origin,
    origin_idx,
    source,
    cache,
):
    output = Path(plot_dir) / 'likelihood'
    output.mkdir(parents=True, exist_ok=True)
    surfaces = []

    for name in config['measurements']:
        results = _condition_regular_results(term_results[name], origin_idx)
        variance = _measurement_variance(
            name, data, greens, misfits, origin, source, cache
        )
        title = _measurement_title(name)
        _call_plot(
            'likelihood_%s' % name,
            plot_likelihood_lune,
            output / ('%s.png' % name),
            results,
            var=variance,
            title=title,
        )
        surfaces.append(_likelihoods_vw_regular(results, variance))

    combined = _product_vw(*surfaces)
    _call_plot(
        'likelihood_total',
        _plot_lune,
        output / 'total.png',
        combined,
        colormap='hot_r',
        title='All data categories',
    )


def _plot_marginal(
    plot_dir,
    config,
    term_results,
    data,
    greens,
    misfits,
    origin,
    origin_idx,
    source,
    cache,
):
    output = Path(plot_dir) / 'marginal'
    output.mkdir(parents=True, exist_ok=True)
    surfaces = []

    for name in config['measurements']:
        results = _condition_regular_results(term_results[name], origin_idx)
        variance = _measurement_variance(
            name, data, greens, misfits, origin, source, cache
        )
        title = _measurement_title(name)
        _call_plot(
            'marginal_%s' % name,
            plot_marginal_vw,
            output / ('%s.png' % name),
            results,
            var=variance,
            title=title,
        )
        surfaces.append(_marginals_vw_regular(results, variance))

    # Mirrors DetailedAnalysis.py. The upstream example notes that the joint
    # likelihood should eventually be marginalized instead of multiplying
    # the already-marginalized measurement surfaces.
    combined = _product_vw(*surfaces)
    _call_plot(
        'marginal_total',
        _plot_vw,
        output / 'total.png',
        combined,
        colormap='hot_r',
        title='All data categories',
    )


def _plot_variance_reduction(
    plot_dir,
    config,
    term_results,
    data,
    misfits,
    origin_idx,
    cache,
):
    output = Path(plot_dir) / 'variance_reduction'
    output.mkdir(parents=True, exist_ok=True)

    for name in config['measurements']:
        results = _condition_regular_results(term_results[name], origin_idx)
        data_norm = _result_data_norm(name, data, misfits, cache)
        _call_plot(
            'variance_reduction_%s' % name,
            plot_variance_reduction_lune,
            output / ('%s.png' % name),
            results,
            data_norm,
            title=_measurement_title(name),
        )


def _plot_trace_attributes(
    plot_dir,
    plot_name,
    plotter,
    config,
    data,
    greens,
    misfits,
    stations,
    origin,
    source,
    cache,
):
    output = Path(plot_dir) / plot_name
    output.mkdir(parents=True, exist_ok=True)

    for name in config['measurements']:
        attrs = _measurement_attributes(
            name, data, greens, misfits, origin, source, cache
        )
        _call_plot(
            '%s_%s' % (plot_name, name),
            plotter,
            output / name,
            attrs,
            stations,
            origin,
        )


def _measurement_variance(
    name,
    data,
    greens,
    misfits,
    origin,
    source,
    cache,
):
    if name in cache['variance']:
        return cache['variance'][name]

    misfit = misfits[name]
    components = _measurement_components(misfit)
    selected_greens = greens[name].select(origin)
    sigma = estimate_sigma(
        data[name],
        selected_greens,
        source,
        misfit.norm,
        components,
        misfit.time_shift_min,
        misfit.time_shift_max,
    )
    variance = float(sigma)**2

    if misfit.normalize:
        variance /= _raw_data_norm(name, data, misfits, cache)

    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError(
            "measurement %r has invalid estimated variance %r"
            % (name, variance)
        )

    cache['variance'][name] = variance
    return variance


def _result_data_norm(name, data, misfits, cache):
    if misfits[name].normalize:
        return 1.0
    return _raw_data_norm(name, data, misfits, cache)


def _raw_data_norm(name, data, misfits, cache):
    if name in cache['data_norm']:
        return cache['data_norm'][name]

    misfit = misfits[name]
    components = _measurement_components(misfit)
    data_norm = calculate_norm_data(data[name], misfit.norm, components)
    if not np.isfinite(data_norm) or data_norm <= 0.0:
        raise ValueError(
            "measurement %r has invalid data norm %r" % (name, data_norm)
        )

    cache['data_norm'][name] = data_norm
    return data_norm


def _measurement_attributes(
    name, data, greens, misfits, origin, source, cache
):
    if name in cache['attributes']:
        return cache['attributes'][name]

    selected_greens = greens[name].select(origin)
    attrs = misfits[name].collect_attributes(
        data[name], selected_greens, source
    )
    cache['attributes'][name] = attrs
    return attrs


def _condition_regular_results(results, origin_idx):
    if 'origin_idx' not in results.dims:
        raise ValueError("result surface is missing 'origin_idx'")
    if origin_idx < 0 or origin_idx >= results.sizes['origin_idx']:
        raise ValueError(
            'origin_idx %r is outside result surface' % origin_idx
        )
    return results.isel(origin_idx=[origin_idx])


def _measurement_title(name):
    return str(name).replace('_', ' ').replace('-', ' ').title()


def _plot_confidence(
    plot_dir,
    event_id,
    config,
    term_results,
    data,
    greens,
    misfits,
    origin,
    origin_idx,
    source,
    source_dict,
    cache=None,
):
    """Plots one confidence curve from the joint measurement likelihood.

    Each measurement surface is conditioned on the reported origin and scalar
    moment, then divided by its own public MTUQ a posteriori variance estimate.
    Summing those dimensionless surfaces is equivalent to multiplying the
    independent per-measurement likelihoods.
    """
    if term_results is None:
        raise ValueError('measurement result surfaces are unavailable')

    if cache is None:
        cache = {
            'variance': {},
            'data_norm': {},
            'attributes': {},
        }

    rho = source_dict['rho']
    combined = None

    for name in config['measurements']:
        variance = _measurement_variance(
            name, data, greens, misfits, origin, source, cache
        )
        conditioned = _condition_confidence_results(
            term_results[name], origin_idx, rho
        )
        scaled = conditioned / variance
        combined = scaled if combined is None else combined + scaled

    if combined is None:
        raise ValueError('confidence requires at least one measurement')

    _call_plot(
        'confidence',
        plot_confidence_curve,
        _plot_filename(plot_dir, event_id, 'confidence'),
        combined,
        1.0,
        m0=source,
        normalized=False,
    )

def _measurement_components(misfit):
    components = []
    for group in misfit.time_shift_groups:
        for component in group:
            if component not in components:
                components.append(component)
    return components


def _condition_confidence_results(results, origin_idx, rho):
    names = set(results.index.names)
    missing = {'origin_idx', 'rho'} - names
    if missing:
        raise ValueError(
            'confidence result index is missing %s'
            % ', '.join(sorted(missing))
        )

    origins = results.index.get_level_values('origin_idx')
    magnitudes = results.index.get_level_values('rho')
    mask = (origins == origin_idx) & np.isclose(
        magnitudes, rho, rtol=1.e-10, atol=0.0
    )
    conditioned = results.loc[mask].copy()
    if conditioned.empty:
        raise ValueError(
            'no confidence samples match origin_idx=%r and rho=%r'
            % (origin_idx, rho)
        )
    return conditioned


def _plot_filename(plot_dir, event_id, plot_type):
    return Path(plot_dir) / ('%s_%s.png' % (event_id, plot_type))


def _call_plot(label, function, filename, *args, **kwargs):
    try:
        function(str(filename), *args, **kwargs)
    except Exception as exc:
        detail = str(exc).strip() or type(exc).__name__
        print("  Plot %r failed: %s" % (label, detail))


def _print_skip(label, reason):
    print("  Skipping plot %r: %s" % (label, reason))
