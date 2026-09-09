import copy
import importlib
import inspect
import json
import textwrap
from unittest.mock import Mock

import numpy as np
import pytest
import yaml


from mtuq.grid import (
    DeviatoricGridRandom,
    DeviatoricGridSemiregular,
    DoubleCoupleGridRandom,
    DoubleCoupleGridRegular,
    FullMomentTensorGridRandom,
    FullMomentTensorGridSemiregular,
)
from mtuq.workflow import WorkflowConfigError, run, validate_config
from mtuq.workflow.build import (
    _build_grid,
    _build_measurements,
    _build_wavelet,
    prepare_workflow,
)
from mtuq.workflow.io import _write_input_config


def _weights_file(tmp_path):
    path = tmp_path / 'weights.dat'
    path.write_text(
        '0.AK.TEST.00 0 1 1 1 1 1 0 0 0 0 0 0\n',
        encoding='utf-8',
    )
    return str(path)


def _write_config(tmp_path, config, name='event.yaml'):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    return path


def _validate(tmp_path, config, name='event.yaml'):
    return validate_config(_write_config(tmp_path, config, name))


def _config(tmp_path, source_type='dc', grid_type='regular'):
    grid = (
        {'type': 'regular', 'npts_per_axis': 2}
        if grid_type == 'regular'
        else {'type': 'random', 'npts': 5}
    )
    return {
        'version': 1,
        'event': {
            'id': 'test',
            'time': '2009-04-07T20:12:55Z',
            'latitude': 61.4542,
            'longitude': -149.7428,
            'depth_in_m': 33033.6,
        },
        'data': {
            'format': 'SAC',
            'path': 'unused/*.[zrt]',
            'weights': _weights_file(tmp_path),
            'units': 'm/s',
        },
        'greens': {
            'format': 'SYNGINE',
            'model': 'ak135',
        },
        'picks': {
            'type': 'taup',
            'model': 'ak135',
        },
        'misfit': {
            'norm': 'L2',
            'normalize': True,
        },
        'measurements': {
            'body': {
                'filter': [0.1, 0.333],
                'window': ['body_wave', 15],
                'components': ['ZR'],
                'time_shift': 2,
            },
            'surface': {
                'filter': [0.025, 0.0625],
                'window': ['surface_wave', 150],
                'components': ['ZR', 'T'],
                'time_shift': 10,
            },
        },
        'source': {
            'type': source_type,
            'magnitudes': [4.4, 4.5, 4.6],
            'grid': grid,
        },
        'wavelet': {'type': 'trapezoid'},
        'output': str(tmp_path / 'output'),
    }


def _resolved_config(tmp_path):
    config = _validate(tmp_path, _config(tmp_path))
    return prepare_workflow(config).resolved


def test_duplicate_yaml_keys_are_rejected(tmp_path):
    path = tmp_path / 'duplicate.yaml'
    path.write_text(
        textwrap.dedent(
            '''
            version: 1
            version: 1
            '''
        ),
        encoding='utf-8',
    )

    with pytest.raises(WorkflowConfigError, match='duplicate key'):
        validate_config(path)


def test_missing_recipe_has_clear_error(tmp_path):
    path = tmp_path / 'missing.yaml'

    with pytest.raises(
        WorkflowConfigError, match='could not read configuration'
    ):
        validate_config(path)


def test_yaml_inputs_are_recipe_relative_and_output_is_cwd_relative(
    tmp_path, monkeypatch
):
    recipe_dir = tmp_path / 'recipes'
    data_dir = tmp_path / 'data'
    run_dir = tmp_path / 'run'
    recipe_dir.mkdir()
    data_dir.mkdir()
    run_dir.mkdir()
    (data_dir / 'weights.dat').write_text(
        '0.AK.TEST.00 0 1 1 1 1 1 0 0 0 0 0 0\n',
        encoding='utf-8',
    )

    config = _config(tmp_path)
    config['data']['path'] = '../data/*.[zrt]'
    config['data']['weights'] = '../data/weights.dat'
    config['greens']['cache_path'] = '../cache'
    config['output'] = 'results'

    path = recipe_dir / 'event.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')

    monkeypatch.chdir(run_dir)
    normalized = validate_config(path)

    assert normalized['data']['path'] == str(
        (data_dir / '*.[zrt]').resolve()
    )
    assert normalized['data']['weights'] == str(
        (data_dir / 'weights.dat').resolve()
    )
    assert normalized['greens']['cache_path'] == str(
        (tmp_path / 'cache').resolve()
    )
    assert normalized['output'] == str((run_dir / 'results').resolve())

    config.pop('output')
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    normalized = validate_config(path)
    assert normalized['output'] == str(
        (run_dir / 'output' / 'test').resolve()
    )


def test_python_mapping_input_is_rejected(tmp_path):
    config = _config(tmp_path)

    with pytest.raises(TypeError, match='YAML recipe path'):
        validate_config(config)
    with pytest.raises(TypeError, match='YAML recipe path'):
        run(config)


def test_run_output_override_requires_path(tmp_path):
    path = _write_config(tmp_path, _config(tmp_path))

    with pytest.raises(TypeError, match='output must be a path string'):
        run(path, output=42)


@pytest.mark.parametrize('version', [True, 1.0, '1'])
def test_version_requires_integer_yaml_value(tmp_path, version):
    config = _config(tmp_path)
    config['version'] = version

    with pytest.raises(
        WorkflowConfigError, match='version must be integer 1'
    ):
        _validate(tmp_path, config)


def test_missing_weights_file_has_clear_error(tmp_path):
    config = _config(tmp_path)
    config['data']['weights'] = str(tmp_path / 'missing.dat')

    with pytest.raises(
        WorkflowConfigError, match='data.weights does not exist'
    ):
        _validate(tmp_path, config)


@pytest.mark.parametrize('tags', [5, [], ['units:m', 1]])
def test_data_tags_requires_non_empty_list_of_strings(tmp_path, tags):
    config = _config(tmp_path)
    config['data']['tags'] = tags

    with pytest.raises(
        WorkflowConfigError,
        match='data.tags must be a non-empty list of strings',
    ):
        _validate(tmp_path, config)


def test_event_id_is_normalized_to_string(tmp_path):
    config = _config(tmp_path)
    config['event']['id'] = 12345

    normalized = _validate(tmp_path, config)

    assert normalized['event']['id'] == '12345'


def test_boolean_event_id_is_rejected(tmp_path):
    config = _config(tmp_path)
    config['event']['id'] = True

    with pytest.raises(WorkflowConfigError, match='event.id'):
        _validate(tmp_path, config)


def test_unquoted_yaml_timestamp_is_accepted(tmp_path):
    config = _config(tmp_path)
    path = tmp_path / 'event.yaml'
    text = yaml.safe_dump(config, sort_keys=False)
    text = text.replace(
        "time: '2009-04-07T20:12:55Z'",
        'time: 2009-04-07T20:12:55Z',
    )
    path.write_text(text, encoding='utf-8')

    normalized = validate_config(path)

    assert normalized['event']['time'].year == 2009


def test_non_syngine_greens_requires_path(tmp_path):
    config = _config(tmp_path)
    config['greens'] = {'format': 'FK', 'path': None}

    with pytest.raises(
        WorkflowConfigError,
        match='greens.path is required for non-SYNGINE formats',
    ):
        _validate(tmp_path, config)


def test_syngine_rejects_path(tmp_path):
    config = _config(tmp_path)
    config['greens']['path'] = 'https://example.com/syngine'

    with pytest.raises(
        WorkflowConfigError, match='greens.path is not supported for SYNGINE'
    ):
        _validate(tmp_path, config)


def test_validation_rejects_per_measurement_norm(tmp_path):
    config = _config(tmp_path)
    config['measurements']['body']['norm'] = 'L1'

    with pytest.raises(WorkflowConfigError, match='unknown key'):
        _validate(tmp_path, config)


def test_measurement_names_reject_path_characters(tmp_path):
    config = _config(tmp_path)
    config['measurements']['body/waves'] = config['measurements'].pop(
        'body'
    )

    with pytest.raises(
        WorkflowConfigError, match='measurement names may contain'
    ):
        _validate(tmp_path, config)


def test_total_is_reserved_for_combined_results(tmp_path):
    config = _config(tmp_path)
    config['measurements']['total'] = config['measurements'].pop('body')

    with pytest.raises(WorkflowConfigError, match='reserved'):
        _validate(tmp_path, config)


@pytest.mark.parametrize('value', ['false', 0, None])
def test_processing_flags_require_booleans(tmp_path, value):
    config = _config(tmp_path)
    config['measurements']['body']['processing'] = {
        'apply_scaling': value,
    }

    with pytest.raises(
        WorkflowConfigError, match='apply_scaling must be boolean'
    ):
        _validate(tmp_path, config)


def test_invalid_selector_types_raise_config_errors(tmp_path):
    config = _config(tmp_path)
    config['picks']['type'] = []
    with pytest.raises(WorkflowConfigError, match='picks.type'):
        _validate(tmp_path, config)

    config = _config(tmp_path)
    config['source']['type'] = []
    with pytest.raises(WorkflowConfigError, match='source.type'):
        _validate(tmp_path, config)


def test_non_string_unknown_key_has_clear_error(tmp_path):
    config = _config(tmp_path)
    config[4] = 'unsupported'

    with pytest.raises(WorkflowConfigError, match='unknown top-level key'):
        _validate(tmp_path, config)


@pytest.mark.parametrize('norm', [None, 'L3'])
def test_misfit_norm_rejects_invalid_values_cleanly(tmp_path, norm):
    config = _config(tmp_path)
    config['misfit']['norm'] = norm

    with pytest.raises(
        WorkflowConfigError, match='misfit.norm must be L1, L2, or hybrid'
    ):
        _validate(tmp_path, config)


@pytest.mark.parametrize('level', [True, 5, '1'])
def test_misfit_level_rejects_invalid_values_cleanly(tmp_path, level):
    config = _config(tmp_path)
    config['misfit']['level'] = level

    with pytest.raises(
        WorkflowConfigError, match='misfit.level must be 0, 1, or 2'
    ):
        _validate(tmp_path, config)


def test_native_processing_errors_include_measurement_path(tmp_path):
    config = _config(tmp_path)
    body = config['measurements']['body']
    body.pop('time_shift')
    body['processing'] = {'apply_padding': True}
    body['waveform_misfit'] = {
        'time_shift_min': -2.0,
        'time_shift_max': 2.0,
        'time_shift_groups': ['ZR'],
    }
    normalized = _validate(tmp_path, config)

    with pytest.raises(
        WorkflowConfigError, match=r'measurements\.body\.processing'
    ):
        _build_measurements(normalized)


def test_fk_metadata_requires_database_path(tmp_path):
    config = _config(tmp_path)
    config['picks'] = {'type': 'FK_metadata'}

    with pytest.raises(WorkflowConfigError, match='FK_database'):
        _validate(tmp_path, config)


@pytest.mark.parametrize(
    'source_type, grid_type, expected_function',
    [
        ('dc', 'regular', DoubleCoupleGridRegular),
        ('dev', 'regular', DeviatoricGridSemiregular),
        ('fmt', 'regular', FullMomentTensorGridSemiregular),
        ('dc', 'random', DoubleCoupleGridRandom),
        ('dev', 'random', DeviatoricGridRandom),
        ('fmt', 'random', FullMomentTensorGridRandom),
    ],
)
def test_grid_function_dispatch(
    tmp_path, source_type, grid_type, expected_function
):
    config = _validate(tmp_path, _config(tmp_path, source_type, grid_type))
    grid = _build_grid(config)

    if grid_type == 'regular':
        expected = expected_function(
            magnitudes=config['source']['magnitudes'],
            npts_per_axis=2,
        )
        assert grid.size == expected.size
        for actual_coord, expected_coord in zip(grid.coords, expected.coords):
            np.testing.assert_allclose(actual_coord, expected_coord)
        return

    # The native random DC/DEV/FMT functions all return UnstructuredGrid
    # objects of the same requested size. Their v/w constraints distinguish
    # the three source families.
    assert grid.size == 5 * len(config['source']['magnitudes'])
    v = np.asarray(grid.coords[1])
    w = np.asarray(grid.coords[2])

    if source_type == 'dc':
        assert np.all(v == 0.0)
        assert np.all(w == 0.0)
    elif source_type == 'dev':
        assert np.any(v != 0.0)
        assert np.all(w == 0.0)
    else:
        assert np.any(v != 0.0)
        assert np.any(w != 0.0)


@pytest.mark.parametrize('literal', ['1_000_000', '1000000', '1e6'])
def test_random_grid_npts_yaml_forms(tmp_path, literal):
    config = _config(tmp_path, 'fmt', 'random')
    path = tmp_path / ('random-%s.yaml' % literal.replace('_', ''))
    text = yaml.safe_dump(config, sort_keys=False)
    text = text.replace('npts: 5\n', 'npts: %s\n' % literal)
    path.write_text(text, encoding='utf-8')

    normalized = validate_config(path)

    assert normalized['source']['grid']['npts'] == 1_000_000


def test_random_grid_npts_preserves_large_integer_exactly(tmp_path):
    config = _config(tmp_path, 'fmt', 'random')
    config['source']['grid']['npts'] = 9_007_199_254_740_993

    normalized = _validate(tmp_path, config)

    assert normalized['source']['grid']['npts'] == 9_007_199_254_740_993


def test_bandpass_filter_accepts_fraction_and_scientific_notation(tmp_path):
    config = _config(tmp_path)
    config['measurements']['body']['filter'] = ['1/50', '1/10']
    config['measurements']['surface']['filter'] = ['2e-2', '1e-1']

    normalized = _validate(tmp_path, config)

    body = normalized['measurements']['body']['processing']
    surface = normalized['measurements']['surface']['processing']
    assert body['freq_min'] == pytest.approx(0.02)
    assert body['freq_max'] == pytest.approx(0.1)
    assert surface['freq_min'] == pytest.approx(0.02)
    assert surface['freq_max'] == pytest.approx(0.1)


@pytest.mark.parametrize('style', ['shorthand', 'native'])
def test_bandpass_period_inputs_use_native_conversion_and_resolve_to_frequency(
    tmp_path, style
):
    config = _config(tmp_path)
    body = config['measurements']['body']
    body.pop('filter')

    if style == 'shorthand':
        body['period'] = [10, 50]
    else:
        body['processing'] = {
            'filter_type': 'Bandpass',
            'period_min': 10,
            'period_max': 50,
        }

    normalized = _validate(tmp_path, config)
    processors, _, measurements = _build_measurements(normalized)

    processor = processors['body']
    resolved = measurements['body']['processing']
    assert processor.freq_min == pytest.approx(1.0 / 50.0)
    assert processor.freq_max == pytest.approx(1.0 / 10.0)
    assert resolved['freq_min'] == pytest.approx(1.0 / 50.0)
    assert resolved['freq_max'] == pytest.approx(1.0 / 10.0)
    assert 'period_min' not in resolved
    assert 'period_max' not in resolved


def test_filter_and_period_shorthand_are_mutually_exclusive(tmp_path):
    config = _config(tmp_path)
    config['measurements']['body']['period'] = [10, 50]

    with pytest.raises(WorkflowConfigError, match='filter or period'):
        _validate(tmp_path, config)


@pytest.mark.parametrize('value', ['1+1', '1__0'])
def test_numeric_strings_reject_non_scientific_syntax(tmp_path, value):
    config = _config(tmp_path)
    config['measurements']['body']['filter'] = [value, '1/10']

    with pytest.raises(WorkflowConfigError, match='simple fraction'):
        _validate(tmp_path, config)


@pytest.mark.parametrize(
    'magnitudes, explicit_magnitude, expected',
    [
        ([4.4, 4.5, 4.6], None, 4.5),
        ([4.6], None, 4.6),
        ([4.4, 4.5, 4.6], 4.7, 4.7),
    ],
)
def test_wavelet_magnitude_resolution(
    tmp_path, magnitudes, explicit_magnitude, expected
):
    config = _config(tmp_path)
    config['source']['magnitudes'] = magnitudes
    if explicit_magnitude is not None:
        config['wavelet']['magnitude'] = explicit_magnitude

    wavelet, magnitude = _build_wavelet(_validate(tmp_path, config))

    assert wavelet is not None
    assert magnitude == pytest.approx(expected)


@pytest.mark.parametrize(
    'source_type, function',
    [
        ('dev', DeviatoricGridSemiregular),
        ('fmt', FullMomentTensorGridSemiregular),
    ],
)
def test_semiregular_grid_arguments_are_exposed_and_used(
    tmp_path, source_type, function
):
    config = _validate(tmp_path, _config(tmp_path, source_type, 'regular'))
    grid_cfg = config['source']['grid']
    signature = inspect.signature(function)

    assert grid_cfg['tightness'] == signature.parameters['tightness'].default
    assert grid_cfg['uniformity'] == signature.parameters['uniformity'].default

    grid_cfg.update(tightness=0.6, uniformity=1.0)
    actual = _build_grid(config)
    expected = function(
        magnitudes=config['source']['magnitudes'],
        npts_per_axis=grid_cfg['npts_per_axis'],
        tightness=0.6,
        uniformity=1.0,
    )
    for actual_coord, expected_coord in zip(actual.coords, expected.coords):
        np.testing.assert_allclose(actual_coord, expected_coord)


def test_resolved_defaults_match_native_objects(tmp_path):
    recipe = _config(tmp_path)
    recipe['measurements']['P-waves'] = recipe['measurements'].pop('body')
    recipe['measurements']['P-waves']['time_shift'] = [-1.5, 2.5]
    prepared = prepare_workflow(_validate(tmp_path, recipe))
    resolved = prepared.resolved

    body = resolved['measurements']['P-waves']
    processor = prepared.processors['P-waves']
    body_misfit = prepared.misfits['P-waves']
    processing = body['processing']
    waveform = body['waveform_misfit']

    assert processing['apply_scaling'] is processor.apply_scaling
    if processor.apply_scaling:
        for key in ('scaling_power', 'scaling_coefficient'):
            assert processing[key] == pytest.approx(getattr(processor, key))
    else:
        assert 'scaling_power' not in processing
        assert 'scaling_coefficient' not in processing

    assert resolved['misfit']['level'] == body_misfit.level
    assert resolved['misfit']['verbose'] == body_misfit.verbose
    assert resolved['misfit']['normalize'] is body_misfit.normalize is True
    assert waveform['time_shift_min'] == pytest.approx(
        body_misfit.time_shift_min
    )
    assert waveform['time_shift_max'] == pytest.approx(
        body_misfit.time_shift_max
    )
    assert body_misfit.time_shift_min == pytest.approx(-1.5)
    assert body_misfit.time_shift_max == pytest.approx(2.5)

    assert {'filter', 'window', 'components', 'time_shift'}.isdisjoint(body)
    assert {'level', 'normalize'}.isdisjoint(waveform)
    assert 'picks' not in resolved
    assert 'units' not in resolved['data']
    assert resolved['data']['tags'] == ['units:m', 'type:velocity']

    assert resolved['source']['grid']['size'] == prepared.grid.size
    assert resolved['source']['grid']['function'] == (
        'mtuq.grid.DoubleCoupleGridRegular'
    )
    assert resolved['wavelet']['magnitude'] == pytest.approx(4.5)
    assert resolved['wavelet']['function'] == 'mtuq.util.cap.Trapezoid'
    assert resolved['objective']['coefficients'] == {
        'P-waves': 1.0,
        'surface': 1.0,
    }


def test_complete_recipe_round_trips_to_same_canonical_config(tmp_path):
    resolved = _resolved_config(tmp_path)

    rerun = _validate(tmp_path, resolved, 'resolved.yaml')
    resolved_again = prepare_workflow(rerun).resolved

    assert resolved_again == resolved


@pytest.mark.parametrize('style', ['abridged', 'resolved'])
def test_processing_override_changes_native_processor(tmp_path, style):
    config = (
        _config(tmp_path) if style == 'abridged'
        else _resolved_config(tmp_path)
    )
    config['measurements']['body'].setdefault('processing', {})[
        'apply_scaling'
    ] = False
    config = _validate(tmp_path, config)

    processors, _, measurements = _build_measurements(config)
    assert processors['body'].apply_scaling is False
    assert measurements['body']['processing']['apply_scaling'] is False
    assert 'scaling_power' not in measurements['body']['processing']
    assert 'scaling_coefficient' not in measurements['body']['processing']


def test_identical_shorthand_and_explicit_override_is_accepted(tmp_path):
    config = _config(tmp_path)
    config['measurements']['body']['processing'] = {
        'freq_min': 0.1,
        'filter_type': 'bandpass',
    }

    normalized = _validate(tmp_path, config)
    processing = normalized['measurements']['body']['processing']

    assert processing['freq_min'] == pytest.approx(0.1)
    assert processing['filter_type'] == 'Bandpass'


def test_conflicting_shorthand_and_explicit_override_is_rejected(tmp_path):
    config = _config(tmp_path)
    config['measurements']['body']['processing'] = {'freq_min': 0.05}

    with pytest.raises(WorkflowConfigError, match='conflicting values'):
        _validate(tmp_path, config)


def test_rerunning_resolved_recipe_preserves_original_input(tmp_path):
    output_dir = tmp_path / 'output'
    output_dir.mkdir()
    original = output_dir / 'config.input.yaml'
    resolved = output_dir / 'config.resolved.yaml'
    original.write_text('# original abridged recipe\n', encoding='utf-8')
    resolved.write_text('# generated resolved recipe\n', encoding='utf-8')

    _write_input_config(resolved, output_dir)

    assert original.read_text(encoding='utf-8') == (
        '# original abridged recipe\n'
    )


def test_custom_objective_coefficients_are_rejected(tmp_path):
    config = _config(tmp_path)
    config['objective'] = {
        'coefficients': {
            'body': 0.7,
            'surface': 0.3,
        }
    }

    with pytest.raises(
        WorkflowConfigError, match='custom objective coefficients'
    ):
        _validate(tmp_path, config)


def test_generated_metadata_is_refreshed(tmp_path):
    resolved = _resolved_config(tmp_path)
    stale = copy.deepcopy(resolved)
    stale['source']['grid']['npts_per_axis'] = 3
    stale['source']['grid']['function'] = 'stale.grid.Function'
    stale['source']['grid']['size'] = -1
    stale['wavelet']['function'] = 'stale.wavelet.Function'
    stale['workflow'] = {'version': -1}
    stale['mtuq'] = {'version': 'stale', 'git_commit': 'stale'}

    config = _validate(tmp_path, stale, 'stale.yaml')
    prepared = prepare_workflow(config)
    refreshed = prepared.resolved

    assert refreshed['source']['grid']['function'] == (
        'mtuq.grid.DoubleCoupleGridRegular'
    )
    assert refreshed['source']['grid']['size'] == prepared.grid.size
    assert refreshed['source']['grid']['size'] != -1
    assert refreshed['wavelet']['function'] == 'mtuq.util.cap.Trapezoid'
    assert refreshed['workflow'] != stale['workflow']
    assert refreshed['mtuq']['version'] != 'stale'
    assert refreshed['mtuq']['git_commit'] != 'stale'


def test_run_combines_terms_selects_source_and_writes_outputs(
    tmp_path, monkeypatch
):
    """Test orchestration with real preparation and JSON/YAML writing.

    Mock data I/O, processing, search and native-result serialization;
    numerical parity and serialization belong to integration tests.
    """
    runner = importlib.import_module('mtuq.workflow.run')
    recipe = _write_config(tmp_path, _config(tmp_path))
    output = tmp_path / 'override'
    data, greens = Mock(), Mock()
    data_terms, greens_terms = [object(), object()], [object(), object()]
    data.map.side_effect = data_terms
    greens.map.side_effect = greens_terms
    database = Mock()
    database.get_greens_tensors.return_value = greens

    class Surface(np.ndarray):
        def source_idxmin(self):
            return int(self.argmin())

    def search(data_term, greens_term, misfit, origin, grid):
        values = np.full(grid.size, 100.0)
        # Neither term prefers source 1 individually; their sum does.
        values[:3] = (
            [0.0, 2.0, 8.0] if data_term is data_terms[0]
            else [8.0, 2.0, 0.0]
        )
        return values.view(Surface)

    search_mock, save_results = Mock(side_effect=search), Mock()
    monkeypatch.setattr(runner, '_mpi_comm', lambda: None)
    monkeypatch.setattr(runner, 'read', Mock(return_value=data))
    monkeypatch.setattr(
        runner, '_open_greens_database', Mock(return_value=database)
    )
    monkeypatch.setattr(runner, 'grid_search', search_mock)
    monkeypatch.setattr(runner, '_save_native_results', save_results)

    result = run(recipe, output=output)

    data.sort_by_distance.assert_called_once_with()
    database.get_greens_tensors.assert_called_once_with(
        data.get_stations.return_value, result['origin']
    )
    greens.convolve.assert_called_once()
    assert data.map.call_count == greens.map.call_count == 2
    assert search_mock.call_count == 2
    for index, name in enumerate(('body', 'surface')):
        processor = result['processors'][name]
        assert data.map.call_args_list[index].args == (processor,)
        assert greens.map.call_args_list[index].args == (processor,)
        args = search_mock.call_args_list[index].args
        assert args[0] is result['data'][name] is data_terms[index]
        assert args[1] is result['greens'][name] is greens_terms[index]
        assert args[2] is result['misfits'][name]
        assert args[3] is result['origin']
        assert args[4] is result['grid']

    np.testing.assert_allclose(
        result['results'], result['terms']['body'] + result['terms']['surface']
    )
    expected_source = result['grid'].get(1)
    assert result['source'].as_dict() == expected_source.as_dict()
    save_results.assert_called_once()
    saved = save_results.call_args.args
    assert saved[0] == output
    assert saved[1] is result['results']
    assert saved[2] is result['terms']
    assert (output / 'config.input.yaml').read_bytes() == recipe.read_bytes()
    resolved = yaml.safe_load((output / 'config.resolved.yaml').read_text())
    assert resolved == result['config']
    assert resolved['output'] == str(output)
    solution = json.loads((output / 'solution.json').read_text())
    for key, value in expected_source.as_dict().items():
        assert solution[key] == pytest.approx(value)
    assert solution['Mw'] == pytest.approx(expected_source.magnitude())
    assert solution['depth_in_m'] == pytest.approx(result['origin'].depth_in_m)
