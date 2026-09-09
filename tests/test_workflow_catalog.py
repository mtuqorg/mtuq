import csv
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from mtuq.workflow import WorkflowConfigError
from mtuq.workflow.catalog import run_catalog


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _weights(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '0.AK.TEST.00 0 1 1 1 1 1 0 0 0 0 0 0\n',
        encoding='utf-8',
    )


def _recipe(directory, event_id, *, output=None, magnitude=4.5):
    data_dir = directory / event_id
    _weights(data_dir / 'weights.dat')

    config = {
        'version': 1,
        'event': {
            'id': event_id,
            'time': '2009-04-07T20:12:55Z',
            'latitude': 61.4542,
            'longitude': -149.7428,
            'depth_in_m': 33033.6,
        },
        'data': {
            'format': 'SAC',
            'path': '%s/*.[zrt]' % event_id,
            'weights': '%s/weights.dat' % event_id,
            'units': 'm/s',
        },
        'greens': {
            'format': 'SYNGINE',
            'model': 'ak135',
        },
        'picks': {'type': 'taup', 'model': 'ak135'},
        'misfit': {'norm': 'L2', 'normalize': True},
        'measurements': {
            'body': {
                'filter': [0.1, 0.333],
                'window': ['body_wave', 15],
                'components': ['ZR'],
                'time_shift': 2,
            }
        },
        'source': {
            'type': 'dc',
            'magnitudes': [magnitude],
            'grid': {'type': 'regular', 'npts_per_axis': 2},
        },
        'wavelet': {'type': 'trapezoid'},
        'output': output or str(directory / 'output' / event_id),
    }

    path = directory / (event_id + '.yaml')
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    return path


def _fake_success(recipe_path):
    config = yaml.safe_load(Path(recipe_path).read_text(encoding='utf-8'))
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.resolved.yaml').write_text(
        'version: 1\n', encoding='utf-8'
    )
    (output / 'results_total.nc').touch()
    (output / 'solution.json').write_text(
        json.dumps(
            {
                'Mw': config['source']['magnitudes'][0],
                'latitude': config['event']['latitude'],
                'longitude': config['event']['longitude'],
                'depth_in_m': config['event']['depth_in_m'],
            }
        ),
        encoding='utf-8',
    )
    return SimpleNamespace(returncode=0)


def test_validate_directory_of_complete_event_recipes(tmp_path):
    directory = tmp_path / 'events'
    directory.mkdir()
    _recipe(directory, 'e1')
    _recipe(directory, 'e2')

    result = run_catalog(directory, validate_only=True)

    assert result['valid'] is True
    assert [item['event_id'] for item in result['recipes']] == ['e1', 'e2']

    config = result['recipes'][0]['config']

    assert config['data']['path'] == str(
        (directory / 'e1' / '*.[zrt]').resolve()
    )
    assert config['data']['weights'] == str(
        (directory / 'e1' / 'weights.dat').resolve()
    )


def test_invalid_recipe_prevents_catalog_start(tmp_path, monkeypatch, capsys):
    directory = tmp_path / 'events'
    directory.mkdir()
    _recipe(directory, 'e1')
    bad = _recipe(directory, 'README')
    config = {'notes': 'not an event recipe'}
    bad.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')

    calls = []
    monkeypatch.setattr(
        'mtuq.workflow.catalog._run_event_process',
        lambda path, timeout=None: calls.append(path),
    )

    result = run_catalog(directory)
    output = capsys.readouterr().out

    assert result['valid'] is False
    assert calls == []
    assert 'README.yaml: cannot be used as a catalog event recipe' in output
    assert 'move it out of the catalog directory' in output


def test_duplicate_event_ids_are_rejected_but_summary_still_works(tmp_path):
    directory = tmp_path / 'events'
    directory.mkdir()
    first = _recipe(directory, 'e1')
    second = _recipe(directory, 'e2')
    config = yaml.safe_load(second.read_text(encoding='utf-8'))
    config['event']['id'] = 'e1'
    second.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding='utf-8'
    )
    _fake_success(first)

    rejected = run_catalog(directory, validate_only=True)
    assert rejected['valid'] is False
    assert rejected['failures'][0][0].name == 'e2.yaml'

    summarized = run_catalog(directory, summary_only=True)
    with summarized['summary'].open(
        'r', encoding='utf-8', newline=''
    ) as handle:
        rows = list(csv.DictReader(handle))

    by_name = {Path(row['recipe']).name: row for row in rows}
    assert by_name['e1.yaml']['status'] == 'complete'
    assert by_name['e2.yaml']['status'] == 'invalid'


def test_output_collisions_are_rejected_but_summary_still_works(tmp_path):
    directory = tmp_path / 'events'
    directory.mkdir()
    shared = str(tmp_path / 'same-output')
    first = _recipe(directory, 'e1', output=shared)
    _recipe(directory, 'e2', output=shared)
    _fake_success(first)

    rejected = run_catalog(directory, validate_only=True)
    assert rejected['valid'] is False
    assert rejected['failures'][0][0].name == 'e2.yaml'

    summarized = run_catalog(directory, summary_only=True)
    with summarized['summary'].open(
        'r', encoding='utf-8', newline=''
    ) as handle:
        rows = list(csv.DictReader(handle))

    by_name = {Path(row['recipe']).name: row for row in rows}
    assert by_name['e1.yaml']['status'] == 'complete'
    assert by_name['e2.yaml']['status'] == 'invalid'


def test_execution_failure_does_not_block_later_event(
    tmp_path, monkeypatch, capsys
):
    directory = tmp_path / 'events'
    directory.mkdir()
    _recipe(directory, 'e1')
    _recipe(directory, 'e2')
    calls = []

    def runner(path, timeout=None):
        calls.append(Path(path).stem)
        if Path(path).stem == 'e1':
            return SimpleNamespace(returncode=7)
        return _fake_success(path)

    monkeypatch.setattr('mtuq.workflow.catalog._run_event_process', runner)
    result = run_catalog(directory)
    output = capsys.readouterr().out

    assert calls == ['e1', 'e2']
    assert len(result['failures']) == 1
    assert result['failures'][0][0].name == 'e1.yaml'
    assert '1 of 2 event(s) failed: e1.yaml' in output


def test_timeout_does_not_block_later_event(tmp_path, monkeypatch):
    directory = tmp_path / 'events'
    directory.mkdir()
    _recipe(directory, 'e1')
    _recipe(directory, 'e2')
    calls = []

    def runner(path, timeout=None):
        calls.append((Path(path).stem, timeout))
        if Path(path).stem == 'e1':
            raise subprocess.TimeoutExpired(str(path), timeout)
        return _fake_success(path)

    monkeypatch.setattr('mtuq.workflow.catalog._run_event_process', runner)
    result = run_catalog(directory, timeout=30)

    assert calls == [('e1', 30), ('e2', 30)]
    assert len(result['failures']) == 1
    assert result['failures'][0][1] == 'timed out after 30 s'


def test_resume_skips_complete_outputs(tmp_path, monkeypatch):
    directory = tmp_path / 'events'
    directory.mkdir()
    e1 = _recipe(directory, 'e1')
    _recipe(directory, 'e2')
    _fake_success(e1)
    calls = []

    def runner(path, timeout=None):
        calls.append(Path(path).stem)
        return _fake_success(path)

    monkeypatch.setattr('mtuq.workflow.catalog._run_event_process', runner)
    result = run_catalog(directory, resume=True)

    assert result['valid'] is True
    assert calls == ['e2']


def test_summary_is_simple_and_can_be_regenerated(tmp_path):
    directory = tmp_path / 'events'
    directory.mkdir()
    e1 = _recipe(directory, 'e1', magnitude=4.3999999999999995)
    _fake_success(e1)
    _recipe(directory, 'e2')
    e3 = _recipe(directory, 'e3')
    _fake_success(e3)
    (directory / 'output' / 'e3' / 'solution.json').write_text(
        '[]\n', encoding='utf-8'
    )
    (directory / 'README.yaml').write_text(
        'notes: not an event recipe\n', encoding='utf-8'
    )

    result = run_catalog(directory, summary_only=True)

    with result['summary'].open('r', encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    by_name = {Path(row['recipe']).name: row for row in rows}
    assert set(by_name) == {'e1.yaml', 'e2.yaml', 'e3.yaml', 'README.yaml'}
    assert by_name['e1.yaml']['event_id'] == 'e1'
    assert by_name['e1.yaml']['status'] == 'complete'
    assert by_name['e1.yaml']['Mw'] == '4.4'
    for name in ('e2.yaml', 'e3.yaml'):
        assert by_name[name]['status'] == 'incomplete'
        assert by_name[name]['Mw'] == ''
    assert by_name['README.yaml']['event_id'] == ''
    assert by_name['README.yaml']['status'] == 'invalid'


def test_catalog_rejects_mpi_world(tmp_path, monkeypatch):
    directory = tmp_path / 'events'
    directory.mkdir()
    _recipe(directory, 'e1')
    monkeypatch.setattr('mtuq.workflow.catalog._mpi_world_size', lambda: 2)

    with pytest.raises(
        WorkflowConfigError, match='must not be launched with MPI'
    ):
        run_catalog(directory, validate_only=True)


def test_empty_directory_is_rejected(tmp_path):
    directory = tmp_path / 'events'
    directory.mkdir()

    with pytest.raises(WorkflowConfigError, match='no YAML event recipes'):
        run_catalog(directory, validate_only=True)
