
from importlib.metadata import entry_points


def _greens_databases():
    databases = {}
    for entry_point in entry_points('greens_databases'):
        databases[entry_point.name] = entry_point.load()
    return databases

def open_db(format='', **kwargs):
    format = format.upper()
    return _greens_databases()[format](**kwargs)

