"""YAML-driven execution of ordinary MTUQ inversions.

The workflow layer translates an event recipe into the same native MTUQ
objects and operations used by an ordinary processing script.  Configuration
is loaded and normalized in :mod:`mtuq.workflow.config`; native ``Origin``,
grid, wavelet, ``ProcessData``, and ``WaveformMisfit`` objects are assembled in
:mod:`mtuq.workflow.build`; :func:`run` performs data and Green's-function I/O,
processing, grid search, result combination, and output writing.  Catalog
orchestration launches that same single-event path in isolated processes.

The stable public interface is ``run()``, ``validate_config()``,
``run_catalog()``, and ``WorkflowConfigError``.
"""

from .catalog import run_catalog as run_catalog
from .config import WorkflowConfigError as WorkflowConfigError
from .config import validate_config as validate_config
from .run import main as main
from .run import run as run


__all__ = [
    'WorkflowConfigError',
    'main',
    'run',
    'run_catalog',
    'validate_config',
]
