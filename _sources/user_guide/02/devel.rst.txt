
Developer notes
===============

Other file formats
------------------

Natively, MTUQ supports only SAC file format.  Other formats can be added using an ObsPy-like plug-in system.

To add support for a new file format

- Create a module, i.e. `mtuq/io/readers/newformat.py`
- In the module, implement a reader, i.e. `_read_newformat`
- Register the reader by adding to `project.entry-points` in `pyproject.toml`

The new reader can then be invoked as ``mtuq.read(filename, format='newformat')``.

To get a sense how it works, see the MTUQ implementation for SAC or the ObsPy implementation for other file formats.


Further considerations
----------------------

To maintain flexibility, it will be important that

- `mtuq.read` handles all file format-dependent metadata parsing
- metadata are stored in a common ObsPy-based internal structure that works the same regardless of the original file format
- data processing, misfit evaluation and other functions work the same regardless of the original file format
