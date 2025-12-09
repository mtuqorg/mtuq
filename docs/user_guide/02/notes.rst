
Detailed notes
==============

How data are read in
--------------------

MTUQ natively supports only SAC file format. For reading SAC files, MTUQ uses only a thin wrapper over `obspy.read`.  Similarly, for storing waveform data, MTUQ uses only a thin layer over ObsPy `Stream`, `Trace` and `Stats` structures.  


Another way of stating data and metadata requirements
-----------------------------------------------------

Because of how data are read in, SAC files must be ObsPy compliant and enough information must be present in SAC headers to populate ObsPy structures.

In the end, all that's required is that the following ObsPy attributes are correctly determined:

- `trace.stats.network`
- `trace.stats.station`
- `trace.stats.location`
- `trace.stats.sac.stala`
- `trace.stats.sac.stlo`
- `trace.stats.deltat`
- `trace.stats.starttime`


How metadata are used
---------------------

The above ObsPy attributes are used as follows:

- `network`, `station`, and `location` codes are used for sorting individual waveforms into `mtuq.Datasets`
- station `latitude` and `longitude` are needed to extract Green's functions, as well as for distance-dependent data processing
- `starttimes` and `deltat` are used to align observations with synthetics during misfit evaluation


On time conventions
-------------------

The `starttime` of a SAC file is determined by combining a reference time (defined by multiple SAC headers) with an offset value (defined by the "begin time" or "B" header). Usually, the reference time represents a catalog origin time given as an actual UTC or GMT date and time.

Other conventions are possible, however.  For example, synthetic inversions may involve a `t=0` reference time rather than an actual date and time. The only requirement is that the observed data follow the same convention as the Green's functions and synthetic data.


On geographic conventions
-------------------------

By default, distance and azimuth calculations are performed on station latitudes and longitudes using ObsPy geodetic functions.

To support local `x,y` coordinates, UTM coordinates, or other custom systems, these distance and azimuth functions (and by extension, the meaning of the `latitude` and `longitude` values themselves) can be overridden through modifications to MTUQ's source code.  (We recommend an "editable" installation of MTUQ for such modifications.)


