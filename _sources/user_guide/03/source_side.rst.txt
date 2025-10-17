
.. warning::

   This page is still under construction.  To help improve the
   documentation, feel free to submit a pull request.


Source-side 3D Green's functions
================================

Generating source-side 3D Green's functions using SPECFEM3D/3D_GLOBE
--------------------------------------------------------------------

In principle, any 3D solver can be used to generate source-side Green's functions, as long as the `requirements  <https://mtuqorg.github.io/mtuq/user_guide/03/source_side.html#requirements-for-mtuq-source-side-green-s-functions>`_ below are satisfied.  

So far, however, the machinery has only been tested using SPECFEM3D/3D_GLOBE.  To convert SPECFEM3D/3D_GLOBE output to MTUQ-compliant Green's functions, the following steps are necessary.


Generate SAC binary files
.........................

SAC binary output format is natively supported by SPECFEM3D_GLOBE (in the parameter file, set `OUTPUT_SEISMOS_SAC_BINARY = .true.`).

Unfortunately, SAC binary output format is not natively supported by SPECFEM3D, so it is necessary to manually convert SPECFEM3D output.


Convert to SI units
...................

SPECFEM3D/3D_GLOBE use a mixed SI and CGS convention, in which moment tensor elements are input in terms of dynes and centimeters, and seismograms are output in meters.    In contrast, MTUQ uses a fully SI :ref:`convention<Units convention>`.   As a result, it is necessary to scale SPECFEM3D/3D_GLOBE seismograms by 10^7 prior to using them as MTUQ Green's functions.


Other amplitude scaling
.......................

In addition to converting to SI units, it is also necessary to account for any scaling factors in the SPECFEM3D/3D_GLOBE input files. Such scaling factors can enter, for example, through the `M_rr,M_tt,M_pp,M_rt,M_rp,M_tp` values in the moment tensor input file or through the `scaling factor <https://github.com/SPECFEM/specfem3d/blob/bf45798f3af9d792326a829de920fd944cf7c7dd/EXAMPLES/applications/homogeneous_halfspace_HEX27_elastic_no_absorbing/DATA/FORCESOLUTION#L8>`_ in the force input file.


Rotate traces
.............

Conveniently, SPECFEM3D_GLOBE can be made to automatically rotate output seismograms into vertical, radial and transverse components (set `ROTATE_SEISMOGRAMS_RT = .true.` in the parameter file).

No modifications are necessary on account of moment tensor basis convention, since MTUQ's `up-south-east` convention matches SPECFEM3D/3D_GLOBE's.




Requirements for MTUQ source-side 3D Green's functions
------------------------------------------------------

File format
...........

Individual Green's functions must be written to SAC binary files.  

A total of 18 SAC binary files are required to represent the response between a given hypocenter and station (corresponding to 6 moment tensor elements times 3 directions of motion).


Units convention
................

For a moment tensor inversion, each SAC binary file must give the response in meters to a 1 Newton-meter force couple.

For a force inversion, each SAC binary file must give the response in meters to a 1 Newton force.

In both cases, MTUQ uses a fully SI units convention (compare with SPECFEM3D/3D_GLOBE notes, above).



Basis convention
................

MTUQ uses an `up-south-east` basis convention in which `r` denotes vertically upward, `t` denotes south, and `p` denotes east.

Green's functions must be rotated into into vertical `Z`, radial `R` and transverse `T` components relative to the source-receiver backazimuth.

Place all seismograms for the same hypocenter in a single directory as follows:

.. code ::

  {event_id}/
      {depth_in_km}/
          {net}.{sta}.{loc}.Z.Mrr.sac
          {net}.{sta}.{loc}.Z.Mtt.sac
          {net}.{sta}.{loc}.Z.Mpp.sac
          {net}.{sta}.{loc}.Z.Mrt.sac
          {net}.{sta}.{loc}.Z.Mrp.sac
          {net}.{sta}.{loc}.Z.Mtp.sac 
          {net}.{sta}.{loc}.R.Mrr.sac
          {net}.{sta}.{loc}.R.Mtt.sac
          {net}.{sta}.{loc}.R.Mpp.sac
          {net}.{sta}.{loc}.R.Mrt.sac
          {net}.{sta}.{loc}.R.Mrp.sac
          {net}.{sta}.{loc}.R.Mtp.sac
          {net}.{sta}.{loc}.T.Mrr.sac
          {net}.{sta}.{loc}.T.Mtt.sac
          {net}.{sta}.{loc}.T.Mpp.sac
          {net}.{sta}.{loc}.T.Mrt.sac
          {net}.{sta}.{loc}.T.Mrp.sac
          {net}.{sta}.{loc}.T.Mtp.sac
          ...

The corresponding convention for force responses is:

.. warning::

   Not yet tested; subject to change without notice.


.. code ::

  {event_id}/
      {depth_in_km}/
          {net}.{sta}.{loc}.Z.Fz.sac
          {net}.{sta}.{loc}.Z.Fs.sac
          {net}.{sta}.{loc}.Z.Fe.sac
          {net}.{sta}.{loc}.R.Fz.sac
          {net}.{sta}.{loc}.R.Fs.sac
          {net}.{sta}.{loc}.R.Fe.sac
          {net}.{sta}.{loc}.T.Fz.sac
          {net}.{sta}.{loc}.T.Fs.sac
          {net}.{sta}.{loc}.T.Fe.sac
          ...


Origin time convention
......................

For origin time, MTUQ uses a centroid convention (`more details <https://github.com/mtuqorg/mtuq/issues/140>`_), so that `t=0` in the `GreensTensor` time discretization corresponds to mean source excitation time.

MTUQ uses the begin time (`B`) and end time (`E`) headers from the SAC binary files to align the Green's functions relative to centroid origin time.  

Currently, these are the only SAC headers used in reading Green's functions.

(Note that different `SAC headers <https://ds.iris.edu/files/sac-manual/manual/file_format.html>`_ are required in reading `observed data <https://mtuqorg.github.io/mtuq/user_guide/02.html#file-format-metadata-and-data-processing-requirements>`_.)



Hypocenter searches (experimental)
----------------------------------

Currently, only searches over source depth are possible with source-side 3D Green's functions (no other hypocenter parameters).

The current `depth search <https://github.com/mtuqorg/mtuq/blob/568e49a73817e4e2dbab1189210214da6906266f/mtuq/io/clients/SPECFEM3D_SAC.py#L117>`_ implementation is especially crude and experimental (consider local modifications to suit your needs).

To allow depth searches, create subdirectories for each centroid depth as follows:

.. code ::

  {event_id}/
      {depth_in_km}/
          {net}.{sta}.{loc}.Z.Mrr.sac
          {net}.{sta}.{loc}.Z.Mtt.sac
          {net}.{sta}.{loc}.Z.Mpp.sac
          {net}.{sta}.{loc}.Z.Mrt.sac
          {net}.{sta}.{loc}.Z.Mrp.sac
          {net}.{sta}.{loc}.Z.Mtp.sac 
          {net}.{sta}.{loc}.R.Mrr.sac
          {net}.{sta}.{loc}.R.Mtt.sac
          {net}.{sta}.{loc}.R.Mpp.sac
          {net}.{sta}.{loc}.R.Mrt.sac
          {net}.{sta}.{loc}.R.Mrp.sac
          {net}.{sta}.{loc}.R.Mtp.sac
          {net}.{sta}.{loc}.T.Mrr.sac
          {net}.{sta}.{loc}.T.Mtt.sac
          {net}.{sta}.{loc}.T.Mpp.sac
          {net}.{sta}.{loc}.T.Mrt.sac
          {net}.{sta}.{loc}.T.Mrp.sac
          {net}.{sta}.{loc}.T.Mtp.sac
      {depth_in_km}/
          {net}.{sta}.{loc}.Z.Mrr.sac
          {net}.{sta}.{loc}.Z.Mtt.sac
          {net}.{sta}.{loc}.Z.Mpp.sac
          {net}.{sta}.{loc}.Z.Mrt.sac
          {net}.{sta}.{loc}.Z.Mrp.sac
          {net}.{sta}.{loc}.Z.Mtp.sac 
          {net}.{sta}.{loc}.R.Mrr.sac
          {net}.{sta}.{loc}.R.Mtt.sac
          {net}.{sta}.{loc}.R.Mpp.sac
          {net}.{sta}.{loc}.R.Mrt.sac
          {net}.{sta}.{loc}.R.Mrp.sac
          {net}.{sta}.{loc}.R.Mtp.sac
          {net}.{sta}.{loc}.T.Mrr.sac
          {net}.{sta}.{loc}.T.Mtt.sac
          {net}.{sta}.{loc}.T.Mpp.sac
          {net}.{sta}.{loc}.T.Mrt.sac
          {net}.{sta}.{loc}.T.Mrp.sac
          {net}.{sta}.{loc}.T.Mtp.sac
      ...


Working example
---------------

A working of example using source-side 3D Green's functions from SPECFEM3D_GLOBE in a moment tensor inversion:

`test_greens_SPECFEM3D_SAC.py <https://github.com/rmodrak/mtuq/blob/master/tests/test_greens_SPECFEM3D_SAC.py>`_

