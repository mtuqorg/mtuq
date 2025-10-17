
Custom backends
===============

Custom backends can be used to modify the appearance of many MTUQ plotting functions.

If a plotting function has a `backend` keyword argument, then feel free to pass it a custom backend function.


Example
-------

The backend function below can be passed to `plot_time_shifts <https://mtuqorg.github.io/mtuq/library/generated/mtuq.graphics.plot_time_shifts.html>`_, `plot_amplitude_ratios <https://mtuqorg.github.io/mtuq/library/generated/mtuq.graphics.plot_amplitude_ratios.html>`_, `plot_log_amplitude_ratios <https://mtuqorg.github.io/mtuq/library/generated/mtuq.graphics.plot_log_amplitude_ratios.html>`_, and `_plot_attrs <https://mtuqorg.github.io/mtuq/library/generated/mtuq.graphics._plot_attrs.html>`_. 

Note that the backend function contains hardcoded settings that improve the appearance `Silwal2016 <https://mtuqorg.github.io/mtuq/references.html>`_ figures, but lack generality for default MTUQ plotting.


.. code::

  def custom_backend(filename, values, stations, origin,
      zero_centered=True, label='Time shift (s)'):

      import pygmt

      fig = pygmt.Figure()

      gmt_region = "g-155/-145/59/64"
      gmt_projection = "D-150/61.5/60/63/9c"
      gmt_frame = ["xa5", "ya2"]

      fig.basemap(
          region=gmt_region,
          projection=gmt_projection,
          frame=gmt_frame,
          )

      fig.coast(
          land="grey80",
          shorelines=True,
          area_thresh=100,
          )

      # construct color palette
      cmap = "polar"

      if zero_centered:
          limits = (-np.max(np.abs(values)), +np.max(np.abs(values)))
      else:
          limits = (np.min(values), np.max(values))

      pygmt.makecpt(
          cmap=cmap,
          series=limits,
          )

      # plot stations
      for _i, station in enumerate(stations):
          fig.plot(
              x=station.longitude, y=station.latitude,
              cmap=True, color="+z", zvalue=values[_i],
              style="t.5c", pen="1p",
              )

      # plot line segments
      for _i, station in enumerate(stations):
          fig.plot(
              x=(origin.longitude, station.longitude),
              y=(origin.latitude, station.latitude),
              cmap=True, zvalue=values[_i],
              pen="thick,+z,-"
              )

      # plot origin
      fig.plot(
          x=origin.longitude, y=origin.latitude,
          style="a.5c", color="black", pen="1p"
          )

      # add colorbar
      fig.colorbar(frame=["x+l%s" % label])

      fig.savefig(filename)

