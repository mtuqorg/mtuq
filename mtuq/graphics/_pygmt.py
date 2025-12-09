
import numpy as np


def exists_pygmt():
    try:
        import pygmt
        return True
    except:
        return False


def plot_force_pygmt(filename, force_dict):

    import pygmt
    fig = pygmt.Figure()

    lat = np.degrees(np.pi/2 - np.arccos(force_dict['h']))
    lon = _wrap(force_dict['phi'] + 90.)

    proj_arg="A0/0/6i"
    area_arg="-180/180/-90/90"

    # specify basemap
    fig.basemap(projection=proj_arg, region=area_arg, frame=['xg180','yg30'])

    # plot arcs
    fig.text(x=90./2., y=0., text='E', font='40p')
    fig.plot(x=[90./2.,90./2.], y=[90.,7.5], pen='1.5p,0/0/0/35')
    fig.plot(x=[90./2.,90./2.], y=[-90.,-7.5], pen='1.5p,0/0/0/35')
    #fig.text(x=0., y=0., text='S', font='40p')
    fig.plot(x=[0.,0.], y=[90.,7.5], pen='1.5p,0/0/0/35')
    fig.plot(x=[0.,0.], y=[-90.,-7.5], pen='1.5p,0/0/0/35')
    fig.text(x=-90./2., y=0., text='W', font='40p')
    fig.plot(x=[-90./2.,-90./2.], y=[90.,7.5], pen='1.5p,0/0/0/35')
    fig.plot(x=[-90./2.,-90./2.], y=[-90.,-7.5], pen='1.5p,0/0/0/35')

    # plot force orientation
    fig.plot(x=lon/2., y=lat, style="d40p", pen="1p,black", fill="black")

    fig.savefig(filename)


def _wrap(angle_in_deg):
    """ Wraps angle to (-180, 180)
    """
    angle_in_deg %= 360.
    if angle_in_deg > 180.:
        angle_in_deg -= 360.
    return angle_in_deg



class PyGMTUtilities:
    """
    Utility class for PyGMT cartographic plots

    This class offers a set of static methods designed for enhancing and simplifying the usage of PyGMT 
    for plotting by handling plotting regions, color maps, LaTeX annotations, and plot headers.

    .. note ::
        The class is designed to be used without instantiation due to its static methods. This approach 
        helps in organizing code related to the PyGMT plotting backend and avoids confusion with other plotting backends.

    Methods include calculating plotting regions with buffers, configuring colormaps, preparing LaTeX 
    annotations for PyGMT, and generating standardized headers for plots.

    Examples and more detailed method descriptions can be found in the documentation of each method.
    """

    @staticmethod
    def calculate_plotting_region(stations, origin, buffer_percentage=0.1):
        """
        Calculates the region for plotting, including a buffer area around specified stations and origin.

        .. rubric :: Parameters

        ``stations`` (`list` of `mtuq.Station` objects):
        The stations to be included in the plot.

        ``origin`` (`mtuq.Origin` object):
        The origin object is used to calculate the region for the plot in case the origin is outside the range of the stations.

        ``buffer_percentage`` (`float`, optional):
        The percentage of the total longitude and latitude range to be added as a buffer around the specified region.
        Defaults to 0.1 (10%).

        .. rubric :: Returns

        ``region`` (`list` of `float`), ``lat_buffer`` (`float`):
        A tuple containing the calculated region as a list `[west, east, south, north]` and the latitude buffer value.
        The latitude buffer is returned to later be used for adjusting text spacing in the plot header.
        
        .. rubric :: Example

        >>> region, lat_buffer = PyGMTUtilities.calculate_plotting_region(stations, origin)
        >>> print(region)
        [149.55, 151.45, -35.1, -32.9]
        >>> print(lat_buffer)
        0.22
        """
        
        longitudes = [station.longitude for station in stations] + [origin.longitude]
        latitudes = [station.latitude for station in stations] + [origin.latitude]

        lon_buffer = (max(longitudes) - min(longitudes)) * buffer_percentage
        lat_buffer = (max(latitudes) - min(latitudes)) * buffer_percentage

        region = [min(longitudes) - lon_buffer, max(longitudes) + lon_buffer,
                min(latitudes) - lat_buffer, max(latitudes) + lat_buffer]
        return region, lat_buffer


    @staticmethod
    def get_resolution(lon_range, lat_range):
        """
        Determines the appropriate PyGMT etopo grid resolution based on longitude and latitude ranges.

        .. rubric :: Parameters

        ``lon_range`` (`float`):
        The longitudinal range of the area of interest.

        ``lat_range`` (`float`):
        The latitudinal range of the area of interest.

        .. rubric :: Returns

        ``resolution`` (`str`):
        The resolution string for PyGMT, e.g., '01m', '15s', ...,  based on the size of the specified area.

        .. note ::
            The resolution is determined based on predefined thresholds for the ranges, aiming to balance 
            detail and performance for different scales of geographic areas

            - If lon_range > 10 or lat_range > 10, the resolution is '01m'.

            - If lon_range > 5 or lat_range > 5, the resolution is '15s'.
            
            - If lon_range > 2 or lat_range > 2, the resolution is '03s'.
            
            - If lon_range > 1 or lat_range > 1, the resolution is '01s'.
            
            Otherwise, the resolution is '05m'.

        """

        if lon_range > 10 or lat_range > 10:
            return '01m'
        elif lon_range > 5 or lat_range > 5:
            return '15s'
        elif lon_range > 2 or lat_range > 2:
            return '03s'
        elif lon_range > 1 or lat_range > 1:
            return '01s'
        else:
            return '05m'

    @staticmethod
    def configure_colormap(colormap):
        """
        Adjusts the given colormap name for compatibility with PyGMT and matplotlib conventions.

        .. rubric :: Parameters

        ``colormap`` (`str`):
        The name of the colormap to be used. If the colormap name ends with '_r', the colormap is
        reversed, and the '_r' suffix is removed.

        .. rubric :: Returns

        ``colormap`` (`str`), ``cmap_reverse_flag`` (`bool`):
        A tuple containing the adjusted colormap name and a boolean indicating whether the colormap should
        be reversed.

        .. note :: 

            The method accept only colormaps that are available in PyGMT. For a list of available colormaps,

        .. rubric :: Example

        >>> colormap, reverse = PyGMTUtilities.configure_colormap('viridis_r')
        >>> print(colormap)
        viridis
        >>> print(reverse)
        True
        """
        cmap_reverse_flag = True if colormap.endswith('_r') else False
        colormap = colormap[:-2] if cmap_reverse_flag else colormap
        return colormap, cmap_reverse_flag

    @staticmethod
    def prepare_latex_annotations(label):
        """
        Prepares LaTeX annotations for plotting. Uses HTML tags instead 
        of $•$ for compatibility with PyGMT/GMT.

        .. rubric :: Parameters

        ``label`` (`str`):
        The LaTeX label to be prepared.

        .. rubric :: Returns

        ``str``:
        The prepared label.

        """

        parts = label.split('$')
        if len(parts) == 1:  # No '$' found
            return label
        new_text = ''
        for i, part in enumerate(parts):
            if i % 2 == 0:
                new_text += part
            else:
                new_text += f"<math>{part}</math>"
        return new_text
     
    @staticmethod
    def get_header(label, origin, filename, process = None):
        """
        Generates a header for a plot based on the provided parameters.

        .. rubric :: Parameters

        ``label`` (`str`):
        The label for the plot. Usually defined in the frontend function.

        ``origin`` (mtuq.Origin):
        mtuq.event.Origin object, used to retrieve the event time and depth. 

        ``filename`` (str):
        The filename of the plot. Defined by default the high-level function. Used to retrieve the component.

        ``process`` (Process, optional):
        mtuq.process_data.ProcessData object for appropriate dataset.

        .. rubric :: Returns

        ``list``:
        A list containing two lines of the header. [Label - (component)], [Event Time: (time) UTC, Depth: (depth) km]
        """
        if process is not None:
            # get type of waves used for the window
            window_type = process.window_type
            if window_type == 'surface_wave' or window_type == 'group_velocity':
                window_type = 'Surface wave'
            elif window_type == 'body_wave':
                window_type = 'Body wave'

        component = filename.split('/')[-1].split('.')[0]
        origin_time = str(origin.time)[0:19]
        origin_depth = origin.depth_in_m/1000

        label = PyGMTUtilities.prepare_latex_annotations(label)

        # if window_type exists, define Rayleigh or Love wave
        if process is not None:
            if window_type == 'Surface wave' and component == 'Z' or window_type == 'Surface wave' and component == 'R':
                # First line of the header defined as: label - Rayleigh wave (component)
                header_line_1 = f"{label} - Rayleigh wave ({component})"
            elif window_type == 'Surface wave' and component == 'T':
                # First line of the header defined as: label - Love wave (component)
                header_line_1 = f"{label} - Love wave ({component})"
            elif window_type == 'Body wave':
                # First line of the header defined as: label - (component)
                header_line_1 = f"{label} - Body wave ({component})"
        else:
            # First line of the header defined as: label - (component)
            header_line_1 = f"{label} - ({component})"

        header_line_2 = f"Event Time: {origin_time} UTC, Depth: {origin_depth:.1f} km"

        return [header_line_1, header_line_2]
    
    @staticmethod
    def draw_coastlines(fig, area_thresh=100, water_color='paleturquoise', water_transparency=55):
        """
        Draws coastlines and fills water areas with a transparent blue shade.

        .. rubric :: Parameters

        ``fig`` (pygmt.Figure): 
        The PyGMT figure object to which the coastlines and water areas will be added.

        ``area_thresh`` (`int`, optional):
        The minimum area of land to be displayed. Defaults to 100.

        ``water_color`` (`str`, optional):
        The color of the water areas. Defaults to 'paleturquoise'.

        ``water_transparency`` (`int`, optional):
        The transparency of the water areas. Defaults to 55.

        """        
        fig.coast(shorelines=True, area_thresh=area_thresh)
        fig.coast(shorelines=False, water=water_color, transparency=water_transparency, area_thresh=area_thresh)
