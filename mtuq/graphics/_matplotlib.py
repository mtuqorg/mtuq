import numpy as np
import matplotlib.pyplot as plt
from mtuq.util.beachball import lambert_azimuthal_equal_area_projection


def plot_force_matplotlib(filename=None, force_dict=None, ax=None, show=False):
    """ Plots force orientation for waveform figure header using matplotlib

    .. rubric :: Parameters

    ``filename`` (`str`)
    Name of output image file (optional if ax is provided)

    ``force_dict`` (`dict`):
    Dictionary containing force orientation parameters

    ``ax`` (matplotlib.axes.Axes):
    Optional. If provided, plot directly on this axes.

    ``show`` (bool):
    If True, call plt.show() at the end (only if ax is not provided)
    """

    fig = None
    close_fig = False

    # Default sizes for file output
    textsize = 40
    markersize = 28
    linewidth_major = 2
    linewidth_minor = 1

    if ax is None:
        # Only use pixel/dpi logic when saving to file
        pixel_width = 1807
        pixel_height = 1806
        dpi = 300  # You can adjust this if needed
        fig_width_in = pixel_width / dpi
        fig_height_in = pixel_height / dpi
        fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=dpi)
        close_fig = True
    else:
        # For inset axes, scale down for better appearance
        # Use axes size to scale font/marker sizes
        bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
        width_in, height_in = bbox.width, bbox.height
        # Use a base scaling factor (tweak as needed)
        scale = min(width_in, height_in)
        textsize = max(8, int(10 * scale))
        markersize = max(4, int(5 * scale))
        linewidth_major = max(0.5, 0.7 * scale)
        linewidth_minor = max(0.3, 0.4 * scale)

    ax.set_xlim(-1.01, 1.01)
    ax.set_ylim(-1.01, 1.01)
    ax.axis('off')
    ax.set_aspect('equal')

    # Minor arcs
    _plot_arc([-90, 90], [0, 0], ax, major=False, linewidth=linewidth_minor)
    _plot_arc([0, 0], [-90, 90], ax, major=False, linewidth=linewidth_minor)
    _plot_arc([-90/2, -90/2], [-90, -7.5], ax, major=False, linewidth=linewidth_minor)
    _plot_arc([-90/2, -90/2], [90, 7.5], ax, major=False, linewidth=linewidth_minor)
    _plot_arc([+90/2, +90/2], [-90, -7.5], ax, major=False, linewidth=linewidth_minor)
    _plot_arc([+90/2, +90/2], [90, 7.5], ax, major=False, linewidth=linewidth_minor)
    _plot_arc([-90, 90], [-30, -30], ax, major=False, linewidth=linewidth_minor)
    _plot_arc([-90, 90], [30, 30], ax, major=False, linewidth=linewidth_minor)
    _plot_arc([-90, 90], [-60, -60], ax, major=False, linewidth=linewidth_minor)
    _plot_arc([-90, 90], [60, 60], ax, major=False, linewidth=linewidth_minor)
    # Major arcs (rim of the circle)
    _plot_arc([-90, -90], [-90, 90], ax, major=True, linewidth=linewidth_major)
    _plot_arc([90, 90], [-90, 90], ax, major=True, linewidth=linewidth_major)

    # Projecting and plotting the text labels
    x_w, y_w = _lambert_azimuthal_equal_area(-90/2, 0)
    x_e, y_e = _lambert_azimuthal_equal_area(90/2, 0)
    ax.text(x_w, y_w, 'W', fontsize=textsize, ha='center', va='center', color='black')
    ax.text(x_e, y_e, 'E', fontsize=textsize, ha='center', va='center', color='black')

    # Plot force orientation -- black diamond piercing point
    lat = np.degrees(np.pi/2 - np.arccos(force_dict['h']))
    lon = _wrap(force_dict['phi'] + 90.)/2 # -- recalling 0 is east, we add 90 to shift from south (center) to east

    x, y = _lambert_azimuthal_equal_area(lon, lat)
    ax.plot(x, y, 'D', c='k', markersize=markersize)

    if close_fig:
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0)
        if show:
            plt.show()
        plt.close(fig)

def _wrap(angle_in_deg):
    """ Wraps angle to (-180, 180)
    """
    angle_in_deg %= 360.
    if angle_in_deg > 180.:
        angle_in_deg -= 360.
    return angle_in_deg

def _lambert_azimuthal_equal_area(lon, lat):
    """ Converts longitude and latitude to x, y coordinates in the Lambert Azimuthal Equal Area projection

    .. rubric :: Required arguments
    
        ``lon`` (`float`):
        Longitude in degrees

        ``lat`` (`float`):
        Latitude in degrees

    .. note ::
        
            This function is used internally by the `plot_force_matplotlib` function
            It produces a circle only for points within -90, 90 longitude and -90, 90 latitude.
            For points outside this range, they will land outside the circle.
    """
    # Radius modifyier so that the map is centered on 0 and is radius 1
    R = np.sqrt(2)/2
    # Define the central longitude and latitude for the projection (in radians)
    lambda_0 = 0.0
    phi_0 = 0.0

    lon = np.radians(lon)
    lat = np.radians(lat)
    k = np.sqrt(2 / (1 + np.sin(phi_0) * np.sin(lat) + np.cos(phi_0) * np.cos(lat) * np.cos(lon - lambda_0)))
    x = R * k * np.cos(lat) * np.sin(lon - lambda_0)
    y = R * k * (np.cos(phi_0) * np.sin(lat) - np.sin(phi_0) * np.cos(lat) * np.cos(lon - lambda_0))
    return x, y


def _plot_arc(lons_extent, lats_extent, ax, npts=100, major=True, linewidth=1):
    # Generate the points along the arc
    lon_arc = np.linspace(lons_extent[0], lons_extent[1], npts)
    lat_arc = np.linspace(lats_extent[0], lats_extent[1], npts)

    x_arc, y_arc = _lambert_azimuthal_equal_area(lon_arc, lat_arc)
    if major:
        ax.plot(x_arc, y_arc, 'k-', linewidth=linewidth)
    else:
        ax.plot(x_arc, y_arc, c='gray', linewidth=linewidth)