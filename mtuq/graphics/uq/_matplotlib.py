
import numpy as np
import math 

from matplotlib import pyplot
from os.path import exists
from xarray import DataArray

from mtuq.graphics.uq import _nothing_to_plot
from mtuq.graphics._gmt import read_cpt, _cpt_path
from mtuq.graphics.uq._gmt import _parse_vw, _parse_lune_array, _parse_force
from matplotlib import ticker
from matplotlib.colors import BoundaryNorm
import matplotlib.patheffects as path_effects
from mtuq.util.math import wrap_180, to_delta_gamma, to_mij
from mpl_toolkits.axes_grid1 import make_axes_locatable

import warnings

def cbformat(st, pos):
    return '{:.1E}'.format(st)

# Define a pure matplotlib backend as an alternative to GMT
# It should behave as similarly as possible to the GMT backend 
# and take the same input arguments
def _plot_lune_matplotlib(filename, longitude, latitude, values, 
    best_vw=None, lune_array=None, colormap='viridis', plot_type='contour', clip_interval=[0,100],**kwargs):

    """ Plots DataArray values on the eigenvalue lune (requires matplotlib)

    .. rubric :: Keyword arguments
    filename : str
        Name of output image file

    longitude : array_like (xarray.DataArray or numpy.ndarray)
        Array of longitudes

    latitude : array_like (xarray.DataArray or numpy.ndarray)
        Array of latitudes

    values : array_like (xarray.DataArray or numpy.ndarray)
        Array of values

    best_vw : list
        List of two floats representing the best-fitting eigenvalues

    lune_array : array_like (xarray.DataArray or numpy.ndarray)
        Used to plot beachball tradeoffs on the lune

    colormap : str
        Name of colormap

    title : str
        Title of plot

    plot_type : str
        Type of plot. Can be either contour, colormesh or scatter

    """

    if _nothing_to_plot(values):
        return
    
    if not isinstance(clip_interval, list) or len(clip_interval) != 2:
        raise Exception('clip_interval must be a list of two floats')

    _development_warning()

    # Check plot_type. Can be either contour or colormesh
    if plot_type not in ['contour', 'colormesh', 'scatter']:
        raise Exception('plot_type must be either contour or colormesh')

    fig, ax = _generate_lune()

    # Transform data to Hammer projection
    # Create a grid for pcollormesh from longitude and latitude arrays
    x, y = np.meshgrid(longitude, latitude)
    x, y = _hammer_projection(x, y)

    # Plot data
    # Use the percentile method to filter out outliers based on clip_interval (in percentage)
    if plot_type == 'colormesh':
        vmin, vmax = np.nanpercentile(np.asarray(values), clip_interval)
        im = ax.pcolormesh(x, y, values, cmap=colormap, vmin=vmin, vmax=vmax, shading='nearest', zorder=10)
    elif plot_type == 'contour':
        vmin, vmax = np.nanpercentile(np.asarray(values), clip_interval)
        im = None
        try:
            im = ax.contourf(x, y, values, cmap=colormap, levels=20, zorder=10, vmin=vmin, vmax=vmax)
        except Exception:
            # Fallback on colormesh if contour fails
            print('Not enough data to plot contour, switching to colormesh...')
            im = ax.pcolormesh(
                x, y, values, cmap=colormap, vmin=vmin, vmax=vmax, shading='nearest', zorder=10)
    elif plot_type == 'scatter':
        # Prepare colormap
        boundaries = np.linspace(0, 5, 6)
        norm = BoundaryNorm(boundaries, ncolors=256, clip=False)
        # Plot using scatter
        im = ax.scatter(x, y, c=values, cmap='Spectral_r', norm=norm, zorder=100)


    # Plot best-fitting moment tensor
    if best_vw is not None:
        best_vw = _parse_vw(best_vw)
        gamma, delta = _hammer_projection(
            best_vw[0], best_vw[1])
        if plot_type not in ['scatter']:
            _add_marker(ax, (gamma, delta))

    # Plot tradeoffs
    if lune_array is not None:
        from mtuq.graphics.beachball import _plot_beachball_matplotlib
        v, w = lune_array[:,1], lune_array[:,2]
        delta, gamma = to_delta_gamma(v, w)
        lon_lats=np.vstack([gamma, delta]).T
        _plot_beachball_matplotlib(None, to_mij(*lune_array.T), lon_lats=lon_lats,
                                    fig=fig, ax=ax, color='black', lune_rotation=True, scale=0.3)
            # ax.scatter(delta, gamma)

    # Set axis limits
    ax.set_xlim(-30.5, 30.5)
    ax.set_ylim(-90, 90)

    # Set axis labels
    ax.set_xlabel('Gamma')
    ax.set_ylabel('Delta')

    # if plot type is colormesh or contour:
    if plot_type in ['colormesh', 'contour']:
        divider = make_axes_locatable(pyplot.gca())
        cax = divider.append_axes("bottom", '2%', pad=0.001)
        cb = pyplot.colorbar(im, cax=cax, orientation='horizontal', ticks=ticker.MaxNLocator(nbins=5))

    elif plot_type == 'scatter':
        divider = make_axes_locatable(pyplot.gca())
        cax = divider.append_axes("bottom", '2%', pad=0.001)
        cb = pyplot.colorbar(im, cax=cax, orientation='horizontal', ticks=boundaries, extend='max')
        cb.set_label('Mismatching polarities')

    # Set the title if provided
    if 'title' in kwargs:
        ax.set_title(kwargs['title'])

    # Save figure
    pyplot.tight_layout()
    pyplot.subplots_adjust(left=0.1,right=0.9,top=0.95, bottom=0.08)
    pyplot.savefig(filename, dpi=300)
    pyplot.close()

def _plot_force_matplotlib(filename, phi, h, values, best_force=None, colormap='viridis', 
                           title=None, plot_type='contour', clip_interval=[0,100], **kwargs):
    """ Plots DataArray values on the force sphere (requires matplotlib)

    .. rubric :: Keyword arguments
    filename : str
        Name of output image file

    phi : array_like (xarray.DataArray or numpy.ndarray)
        Array of longitudes centered eastward, increasing anticlockwise (E = O°, N=90°, W=180°, S=270°)

    h : array_like (xarray.DataArray or numpy.ndarray)
        Array of colatitudes (from -1 to 1)

    values : array_like (xarray.DataArray or numpy.ndarray)
        Array of values

    best_force : list
        List of two floats representing the best-fitting force direction

    colormap : str
        Name of colormap

    title : str
        Title of plot

    plot_type : str
        Type of plot. Can be either contour, colormesh or scatter

    """

    if _nothing_to_plot(values):
        return

    if not isinstance(clip_interval, list) or len(clip_interval) != 2:
        raise Exception('clip_interval must be a list of two floats')

    _development_warning()

    # Check plot_type. Can be either contour or colormesh
    if plot_type not in ['contour', 'colormesh', 'scatter']:
        raise Exception('plot_type must be either contour or colormesh')

    fig, ax = _generate_sphere()

    # Transform data to Hammer projection
    # Create a grid for pcollormesh from longitude and latitude arrays
    latitude = np.degrees(np.pi/2 - np.arccos(h))
    longitude = wrap_180(phi + 90)
    x, y = np.meshgrid(longitude, latitude)
    x, y = _hammer_projection(x, y)

    if plot_type == 'colormesh':
        # A tweak is needed for plotting quad-based colormesh, as the longitude axis doesn't increase monotically 
        # Due to the wrap-around (0 deg centered on the easet), it is easier to split the array in two and 
        # plot them separately

        # Find the index of the discontinuity, where the longitude goes from 180 to -180
        discontinuity_index = np.where(longitude < 0)[0][0]

        # Split the arrays at the discontinuity
        lon1 = longitude[:discontinuity_index]
        lon2 = longitude[discontinuity_index:]
        values1 = values[:, :discontinuity_index]
        values2 = values[:, discontinuity_index:]

        # Mesh, project onto the Hammer projection and plot
        x1, y1 = np.meshgrid(lon1, latitude)
        x2, y2 = np.meshgrid(lon2, latitude)
        x1, y1 = _hammer_projection(x1, y1)
        x2, y2 = _hammer_projection(x2, y2)
        vmin, vmax = np.nanpercentile(np.asarray(values), clip_interval)
        im = ax.pcolormesh(x1, y1, values1, cmap=colormap, vmin=vmin, vmax=vmax, shading='auto', zorder=10)
        im = ax.pcolormesh(x2, y2, values2, cmap=colormap, vmin=vmin, vmax=vmax, shading='auto', zorder=10)

    elif plot_type == 'contour':
        # Plot using contourf
        levels = 20
        mask = ~np.isfinite(values)
        x_masked = x[~mask]
        y_masked = y[~mask]
        values_masked = values[~mask]
        im = ax.tricontourf(x_masked.flatten(), y_masked.flatten(), values_masked.flatten(), cmap=colormap, levels=levels, zorder=10)
    elif plot_type == 'scatter':
        # Prepare colormap
        boundaries = np.linspace(0, 5, 6)
        norm = BoundaryNorm(boundaries, ncolors=256, clip=False)
        # Plot using scatter
        im = ax.scatter(x, y, c=values, cmap='Spectral_r', norm=norm, zorder=100)

    # Plot best-fitting force orientation
    if best_force is not None:
        best_force = _parse_force(best_force)
        x_best, y_best = _hammer_projection(
            best_force[0], best_force[1])
        if plot_type not in ['scatter']:
            _add_marker(ax, (x_best, y_best))

    # Set axis limits
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)

    # Set axis labels
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # if plot type is colormesh or contour:
    if plot_type in ['colormesh', 'contour']:
        cbmin, cbmax = im.get_clim()
        ticks = np.linspace(cbmin, cbmax, 3)
        cb = pyplot.colorbar(im, ticks=ticks, location='bottom', ax=ax, pad=0.001, fraction=0.02)
        if 'colorbar_label' in kwargs:
            cb.set_label(kwargs['colorbar_label'])

    elif plot_type == 'scatter':
        cb = pyplot.colorbar(im, location='bottom', ax=ax, pad=0.001, fraction=0.02, ticks=boundaries, extend='max')
        cb.set_label('Mismatching polarities')

    # Set title
    if title is not None:
        ax.set_title(title)
        
    # Save figure
    pyplot.tight_layout()
    pyplot.savefig(filename, dpi=300, bbox_inches='tight')
    pyplot.close()


def _plot_dc_matplotlib(filename, coords, 
    values_h_kappa, values_sigma_kappa, values_sigma_h, colormap='viridis',
    title=None, best_dc=None, plot_colorbar=True, figsize=(8., 8.), fontsize=14, clip_interval=[0,100],**kwargs):

    if not isinstance(clip_interval, list) or len(clip_interval) != 2:
        raise Exception('clip_interval must be a list of two floats')

    _development_warning()

    # Prepare axes
    fig, axes = pyplot.subplots(2, 2,
        figsize=figsize,
    )

    pyplot.subplots_adjust(
        wspace=0.4,
        hspace=0.4,
        right=0.88
    )

    # Parse colormap
    if exists(_cpt_path(colormap)):
       colormap = read_cpt(_cpt_path(colormap))

    # note the following parameterization details
    #     kappa = strike
    #     sigma = slip
    #     h = cos(dip)

    vals = np.append(np.append(values_sigma_kappa.ravel(), values_sigma_kappa.ravel()),(values_sigma_h.ravel()))
    # Plot data
    # Use the percentile method to filter out outliers (Will alway clip the 10% greater values)
    vmin, vmax = np.nanpercentile(vals, clip_interval)

    # plot surfaces
    im0 = _pcolor(axes[0][0], coords['h'], coords['kappa'], values_h_kappa, colormap, vmin=vmin, vmax=vmax)

    _pcolor(axes[0][1], coords['sigma'], coords['kappa'], values_sigma_kappa, colormap, vmin=vmin, vmax=vmax)

    _pcolor(axes[1][1], coords['sigma'], coords['h'], values_sigma_h, colormap, vmin=vmin, vmax=vmax)

    # optional markers
    if best_dc:
        _kappa, _sigma, _h = best_dc
        _add_marker(axes[0][0], (_h, _kappa))
        _add_marker(axes[0][1], (_sigma, _kappa))
        _add_marker(axes[1][1], (_sigma, _h))

    _set_dc_labels(axes, fontsize=fontsize)

    if plot_colorbar:
        cbar_ax = fig.add_axes([0.91, 0.13, 0.0125, 0.27])  # [left, bottom, width, height]
        cb = pyplot.colorbar(im0, cax=cbar_ax, orientation='vertical')
        formatter = ticker.ScalarFormatter()
        formatter.set_powerlimits((-2, 2))  # Avoid scientific notation between 10^-2 and 10^2
        cb.ax.yaxis.set_major_formatter(formatter)
        if 'colorbar_label' in kwargs:
            cb.set_label(kwargs['colorbar_label'])


    # Set the title if provided
    if title:
        fig.suptitle(title, fontsize=fontsize + 2)

    pyplot.savefig(filename, dpi=300)
    pyplot.close()



def _plot_vw_matplotlib(filename, v, w, values, best_vw=None, lune_array=None, 
    colormap='viridis', title='', figsize=(3., 8.)):

    if _nothing_to_plot(values):
        return

    fig, ax = pyplot.subplots(figsize=figsize, constrained_layout=True)

    # pcolor requires corners of pixels
    corners_v = _centers_to_edges(v)
    corners_w = _centers_to_edges(w)

    # `values` gets mapped to pixel colors
    pyplot.pcolor(corners_v, corners_w, values, cmap=colormap)

    # v and w have the following bounds
    # (see https://doi.org/10.1093/gji/ggv262)
    pyplot.xlim([-1./3., 1./3.])
    pyplot.ylim([-3./8.*np.pi, 3./8.*np.pi])

    pyplot.xticks([], [])
    pyplot.yticks([], [])

    if exists(_cpt_path(colormap)):
       cmap = read_cpt(_cpt_path(colormap))

    if True:
        cbar = pyplot.colorbar(
            orientation='horizontal',
            pad=0.,
            )

        cbar.formatter.set_powerlimits((-2, 2))

    if title:
        fontdict = {'fontsize': 16}
        pyplot.title(title, fontdict=fontdict)


    if best_vw:
        pyplot.scatter(*best_vw, s=333,
            marker='o',
            facecolors='none',
            edgecolors=[0,1,0],
            linewidths=1.75,
            )

    pyplot.savefig(filename)
    pyplot.close()


def _plot_depth_matplotlib(filename, depths, values,
        magnitudes=None, lune_array=None,
        title=None, xlabel=None, ylabel=None, figsize=(6., 6.), fontsize=16.):
    
    _development_warning()

    fig, ax = pyplot.subplots(figsize=figsize)
    ax.plot(depths, values, 'k-')

    if title:
        pyplot.title(title, fontsize=fontsize)

    if xlabel:
         pyplot.xlabel(xlabel, fontsize=fontsize)

    if ylabel:
         pyplot.ylabel(ylabel, fontsize=fontsize)

    if magnitudes is not None:
        # add magnitude text labels to each point
        # String formating to display only x.xx values
        magnitudes = ['{:.2f}'.format(m) for m in magnitudes]
        for i, magnitude in enumerate(magnitudes):
            text = pyplot.text(depths[i], 
                        values[i]+0.045*(values.max()-values.min()), 
                        magnitude, fontsize=fontsize-6, ha='center', zorder=2000)
                    # Adding a white stroke around the text
            text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), 
                                   path_effects.Normal()])

    # Fit a parabola to the best-fit depth
    imin = np.argmin(values)
    
    if imin == 0:
        # Best fit at the left edge, use first three points
        x1, y1 = depths[0], values[0]
        x2, y2 = depths[1], values[1]
        x3, y3 = depths[2], values[2]
    elif imin == len(depths) - 1:
        # Best fit at the right edge, use last three points
        x1, y1 = depths[-3], values[-3]
        x2, y2 = depths[-2], values[-2]
        x3, y3 = depths[-1], values[-1]
    else:
        # Best fit not at an edge, use the point and its neighbors
        x1, y1 = depths[imin-1], values[imin-1]
        x2, y2 = depths[imin], values[imin]
        x3, y3 = depths[imin+1], values[imin+1]
    
    # Calculate coefficients for the parabola (y = ax^2 + bx + c)
    denom = (x1-x2)*(x1-x3)*(x2-x3)
    a = (x3*(y2-y1) + x2*(y1-y3) + x1*(y3-y2)) / denom
    b = (x3**2*(y1-y2) + x2**2*(y3-y1) + x1**2*(y2-y3)) / denom
    c = (x2*x3*(x2-x3)*y1 + x3*x1*(x3-x1)*y2 + x1*x2*(x1-x2)*y3) / denom
    

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Generate parabola points
    x_parabola = np.linspace(min(depths)-0.1*(max(depths)-min(depths)), 
                                max(depths)+0.1*(max(depths)-min(depths)), 100)

    y_parabola = a*x_parabola**2 + b*x_parabola + c

    best_fit_depth = -b / (2*a) 
    best_fit_value = a * best_fit_depth**2 + b * best_fit_depth + c
    print('Best fitting depth:', best_fit_depth)

    percentage = 0.05  # Define the percentage of the best fit value to use as a threshold
    delta_y = percentage * best_fit_value
    y_threshold = best_fit_value + delta_y

    # Step 3: Get the roots where the parabola equals the threshold
    discriminant = np.sqrt(b**2 - 4*a*(c - y_threshold))

    x1 = (-b + discriminant) / (2 * a)
    x2 = (-b - discriminant) / (2 * a)

    # Step 4: Calculate the uncertainty k
    k = abs(x2 - x1) / 2
    print('Uncertainty: ±', k)
    
    ax.hlines(y=y_threshold, xmin=x1, xmax=x2, colors='gray', linestyles='dashed', label='5% Threshold')
    ax.plot(best_fit_depth, ylim[0], marker='v', color='white', markersize=10, 
            markeredgecolor='black', markeredgewidth=2, clip_on=False)

    # Plot the parabola
    ax.plot(x_parabola, y_parabola, '--', color='gray')
    # ax.legend()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
            
    if lune_array is not None:
        from mtuq.graphics.beachball import _plot_beachball_matplotlib

        # Get current axis limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Pad the limits to make the beachballs fit better
        xlim = (xlim[0] - 0.05*(xlim[1]-xlim[0]), xlim[1] + 0.05*(xlim[1]-xlim[0]))
        ylim = (ylim[0] - 0.025*(ylim[1]-ylim[0]), ylim[1] + 0.025*(ylim[1]-ylim[0]))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # Normalize depths and values to fit within [-1, 1] based on axis limits
        depths_norm = 2 * (depths - xlim[0]) / (xlim[1] - xlim[0]) - 1
        values_norm = 2 * (values - ylim[0]) / (ylim[1] - ylim[0]) - 1
        
        lon_lats = np.vstack([depths_norm, values_norm]).T

        # Create a secondary axis with an even aspect ratio
        # This is a trick so that we'll always have round beahcball even though
        # the axes are always going to be very different scales.
        bbox = ax.get_position()
        even_ax = fig.add_axes(bbox, frameon=False)
        even_ax.set_xlim(-1, 1)
        even_ax.set_ylim(-1, 1)
        even_ax.set_aspect('equal')
        even_ax.axis('off')        

        _plot_beachball_matplotlib(None, to_mij(*lune_array.T), lon_lats=lon_lats,
                                   fig=fig, ax=even_ax, color='black', lune_rotation=False, scale=0.3)


    pyplot.savefig(filename, dpi=300, bbox_inches='tight')
    pyplot.close()

def _plot_latlon_matplotlib(filename, lon, lat, values, best_latlon=None, lune_array=None,
    **kwargs):

    if _nothing_to_plot(values):
        return
    
    _development_warning()
    
    data = np.column_stack((lon, lat, values))
    from scipy.interpolate import griddata, SmoothBivariateSpline

    lon_buffer = (lon.max() - lon.min())*0.1
    lat_buffer = (lat.max() - lat.min())*0.1

    # Create a slightly larger grid for extrapolation
    grid_lon, grid_lat = np.meshgrid(np.linspace(lon.min() - lon_buffer, lon.max() + lon_buffer, 100),
                                    np.linspace(lat.min() - lat_buffer, lat.max() + lat_buffer, 100))

    # Interpolation using SmoothBivariateSpline
    spline = SmoothBivariateSpline(x=lon, y=lat, z=values, s=3, kx=2, ky=2)
    grid_values = spline.ev(grid_lon, grid_lat)

    # Create the plot
    fig, ax = pyplot.subplots(figsize=(10, 8))

    # Contour plot
    contour = ax.contourf(grid_lon, grid_lat, grid_values, cmap='viridis', levels=20)

    # Create a divider for the existing axes instance
    divider = make_axes_locatable(ax)
    # Append axes to the right of ax, with the same height
    cax = divider.append_axes("right", size="4%", pad=0.2)

    # Add a color bar
    cbar = pyplot.colorbar(contour, cax=cax)
    if 'colorbar_label' in kwargs:
        cbar.set_label(kwargs['colorbar_label'])

    if lune_array is not None:
        from mtuq.graphics.beachball import _plot_beachball_matplotlib
        for lune in lune_array:
            lon_lats=np.vstack([lon, lat]).T
            _plot_beachball_matplotlib(None, to_mij(*lune_array.T), lon_lats=lon_lats,
                                        fig=fig, ax=ax, color='black', lune_rotation=False, scale=0.4)

    if best_latlon is not None:
        _add_marker(ax, best_latlon, size=800)

    # Set labels and title
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Contour Plot with Latitude and Longitude')
    ax.set_aspect('equal')

    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))

    pyplot.tight_layout()
    pyplot.savefig(filename, dpi=300, bbox_inches='tight')
    pyplot.close()


def _plot_omega_matplotlib(filename, omega, values,
    title=None, xlabel='Angular distance', ylabel=None, figsize=(6., 6.), fontsize=16.):

    pyplot.figure(figsize=figsize)
    pyplot.plot(omega, values, 'k-')

    pyplot.xlim([0., 180.])

    if title:
        pyplot.title(title, fontsize=fontsize)

    if xlabel:
         pyplot.xlabel(xlabel, fontsize=fontsize)

    if ylabel:
         pyplot.ylabel(ylabel, fontsize=fontsize)

    pyplot.savefig(filename)
    pyplot.close()

#
# utility functions
#

def _centers_to_edges(v):
    if issubclass(type(v), DataArray):
        v = v.values.copy()
    else:
        v = v.copy()

    dv = (v[1]-v[0])
    v -= dv/2
    v = np.pad(v, (0, 1))
    v[-1] = v[-2] + dv

    return v


def _add_marker(axis, coords, size=250):
    axis.scatter(*coords, s=size,
        marker='o',
        facecolors='none',
        edgecolors=[0,1,0],
        linewidths=1.75,
        clip_on=False,
        zorder=200,
        )


def _pcolor(axis, x, y, values, colormap, **kwargs):
    # workaround matplotlib compatibility issue
    try:
        axis.pcolor(x, y, values, cmap=colormap, shading='auto', **kwargs)
    except:
        axis.pcolor(x, y, values, cmap=colormap, **kwargs)

    # Return a mappable object
    return axis.collections[0]


def _set_dc_labels(axes, **kwargs):

    # note the following parameterization details
    #     kappa = strike
    #     sigma = slip
    #     h = cos(dip)

    kappa_ticks = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    kappa_ticklabels = ['0', '', '90', '', '180', '', '270', '', '360']

    sigma_ticks = [-90, -67.5, -45, -22.5, 0, 22.5, 45, 67.5, 90]
    sigma_ticklabels = ['-90', '', '-45', '', '0', '', '45', '', '90']

    h_ticks = [np.cos(np.radians(tick)) for tick in [0, 15, 30, 45, 60, 75, 90]]
    h_ticklabels = ['0', '', '30', '', '60', '', '90']

    # upper left panel
    axis = axes[0][0]
    axis.set_xlabel('Dip', **kwargs)
    axis.set_xticks(h_ticks)
    axis.set_xticklabels(h_ticklabels)
    axis.set_ylabel('Strike', **kwargs)
    axis.set_yticks(kappa_ticks)
    axis.set_yticklabels(kappa_ticklabels)

    # upper right panel
    axis = axes[0][1]
    axis.set_xlabel('Slip', **kwargs)
    axis.set_xticks(sigma_ticks)
    axis.set_xticklabels(sigma_ticklabels)
    axis.set_ylabel('Strike', **kwargs)
    axis.set_yticks(kappa_ticks)
    axis.set_yticklabels(kappa_ticklabels)

    # lower right panel
    axis = axes[1][1]
    axis.set_xlabel('Slip', **kwargs)
    axis.set_xticks(sigma_ticks)
    axis.set_xticklabels(sigma_ticklabels)
    axis.set_ylabel('Dip', **kwargs)
    axis.set_yticks(h_ticks)
    axis.set_yticklabels(h_ticklabels)

    # lower left panel
    axes[1][0].axis('off')

# Define the Hammer projection function for matplotlib backend
def _hammer_projection(lon, lat):
    lon = np.radians(lon)
    lat = np.radians(lat)
    alpha = np.sqrt(1 + np.cos(lat) * np.cos(lon / 2))
    x = (2 * np.sqrt(2) * np.cos(lat) * np.sin(lon / 2)) / alpha
    y = (np.sqrt(2) * np.sin(lat)) / alpha
    return np.degrees(x), np.degrees(y)


def _generate_lune(ax=None):
    if ax is None:
        fig = pyplot.figure(figsize=(2.5, 7.0))
        ax = fig.add_subplot(111)
        ax.set_axis_off()
    else:
        fig = ax.figure

    # Generate curved gridlines
    num_lines = 7
    lon_lines = np.linspace(-30, 30, num_lines)
    lat_lines = np.linspace(-90, 90, num_lines * 2 + 1)

    for lon_line in lon_lines:
        lat_line = np.linspace(-90, 90, 1000)
        x_line, y_line = _hammer_projection(np.full_like(lat_line, lon_line), lat_line)
        ax.plot(x_line, y_line, 'k--', linewidth=0.5, alpha=0.5)

    for lat_line in lat_lines:
        lon_line = np.linspace(-30, 30, 1000)
        x_line, y_line = _hammer_projection(lon_line, np.full_like(lon_line, lat_line))
        ax.plot(x_line, y_line, 'k--', linewidth=0.5, alpha=0.5)

    _plot_lune_arcs(ax, _compute_lune_arcs())
    # Transform data to Hammer projection

    return fig, ax

def _generate_sphere(ax=None):
    if ax is None:
        fig = pyplot.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        ax.set_axis_off()
    else:
        fig = ax.figure

    # Generate curved gridlines
    num_lines = 7
    lon_lines = np.linspace(-180, 180, num_lines * 2 + 1)
    lat_lines = np.linspace(-90, 90, num_lines*2+1)

    for lon_line in lon_lines:
        lat_line = np.linspace(-90, 90, 1000)
        x_line, y_line = _hammer_projection(np.full_like(lat_line, lon_line), lat_line)
        ax.plot(x_line, y_line, 'k--', linewidth=0.5, alpha=0.5)

    for lat_line in lat_lines:
        lon_line = np.linspace(-185, 185, 1000)
        x_line, y_line = _hammer_projection(lon_line, np.full_like(lon_line, lat_line))
        ax.plot(x_line, y_line, 'k--', linewidth=0.5, alpha=0.5)

    _plot_sphere_arcs(ax, _compute_sphere_arcs())
    _plot_directions_text(ax)
    # Transform data to Hammer projection

    return fig, ax

def _compute_center_of_minimum_distance(lon_a, lat_a, lon_b, lat_b, iterations):
    if iterations == 0:
        return [(lon_a, lat_a), (lon_b, lat_b)]
    
    # Convert coordinates to radians
    lon_a_rad = math.radians(lon_a)
    lat_a_rad = math.radians(lat_a)
    lon_b_rad = math.radians(lon_b)
    lat_b_rad = math.radians(lat_b)
    
    # Convert lat/lon to cartesian coordinates
    x_a = math.cos(lat_a_rad) * math.cos(lon_a_rad)
    y_a = math.cos(lat_a_rad) * math.sin(lon_a_rad)
    z_a = math.sin(lat_a_rad)
    
    x_b = math.cos(lat_b_rad) * math.cos(lon_b_rad)
    y_b = math.cos(lat_b_rad) * math.sin(lon_b_rad)
    z_b = math.sin(lat_b_rad)
    
    # Compute barycenter of the points in cartesian coordinates
    x_mid = (x_a + x_b) / 2
    y_mid = (y_a + y_b) / 2
    z_mid = (z_a + z_b) / 2
    
    # Convert cartesian coordinate to latitude and longitude for the midpoint
    lat_mid_rad = math.atan2(z_mid, math.sqrt(x_mid**2 + y_mid**2))
    lon_mid_rad = math.atan2(y_mid, x_mid)
    
    # Convert midpoint lat and lon from radians to degrees
    lat_mid = math.degrees(lat_mid_rad)
    lon_mid = math.degrees(lon_mid_rad)
    
    # Recursive calls
    coordinates = []
    coordinates.extend(_compute_center_of_minimum_distance(lon_a, lat_a, lon_mid, lat_mid, iterations - 1))
    coordinates.extend(_compute_center_of_minimum_distance(lon_mid, lat_mid, lon_b, lat_b, iterations - 1))
    
    return coordinates

def _compute_lune_arcs():
    arc_points = [
    [-30, 35.2644, 30, 54.7356],
    [-30, -54.7356, 30, -35.2644],
    [-30, 35.2644, 30, -35.2644]
]
    arcs = []
    for arc in arc_points:
        arcs.append(_compute_center_of_minimum_distance(arc[0], arc[1], arc[2], arc[3], 6))
    return arcs

def _plot_lune_arcs(axis, arcs):
    for arc in arcs:
        lon = [x[0] for x in arc]
        lat = [x[1] for x in arc]
        axis.plot(*_hammer_projection(lon, lat), color='silver', linewidth=1.5, zorder=100)

def _compute_sphere_arcs():
    arc_points = [
    [0, -90, 0, -10],
    [0, 10, 0, 90],
    [-90, -90, -90, -10],
    [-90, 10, -90, 90],
    [90, -90, 90, -10],
    [90, 10, 90, 90],
    [-180,-90,-180, 90],
    [180,-90,180, 90]
    ]

    arcs = []
    for arc in arc_points:
        arcs.append(_compute_center_of_minimum_distance(arc[0], arc[1], arc[2], arc[3], 6))
    return arcs

def _plot_sphere_arcs(axis, arcs):
    for arc in arcs:
        lon = [x[0] for x in arc]
        lat = [x[1] for x in arc]
        axis.plot(*_hammer_projection(lon, lat), color='silver', linewidth=1.5, zorder=100)
        
def _plot_directions_text(axis):
    axis.text(0,0,'S', verticalalignment='center', horizontalalignment='center', color='silver', zorder=100)
    axis.text(88,0,'E', verticalalignment='center', horizontalalignment='center', color='silver', zorder=100)
    axis.text(-88,0,'W', verticalalignment='center', horizontalalignment='center', color='silver', zorder=100)


def _development_warning():
    warnings.warn(
       """'\n'
    You are using the new matplotlib visualization backend, which is currently
    being tested. If you encounter issues or unexpected behavior, please report
    them on GitHub.
       """,
       UserWarning
       )
    print()
