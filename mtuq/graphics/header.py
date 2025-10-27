#
# graphics/header.py - figure headers and text
#

import numpy as np
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from matplotlib import pyplot
from matplotlib.font_manager import FontProperties
try:
    from matplotlib.textpath import TextPath
except ImportError:  
    TextPath = None
from mtuq.graphics.beachball import plot_beachball, _plot_beachball_matplotlib
from mtuq.graphics._pygmt import exists_pygmt, plot_force_pygmt
from mtuq.graphics._matplotlib import plot_force_matplotlib
from mtuq.util.math import to_delta_gamma


# =============================================================================
# Constants and Configuration
# =============================================================================

# Header layout tuning parameters
HEADER_TOP_MARGIN = 0.10  # fraction of header height
HEADER_TEXT_LEFT_MARGIN = 0.20  # fraction of header height
HEADER_TEXT_FONT_SIZE = 16
HEADER_TEXT_VSPACE = 0.32  # vertical space between lines (in data units)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class HeaderStyle:
    """Style configuration for header rendering."""
    top_margin: float = HEADER_TOP_MARGIN
    text_left_margin: float = HEADER_TEXT_LEFT_MARGIN
    font_size: int = HEADER_TEXT_FONT_SIZE
    text_vspace: float = HEADER_TEXT_VSPACE
    text_right_margin: float = HEADER_TEXT_LEFT_MARGIN
    min_font_size: int = 10


@dataclass
class BaseHeaderInfo:
    """Base class for header information with common fields."""
    event_name: str
    latlon: str
    depth_label: str
    depth_str: str
    model: str
    solver: str
    norm: str
    best_misfit: float
    passband_line: str
    N_total: int
    N_p_used: int
    N_s_used: int


@dataclass
class MomentTensorHeaderInfo(BaseHeaderInfo):
    """Moment tensor specific header information."""
    magnitude: float
    tensor_coords_line: str
    mt: any
    lune_dict: dict


@dataclass
class ForceHeaderInfo(BaseHeaderInfo):
    """Force inversion specific header information."""
    F0: float
    phi_theta_line: str
    force_dict: dict


# =============================================================================
# Abstract Base Classes
# =============================================================================

class HeaderBlock(ABC):
    """Abstract base class for all header blocks."""
    
    @abstractmethod
    def render(self, ax, info, px: float, py: float,
               *, height: float, width: float, margin_left: float, margin_top: float,
               style: HeaderStyle) -> float:
        """Render block on ax at (px, py). Returns new py after rendering.
        """
        pass


# =============================================================================
# Concrete Header Block Classes
# =============================================================================

class TextBlock(HeaderBlock):
    """A text block for displaying formatted text in headers."""
    
    def __init__(self, text: str, fontsize=HEADER_TEXT_FONT_SIZE, bold=False, 
                 italic=False, vspace=HEADER_TEXT_VSPACE, auto_shrink=True,
                 min_fontsize=None, **kwargs):
        self.text = text
        self.fontsize = fontsize
        self.bold = bold
        self.italic = italic
        self.vspace = vspace
        self.auto_shrink = auto_shrink
        self.min_fontsize = min_fontsize
        self.kwargs = kwargs
    
    def render(self, ax, info, px, py, *, height: float, width: float, 
               margin_left: float, margin_top: float, style: HeaderStyle):
        font = FontProperties()
        if self.bold:
            font.set_weight('bold')
        if self.italic:
            font.set_style('italic')

        # Prefer explicit style defaults when block-level values are not set
        fontsize = self.fontsize or style.font_size
        vspace = self.vspace or style.text_vspace

        text_value = self.text.format(**info.__dict__)

        # Compute available width in axis units and shrink fonts if required
        max_width = max(width - px - style.text_right_margin * height, 0.)
        target_min = self.min_fontsize if self.min_fontsize is not None else style.min_font_size

        if self.auto_shrink and max_width > 0 and TextPath is not None:
            font_for_metrics = font.copy()
            font_for_metrics.set_size(fontsize)
            try:
                text_path = TextPath((0, 0), text_value, prop=font_for_metrics)
                text_width = text_path.get_extents().width / 72.0
            except Exception:
                text_width = 0.0

            if text_width > max_width and text_width > 0.0:
                scale = max_width / text_width
                fontsize = max(target_min, fontsize * scale)

        font.set_size(fontsize)

        # Use attribute access for info container
        ax.text(px, py, text_value, fontproperties=font, fontsize=fontsize, 
                **self.kwargs)
        return py - vspace


class MomentTensorFigureBlock(HeaderBlock):
    """Block for displaying moment tensor beachball figures."""
    
    def __init__(self, diameter_scale=0.75, backend=_plot_beachball_matplotlib):
        self.diameter_scale = diameter_scale
        self.backend = backend
    
    def render(self, ax, info, px, py, *, height: float, width: float, 
               margin_left: float, margin_top: float, style: HeaderStyle):
        mt = info.mt
        diameter = self.diameter_scale * height
        xp = margin_left
        yp = 0.075 * height

        if self.backend == _plot_beachball_matplotlib:
            # Direct matplotlib rendering
            inset_ax = ax.inset_axes([xp, yp, diameter, diameter], transform=ax.transData)
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            inset_ax.set_frame_on(False)
            plot_beachball(None, mt, None, None, fig=inset_ax.figure, ax=inset_ax, backend=self.backend)
            inset_ax.axis('off')
        else:
            # File-based rendering for other backends
            self._render_via_file(ax, mt, xp, yp, diameter)

        return py  # no vertical change
    
    def _render_via_file(self, ax, mt, xp, yp, diameter):
        """Helper method for file-based rendering."""
        plot_beachball('tmp.png', mt, None, None, backend=self.backend)
        img = pyplot.imread('tmp.png')
        self._cleanup_temp_files()
        ax.imshow(img, extent=(xp, xp+diameter, yp, yp+diameter))
    
    def _cleanup_temp_files(self):
        """Clean up temporary files."""
        for filename in ['tmp.png', 'tmp.ps']:
            try:
                os.remove(filename)
            except OSError:
                pass


class ForceImageBlock(HeaderBlock):
    """Block for displaying force vector figures."""
    
    def __init__(self, diameter_scale=0.8, backend=plot_force_matplotlib):
        self.diameter_scale = diameter_scale
        self.backend = backend
    
    def render(self, ax, info, px, py, *, height: float, width: float, 
               margin_left: float, margin_top: float, style: HeaderStyle):
        force_dict = info.force_dict
        diameter = self.diameter_scale * height
        xp = margin_left
        yp = (height - diameter) / 2 - 0.1 * diameter

        if self.backend == plot_force_matplotlib:
            # Direct matplotlib rendering
            inset_ax = ax.inset_axes([xp, yp, diameter, diameter], transform=ax.transData)
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            inset_ax.set_frame_on(False)
            plot_force_matplotlib(force_dict=force_dict, ax=inset_ax)
            inset_ax.axis('off')
        else:
            # File-based rendering for other backends
            if self.backend == plot_force_pygmt and not exists_pygmt():
                return py
            self._render_via_file(ax, force_dict, xp, yp, diameter)

        return py
    
    def _render_via_file(self, ax, force_dict, xp, yp, diameter):
        """Helper method for file-based rendering."""
        self.backend('tmp.png', force_dict)
        img = pyplot.imread('tmp.png')
        self._cleanup_temp_files()
        ax.imshow(img, extent=(xp, xp+diameter, yp, yp+diameter))
    
    def _cleanup_temp_files(self):
        """Clean up temporary files."""
        for filename in ['tmp.png', 'tmp.ps']:
            try:
                os.remove(filename)
            except OSError:
                pass


# =============================================================================
# Header Container Class
# =============================================================================

class Header:
    """Container for header blocks. Handles axis creation and rendering."""
    
    def __init__(self, blocks: List[HeaderBlock], top_margin=HEADER_TOP_MARGIN, 
                 text_left_margin=HEADER_TEXT_LEFT_MARGIN):
        self.blocks = blocks
        self.top_margin = top_margin
        self.text_left_margin = text_left_margin
    
    def _get_axis(self, height, fig=None):
        if hasattr(self, '_axis'):
            return self._axis
        
        if fig is None:
            fig = pyplot.gcf()
        
        width, figure_height = fig.get_size_inches()
        assert height < figure_height, "Header height exceeds entire figure height. Please double check input arguments."
        
        x0 = 0.
        y0 = 1. - height / figure_height
        self._axis = fig.add_axes([x0, y0, 1., height / figure_height])
        
        ax = self._axis
        ax.set_xlim([0., width])
        ax.set_ylim([0., height])
        
        # Hide all spines and ticks
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
        return ax
    
    def write(self, height, width, margin_left, margin_top, info):
        ax = self._get_axis(height)
        
        # Build a style object (use existing per-header overrides if present)
        style = getattr(self, 'style', None)
        if style is None:
            style = HeaderStyle(top_margin=self.top_margin, text_left_margin=self.text_left_margin)
            self.style = style

        # Compute text start positions using explicit rendering params
        px = 0.68 * height + margin_left + style.text_left_margin * height  # left margin for text
        py = height - margin_top - style.top_margin * height  # top margin above first line

        # Pass rendering params explicitly to blocks; do NOT mutate the provided info
        for block in self.blocks:
            py = block.render(ax, info, px, py, height=height, width=width,
                              margin_left=margin_left, margin_top=margin_top, style=style)


# =============================================================================
# Utility Functions
# =============================================================================

def format_lat_lon(origin):
    """Format latitude and longitude from origin object."""
    if origin.latitude >= 0:
        latlon = '%.2f%s%s' % (+origin.latitude, u'\N{DEGREE SIGN}', 'N')
    else:
        latlon = '%.2f%s%s' % (-origin.latitude, u'\N{DEGREE SIGN}', 'S')
    if origin.longitude > 0:
        latlon += '% .2f%s%s' % (+origin.longitude, u'\N{DEGREE SIGN}', 'E')
    else:
        latlon += '% .2f%s%s' % (-origin.longitude, u'\N{DEGREE SIGN}', 'W')
    return latlon


def format_focal_mechanism(lune_dict):
    """Format focal mechanism parameters from lune dictionary."""
    strike = lune_dict['kappa']
    try:
        dip = np.degrees(np.arccos(lune_dict['h']))
    except:
        dip = lune_dict['theta']
    slip = lune_dict['sigma']
    return f"strike  dip  slip:  {strike:.0f}  {dip:.0f}  {slip:.0f}"


def format_gamma_delta(lune_dict):
    """Format gamma and delta coordinates from lune dictionary."""
    try:
        v, w = lune_dict['v'], lune_dict['w']
        delta, gamma = to_delta_gamma(v, w)
    except:
        delta, gamma = lune_dict['delta'], lune_dict['gamma']
    return f'lune coords γ  δ:  {gamma:.0f}  {delta:.0f}'


def format_phi_theta(force_dict):
    """Format phi and theta angles from force dictionary."""
    try:
        phi, theta = force_dict['phi'], force_dict['theta']
    except:
        phi, h = force_dict['phi'], force_dict['h']
        theta = np.degrees(np.arccos(h))
    return f'φ  θ:  {phi:.0f}  {theta:.0f}'


def _format_depth_string(origin):
    """Format depth string from origin object."""
    depth_in_m = origin.depth_in_m
    depth_in_km = depth_in_m / 1000.
    
    if depth_in_m >= 0:
        if depth_in_m < 1000.:
            depth_str = f'{depth_in_m:.0f} m'
        elif depth_in_km <= 100.:
            depth_str = f'{depth_in_km:.1f} km'
        else:
            depth_str = f'{depth_in_km:.0f} km'
        depth_label = 'Depth'
    else:
        height_in_m = abs(depth_in_m)
        height_in_km = abs(depth_in_km)
        if height_in_m < 1000.:
            depth_str = f'{height_in_m:.0f} m'
        elif height_in_km <= 100.:
            depth_str = f'{height_in_km:.1f} km'
        else:
            depth_str = f'{height_in_km:.0f} km'
        depth_label = 'Height'
    
    return depth_label, depth_str


def _format_passband_line(process_bw, process_sw, process_sw_supp=None):
    """Format passband line from processing objects."""
    if process_bw and process_sw and process_sw_supp:
        return ('body waves: %s (%.1f s), Rayleigh: %s (%.1f s), Love: %s (%.1f s)' %
            (_passband_formatting(process_bw), process_bw.window_length,
             _passband_formatting(process_sw), process_sw.window_length,
             _passband_formatting(process_sw_supp), process_sw_supp.window_length))
    elif process_bw and process_sw:
        return ('body waves:  %s (%.1f s),  surface waves: %s (%.1f s)' %
            (_passband_formatting(process_bw), process_bw.window_length,
             _passband_formatting(process_sw), process_sw.window_length))
    elif process_sw:
        return ('passband: %s,  window length: %.1f s ' % 
            (_passband_formatting(process_sw), process_sw.window_length))
    else:
        return ''


def _calculate_best_misfit(best_misfit_bw, best_misfit_sw, best_misfit_sw_supp=None):
    """Calculate total best misfit."""
    best_misfit = float(best_misfit_bw) + float(best_misfit_sw)
    if best_misfit_sw_supp is not None:
        best_misfit += float(best_misfit_sw_supp)
    return best_misfit


def _format_station_info(N_total, N_p_used, N_s_used):
    """Format station information string."""
    if N_total and N_p_used and N_s_used:
        return f',   N-Np-Ns : {N_total}-{N_p_used}-{N_s_used}'
    elif N_s_used:
        return f',   N : {N_s_used}'
    elif N_total:
        return f',   N : {N_total}'
    else:
        return ''


def _passband_formatting(process):
    """Format passband information from process object."""
    if process.freq_max > 1.:
        return '%.1f - %.1f Hz' % (process.freq_min, process.freq_max)
    else:
        return '%.1f - %.1f s' % (process.freq_max**-1, process.freq_min**-1)


def _station_counts(data_bw, data_sw, data_sw_supp):
    """Calculate station counts from data arrays."""
    def get_station_info(data_list):
        if not data_list:
            return set()
        return {sta.id for sta in data_list if sta.count() > 0}
    
    station_ids_bw = get_station_info(data_bw)
    station_ids_sw = get_station_info(data_sw)
    N_p_used = len(station_ids_bw)
    N_s_used = len(station_ids_sw)
    
    if data_sw_supp:
        station_ids_supp = get_station_info(data_sw_supp)
        new_ids_supp = station_ids_supp - station_ids_sw
        N_s_used += len(new_ids_supp)
        station_ids_sw.update(new_ids_supp)
    
    all_station_ids = station_ids_bw.union(station_ids_sw)
    N_total = len(all_station_ids)
    return N_total, N_p_used, N_s_used


# =============================================================================
# Header Factory Functions
# =============================================================================
def build_moment_tensor_header(info):
    """Build header layout for moment tensor inversion results."""
    blocks = [
        MomentTensorFigureBlock(),
        TextBlock('{event_name}  {latlon}  $M_w$ {magnitude:.2f}  {depth_label} {depth_str}', bold=False),
        TextBlock('model: {model}   solver: {solver}   misfit ({norm}): {best_misfit:.3e}', bold=False),
        TextBlock('{passband_line}', bold=False),
        TextBlock('{tensor_coords_line}', bold=False),
        # TextBlock('This adds a bold text line to the header', bold=True),
    ]
    return Header(blocks)


def build_force_header(info):
    """Build header layout for force inversion results."""
    blocks = [
        ForceImageBlock(),
        TextBlock('{event_name}  {latlon}  $F$ {F0:.2e} N   {depth_label} {depth_str}', bold=False),
        TextBlock('model: {model}   solver: {solver}   misfit ({norm}): {best_misfit:.3e}', bold=False),
        TextBlock('{passband_line}', bold=False),
        TextBlock('{phi_theta_line}', bold=False),
    ]
    return Header(blocks)


# =============================================================================
# Header Information Preparation Functions
# =============================================================================

def prepare_moment_tensor_header_info(origin, mt, lune_dict, process_bw, process_sw, process_sw_supp, 
                                     misfit_bw, misfit_sw, best_misfit_bw, best_misfit_sw, model, solver, 
                                     data_bw=None, data_sw=None, mt_grid=None, event_name=None, **kwargs):
    """Prepare moment tensor header information dataclass."""
    # Compute all fields needed for header rendering
    if not event_name:
        event_name = str(origin.time)[:-8]
    
    magnitude = float(mt.magnitude())  # Ensure scalar float
    latlon = format_lat_lon(origin)
    depth_label, depth_str = _format_depth_string(origin)
    
    # Misfit
    norm_label = misfit_sw.norm  # Use as string for label
    best_misfit = _calculate_best_misfit(best_misfit_bw, best_misfit_sw, 
                                       kwargs.get('best_misfit_sw_supp'))
    
    # Passband line
    passband_line = _format_passband_line(process_bw, process_sw, process_sw_supp)
    
    # Station info
    N_total, N_p_used, N_s_used = _station_counts(data_bw, data_sw, kwargs.get('data_sw_supp', None))
    station_info = _format_station_info(N_total, N_p_used, N_s_used)
    
    # Assemble info as dataclass
    info = MomentTensorHeaderInfo(
        event_name=event_name,
        latlon=latlon,
        magnitude=magnitude,
        depth_label=depth_label,
        depth_str=depth_str,
        model=model,
        solver=solver,
        norm=norm_label,
        best_misfit=best_misfit,
        passband_line=passband_line,
        tensor_coords_line=f"{format_focal_mechanism(lune_dict)},   {format_gamma_delta(lune_dict)}{station_info}",
        N_total=N_total,
        N_p_used=N_p_used,
        N_s_used=N_s_used,
        mt=mt,
        lune_dict=lune_dict,
    )
    return info


def prepare_force_header_info(origin, force, force_dict, process_bw, process_sw, process_sw_supp, 
                             misfit_bw, misfit_sw, best_misfit_bw, best_misfit_sw, model, solver, 
                             data_bw=None, data_sw=None, force_grid=None, event_name=None, **kwargs):
    """Prepare force header information dataclass."""
    if not event_name:
        event_name = str(origin.time)[:-8]
    
    latlon = format_lat_lon(origin)
    depth_label, depth_str = _format_depth_string(origin)
    
    norm_label = misfit_sw.norm  # Use as string for label
    best_misfit = _calculate_best_misfit(best_misfit_bw, best_misfit_sw, 
                                       kwargs.get('best_misfit_sw_supp'))
    
    passband_line = _format_passband_line(process_bw, process_sw, process_sw_supp)
    phi_theta_line = format_phi_theta(force_dict)
    F0 = float(force_dict['F0'])  # Ensure scalar float
    
    N_total, N_p_used, N_s_used = _station_counts(data_bw, data_sw, kwargs.get('data_sw_supp', None))
    station_info = _format_station_info(N_total, N_p_used, N_s_used)
    
    info = ForceHeaderInfo(
        event_name=event_name,
        latlon=latlon,
        F0=F0,
        depth_label=depth_label,
        depth_str=depth_str,
        model=model,
        solver=solver,
        norm=norm_label,
        best_misfit=best_misfit,
        passband_line=passband_line,
        phi_theta_line=f"{phi_theta_line}{station_info}",
        N_total=N_total,
        N_p_used=N_p_used,
        N_s_used=N_s_used,
        force_dict=force_dict,
    )
    return info


# =============================================================================
# High-Level Header Creation Functions
# =============================================================================

def create_moment_tensor_header(process_bw, process_sw, misfit_bw, misfit_sw,
                               best_misfit_bw, best_misfit_sw, model, solver, mt, lune_dict, origin,
                               data_bw=None, data_sw=None, mt_grid=None, event_name=None, **kwargs):
    """Create a complete moment tensor header"""
    process_sw_supp = kwargs.pop('process_sw_supp', None)
    header_info = prepare_moment_tensor_header_info(
        origin, mt, lune_dict, process_bw, process_sw, process_sw_supp,
        misfit_bw, misfit_sw, best_misfit_bw, best_misfit_sw, model, solver,
        data_bw=data_bw, data_sw=data_sw, mt_grid=mt_grid, event_name=event_name, **kwargs)
    header = build_moment_tensor_header(header_info)
    return header, header_info


def create_force_header(process_bw, process_sw, misfit_bw, misfit_sw,
                       best_misfit_bw, best_misfit_sw, model, solver, force, force_dict, origin,
                       data_bw=None, data_sw=None, force_grid=None, event_name=None, **kwargs):
    """Create a complete force header"""
    process_sw_supp = kwargs.pop('process_sw_supp', None)
    header_info = prepare_force_header_info(
        origin, force, force_dict, process_bw, process_sw, process_sw_supp,
        misfit_bw, misfit_sw, best_misfit_bw, best_misfit_sw, model, solver,
        data_bw=data_bw, data_sw=data_sw, force_grid=force_grid, event_name=event_name, **kwargs)
    header = build_force_header(header_info)
    return header, header_info


# =============================================================================
# Legacy Wrapper Classes (for backward compatibility)
# =============================================================================
class MomentTensorHeader:
    """Legacy wrapper class for moment tensor headers"""
    
    def __init__(self, process_bw, process_sw, misfit_bw, misfit_sw,
                 best_misfit_bw, best_misfit_sw, model, solver, mt, lune_dict, origin,
                 data_bw=None, data_sw=None, mt_grid=None, event_name=None, **kwargs):
        self.header, self.header_info = create_moment_tensor_header(
            process_bw, process_sw, misfit_bw, misfit_sw, best_misfit_bw, best_misfit_sw,
            model, solver, mt, lune_dict, origin, data_bw, data_sw, mt_grid, event_name, **kwargs)
    
    def write(self, height, width, margin_left, margin_top):
        self.header.write(height, width, margin_left, margin_top, self.header_info)


class ForceHeader:
    """Legacy wrapper class for force headers"""
    
    def __init__(self, process_bw, process_sw, misfit_bw, misfit_sw,
                 best_misfit_bw, best_misfit_sw, model, solver, force, force_dict, origin,
                 data_bw=None, data_sw=None, force_grid=None, event_name=None, **kwargs):
        self.header, self.header_info = create_force_header(
            process_bw, process_sw, misfit_bw, misfit_sw, best_misfit_bw, best_misfit_sw,
            model, solver, force, force_dict, origin, data_bw, data_sw, force_grid, event_name, **kwargs)
    
    def write(self, height, width, margin_left, margin_top):
        self.header.write(height, width, margin_left, margin_top, self.header_info)