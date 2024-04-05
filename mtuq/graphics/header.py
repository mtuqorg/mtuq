
#
# graphics/header.py - figure headers and text
#


import numpy as np
import os
from matplotlib import pyplot
from matplotlib.font_manager import FontProperties
from mtuq.event import MomentTensor
from mtuq.graphics.beachball import plot_beachball
from mtuq.graphics._pygmt import exists_pygmt, plot_force
from mtuq.util.math import to_delta_gamma


class Base(object):
    """ Base class for writing headers to matplotlib figures
    """
    def __init__(self):
        raise NotImplementedError("Must be implemented by subclass")


    def _get_axis(self, height, fig=None):
        """ Returns matplotlib axes of given height along top of figure
        """
        if hasattr(self, '_axis'):
            return self._axis

        if fig is None:
            fig = pyplot.gcf()
        width, figure_height = fig.get_size_inches()

        assert height < figure_height, Exception(
             "Header height exceeds entire figure height. Please double check "
             "input arguments.")
               
        x0 = 0.
        y0 = 1.-height/figure_height

        self._axis = fig.add_axes([x0, y0, 1., height/figure_height])
        ax = self._axis

        ax.set_xlim([0., width])
        ax.set_ylim([0., height])

        # hides axes lines, ticks, and labels
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        return ax


    def write(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented by subclass")


class TextHeader(Base):
    """ Generic text header

    Prints header text from a list ((xp, yp, text), ...)
    """
    def __init__(self, items):
        # validates items
        for item in items:
            assert len(item) >= 3
            xp, yp, text  = item[0], item[1], item[2]
            assert 0. <= xp <= 1.
            assert 0. <= yp <= 1.

        self.items = items


    def write(self, height, width, margin_left, margin_top):
        ax = self._get_axis(height)

        for item in self.items:
            xp, yp, text = item[0], item[1], item[2]
            kwargs = {}
            if len(item) > 3:
                kwargs = item[3]

            _write_text(text, xp, yp, ax, **kwargs)


class SourceHeader(Base):
    """ Base class for moment tensor and force headers

    (Added to reduce duplication, still somewhat of an afterthought)
    """

    def parse_origin(self):
        depth_in_m = self.origin.depth_in_m
        depth_in_km = self.origin.depth_in_m/1000.
        if depth_in_m < 1000.:
            self.depth_str = '%.0f m' % depth_in_m
        elif depth_in_km <= 100.:
            self.depth_str = '%.1f km' % depth_in_km
        else:
            self.depth_str = '%.0f km' % depth_in_km


    def parse_misfit(self):
        # TODO - keep track of body and surface wave norms
        self.norm = self.misfit_sw.norm
        self.best_misfit = self.best_misfit_bw + self.best_misfit_sw


    def parse_data_processing(self):
        if not self.process_bw:
            pass
        if not self.process_sw:
            raise Exception()

        if self.process_sw.freq_max > 1.:
            units = 'Hz'
        else:
            units = 's'

        if self.process_bw and units=='Hz':
            self.passband_bw = '%.1f - %.1f Hz' %\
                (self.process_bw.freq_min, self.process_bw.freq_max)

        elif self.process_bw and units=='s':
            self.passband_bw = '%.1f - %.1f s' %\
                (self.process_bw.freq_max**-1, self.process_bw.freq_min**-1)

        if self.process_sw and units=='Hz':
            self.passband_sw = '%.1f - %.1f Hz' %\
                (self.process_sw.freq_min, self.process_sw.freq_max)

        elif self.process_sw and units=='s':
            self.passband_sw = '%.1f - %.1f s' %\
                (self.process_sw.freq_max**-1, self.process_sw.freq_min**-1)



class MomentTensorHeader(SourceHeader):
    """ Writes moment tensor inversion summary to the top of a 
    matplotlib figure
    """
    def __init__(self, process_bw, process_sw, misfit_bw, misfit_sw,
        best_misfit_bw, best_misfit_sw, model, solver, mt, lune_dict, origin,
        data_bw=None, data_sw=None, mt_grid=None, event_name=None):

        if not event_name:
           # YYYY-MM-DDTHH:MM:SS.??????Z
           event_name = '%s' % origin.time

           # trim fraction of second
           event_name = event_name[:-8]

        self.event_name = event_name

        # required arguments
        self.process_bw = process_bw
        self.process_sw = process_sw
        self.misfit_bw = misfit_bw
        self.misfit_sw = misfit_sw
        self.best_misfit_bw = best_misfit_bw
        self.best_misfit_sw = best_misfit_sw
        self.model = model
        self.solver = solver
        self.mt = mt
        self.lune_dict = lune_dict
        self.origin = origin

        # optional arguments, reserved for possible future use
        # (or for use by subclasses)
        self.data_bw = data_bw
        self.data_sw = data_sw
        self.mt_grid = mt_grid

        # moment tensor-derived attributes
        self.magnitude = mt.magnitude()


        self.parse_origin()
        self.parse_misfit()
        self.parse_data_processing()


    def display_source(self, ax, height, width, offset):

        #
        # If ObsPy plotted focal mechanisms correctly we could do the following
        #

        #from obspy.imaging.beachball import beach
        ## beachball size
        #diameter = 0.75*height
        #xp = 0.50*diameter + offset
        #yp = 0.45*height
        #ax.add_collection(
        #    beach(self.mt, xy=(xp, yp), width=diameter,
        #    linewidth=0.5, facecolor='gray'))

        #
        # Instead, we must use this workaround
        #

        # beachball size
        diameter = 0.75*height

        # beachball placement
        xp = offset
        yp = 0.075*height

        plot_beachball('tmp.png', self.mt, None, None)
        img = pyplot.imread('tmp.png')

        try:
            os.remove('tmp.png')
            os.remove('tmp.ps')
        except:
            pass

        ax.imshow(img, extent=(xp,xp+diameter,yp,yp+diameter))


    def write(self, height, width, margin_left, margin_top):
        """ Writes header text to current figure
        """
        ax = self._get_axis(height)

        self.display_source(ax, height, width, margin_left)

        px = 2.*margin_left + 0.75*height
        py = height - margin_top

        # write text line #1
        px += 0.00
        py -= 0.35

        line = '%s  %s  $M_w$ %.2f  Depth %s' % (
            self.event_name, _lat_lon(self.origin), self.magnitude, self.depth_str)
        _write_bold(line, px, py, ax, fontsize=16.5)


        # write text line #2
        px += 0.00
        py -= 0.30

        line = u'model: %s   solver: %s   misfit (%s): %.3e' % \
                (self.model, self.solver, self.norm, self.best_misfit)
        _write_text(line, px, py, ax, fontsize=14)


        # write text line #3
        px += 0.00
        py -= 0.30

        if self.process_bw and self.process_bw:
            line = ('body waves:  %s (%.1f s),  ' +\
                    'surface waves: %s (%.1f s)') %\
                    (self.passband_bw, self.process_bw.window_length,
                     self.passband_sw, self.process_sw.window_length)

        elif self.process_sw:
            line = 'passband: %s,  window length: %.1f s ' %\
                    (self.passband_sw, self.process_sw.window_length)

        _write_text(line, px, py, ax, fontsize=14)


        # write text line #4
        px += 0.00
        py -= 0.30

        line = _focal_mechanism(self.lune_dict)
        line +=  ',   '+_gamma_delta(self.lune_dict)
        _write_text(line, px, py, ax, fontsize=14)



class ForceHeader(SourceHeader):
    """ Writes force inversion summary to the top of a matplotlib figure
    """

    def __init__(self, process_bw, process_sw, misfit_bw, misfit_sw,
        best_misfit_bw, best_misfit_sw, model, solver, force, force_dict, origin,
        data_bw=None, data_sw=None, force_grid=None, event_name=None):

        if not event_name:
           # YYYY-MM-DDTHH:MM:SS.??????Z
           event_name = '%s' % origin.time

           # trim fraction of second
           event_name = event_name[:-8]

        self.event_name = event_name

        # required arguments
        self.process_bw = process_bw
        self.process_sw = process_sw
        self.misfit_bw = misfit_bw
        self.misfit_sw = misfit_sw
        self.best_misfit_bw = best_misfit_bw
        self.best_misfit_sw = best_misfit_sw
        self.model = model
        self.solver = solver
        self.force = force
        self.force_dict = force_dict
        self.origin = origin

        # optional arguments, reserved for possible future use
        # (or for use by subclasses)
        self.data_bw = data_bw
        self.data_sw = data_sw
        self.force_grid = force_grid

        self.parse_origin()
        self.parse_misfit()
        self.parse_data_processing()


    def write(self, height, width, margin_left, margin_top):

        ax = self._get_axis(height)

        self.display_source(ax, height, width, margin_left)


        px = 2.*margin_left + 0.75*height
        py = height - margin_top


        # write text line #1
        px += 0.00
        py -= 0.35

        line = '%s  %s  $F$ %.2e N   Depth %s' % (
            self.event_name, _lat_lon(self.origin), self.force_dict['F0'], self.depth_str)
        _write_bold(line, px, py, ax, fontsize=16)


        # write text line #2
        px += 0.00
        py -= 0.30

        line = u'model: %s   solver: %s   misfit (%s): %.3e' % \
                (self.model, self.solver, self.norm, self.best_misfit)
        _write_text(line, px, py, ax, fontsize=14)


        # write text line #3
        px += 0.00
        py -= 0.30

        if self.process_bw and self.process_bw:
            line = ('body waves:  %s (%.1f s),  ' +\
                    'surface waves: %s (%.1f s) ') %\
                    (self.passband_bw, self.process_bw.window_length,
                     self.passband_sw, self.process_sw.window_length)

        elif self.process_sw:
            line = 'passband: %s,  window length: %.1f s ' %\
                    (self.passband_sw, self.process_sw.window_length)

        _write_text(line, px, py, ax, fontsize=14)


        # write text line #4
        px += 0.00
        py -= 0.30

        line = _phi_theta(self.force_dict)
        _write_text(line, px, py, ax, fontsize=14)


    def display_source(self, ax, height, width, offset):

        if not exists_pygmt():
            return

        # image size
        diameter = 0.75*height

        # image placement
        xp = offset
        yp = 0.075*height

        plot_force('tmp.png', self.force_dict)
        img = pyplot.imread('tmp.png')

        try:
            os.remove('tmp.png')
            os.remove('tmp.ps')
        except:
            pass

        ax.imshow(img, extent=(xp,xp+diameter,yp,yp+diameter))



def _lat_lon(origin):
    if origin.latitude >= 0:
        latlon = '%.2f%s%s' % (+origin.latitude, u'\N{DEGREE SIGN}', 'N')
    else:
        latlon = '%.2f%s%s' % (-origin.latitude, u'\N{DEGREE SIGN}', 'S')

    if origin.longitude > 0:
        latlon += '% .2f%s%s' % (+origin.longitude, u'\N{DEGREE SIGN}', 'E')
    else:
        latlon += '% .2f%s%s' % (-origin.longitude, u'\N{DEGREE SIGN}', 'W')

    return latlon


def _focal_mechanism(lune_dict):
    strike = lune_dict['kappa']

    try:
        dip = np.degrees(np.arccos(lune_dict['h']))
    except:
        dip = lune_dict['theta']

    slip = lune_dict['sigma']

    return ("strike  dip  slip:  %.f  %.f  %.f" %
        (strike, dip, slip))


def _gamma_delta(lune_dict):
    try:
        v, w = lune_dict['v'], lune_dict['w']
        delta, gamma = to_delta_gamma(v, w)
    except:
        delta, gamma = lune_dict['delta'], lune_dict['gamma']

    return 'lune coords %s  %s:  %.f  %.f' % (u'\u03B3', u'\u03B4', gamma, delta)



def _phi_theta(force_dict):
    try:
        phi, theta = force_dict['phi'], force_dict['theta']
    except:
        phi, h = force_dict['phi'], force_dict['h']
        theta = np.degrees(np.arccos(h))

    return '%s  %s:  %.f  %.f' % (u'\u03C6', u'\u03B8', phi, theta)


def _write_text(text, x, y, ax, fontsize=12, **kwargs):
    pyplot.text(x, y, text, fontsize=fontsize, **kwargs)


def _write_bold(text, x, y, ax, fontsize=14):
    font = FontProperties()
    #font.set_weight('bold')
    pyplot.text(x, y, text, fontproperties=font, fontsize=fontsize)


def _write_italic(text, x, y, ax, fontsize=12):
    font = FontProperties()
    font.set_style('italic')
    pyplot.text(x, y, text, fontproperties=font, fontsize=fontsize)


