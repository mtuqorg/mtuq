
import matplotlib
from os.path import dirname, join


from mtuq.graphics.attrs import\
    plot_time_shifts, plot_amplitude_ratios, plot_cc_max, _plot_attrs

from mtuq.graphics.beachball import\
    plot_beachball, plot_polarities, _plot_beachball_matplotlib

from mtuq.graphics.uq.lune import\
    plot_misfit_lune, plot_likelihood_lune, plot_marginal_lune,\
    plot_variance_reduction_lune, plot_magnitude_tradeoffs_lune, _plot_lune

from mtuq.graphics.uq.vw import\
    plot_misfit_vw, plot_likelihood_vw, plot_marginal_vw, \
    _misfit_vw_regular, _likelihoods_vw_regular, _marginals_vw_regular, \
    _plot_vw, _product_vw

from mtuq.graphics.uq.double_couple import\
    plot_misfit_dc, plot_likelihood_dc, plot_marginal_dc, _plot_dc, \
    plot_variance_reduction_dc

from mtuq.graphics.uq.force import\
    plot_misfit_force, plot_likelihood_force, plot_marginal_force,\
    plot_magnitude_tradeoffs_force, _plot_force

from mtuq.graphics.uq.depth import\
    plot_misfit_depth, plot_likelihood_depth, plot_marginal_depth, _plot_depth

from mtuq.graphics.uq.hypocenter import\
    plot_misfit_latlon, plot_likelihood_latlon, plot_marginal_latlon, _plot_latlon

from mtuq.graphics.waveforms import\
    plot_waveforms1, plot_waveforms2, plot_waveforms3,\
    plot_data_greens1, plot_data_greens2, plot_data_greens3

from mtuq.graphics.uq.omega import\
    plot_cdf, plot_pdf, plot_screening_curve

from mtuq.graphics.uq import\
    likelihood_analysis


# use Nimbus Sans L as default font
try:
    import matplotlib.font_manager as font_manager
    path = join(dirname(__file__), '__fonts')
    for font in font_manager.findSystemFonts(path):
        font_manager.fontManager.addfont(font)
    matplotlib.rcParams['font.family'] = "Nimbus Sans L"

except:
    pass

