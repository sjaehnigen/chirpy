# -------------------------------------------------------------------
#
#  ChirPy
#
#    A buoyant python package for analysing supramolecular
#    and electronic structure, chirality and dynamics.
#
#    https://hartree.chimie.ens.fr/sjaehnigen/chirpy.git
#
#
#  Copyright (c) 2010-2021, The ChirPy Developers.
#
#
#  Released under the GNU General Public Licence, v3 or later
#
#   ChirPy is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published
#   by the Free Software Foundation, either version 3 of the License,
#   or any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.
#   If not, see <https://www.gnu.org/licenses/>.
#
# -------------------------------------------------------------------


import numpy as np
import warnings
import copy

from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from ..classes.core import AttrDict


class pub_label():
    def __init__(self, ax,
                 X=1400,
                 Y=0.0,
                 color="k",
                 size=24,
                 pad=0.0,
                 alpha=1.0,
                 stancil=r'\textbf{%s}',
                 ):
        self.ax = ax
        self.X = X
        self.Y = Y
        self.color = color
        self.size = size
        self.stancil = stancil
        self.pad = pad
        self.alpha = alpha

    def print(self,  string,  **kwargs):
        args = AttrDict()
        for _key in self.__dict__.keys():
            args[_key] = kwargs.get(_key,  getattr(self,  _key))
            try:
                del kwargs[_key]
            except KeyError:
                pass
        self.ax.text(args.X,  args.Y + args.pad,  args.stancil % string,
                     color=args.color,  alpha=args.alpha,  size=args.size,
                     **kwargs)
        return args


def source_params(matplotlib):
    # mpl.rcParams.update({
    # 'font.size':16,
    # 'axes.linewidth':6,
    # 'xtick.major.width':4,
    # 'ytick.major.width':4,
    # 'xtick.major.size':13,
    # 'ytick.major.size':13,
    # 'grid.linewidth':4,
    # 'grid.color':'0.8',
    # })
    matplotlib.rc('xtick',  labelsize=22)
    matplotlib.rc('ytick',  labelsize=22)
    matplotlib.rc('font',   size=22)

    # matplotlib.rcParams['pdf.fonttype'] = 42
    # matplotlib.rcParams['ps.fonttype'] = 42
    # matplotlib.rc('font',
    # **{'family':'sans-serif', 'sans-serif':['Helvetica']}) #gives warning
    matplotlib.rcParams['mathtext.fontset'] = 'stixsans'
    matplotlib.rc('text',  usetex=True)
    matplotlib.rcParams['text.latex.preamble'] = r'''
\usepackage[utf8]{inputenc}
\usepackage{upgreek}
\usepackage{bm}
% \usepackage{xcolor} % destroys vertical alignment
\usepackage{amsmath}
\usepackage{helvet}
\usepackage{sansmath}
\sansmath
\def\mymathhyphen{{\hbox{-}}}
'''
    matplotlib.rcParams.update({'mathtext.default':  'regular'})


def make_nice_ax(p):
    '''p object ... AxesSubplot'''
    p.tick_params('both',  length=5,   width=3,  which='minor')
    p.tick_params('both',  length=10,  width=3,  which='major')
    p.tick_params(axis='both',  which='both',  pad=10,  direction='out')
    # , top=False, right=False)
    # p.yaxis.set_ticks_position('left')
    # p.spines['top'].set_visible(False)
    p.spines['top'].set_linewidth(3.0)
    p.spines['bottom'].set_linewidth(3.0)
    p.spines['left'].set_linewidth(3.0)
    p.spines['right'].set_linewidth(3.0)


def set_mutliple_y_axes(ax, sep, n_axes,
                        offset=0.0,
                        minor=(-1, 1, 11),
                        major=(-1, 1, 5),
                        fmt='%5.1f',
                        auxiliary_axis=False,
                        ):
    if not isinstance(offset, list):
        offset = [offset] * n_axes
    if not isinstance(minor, list):
        minor = [minor] * n_axes
    if not isinstance(major, list):
        major = [major] * n_axes

    _minor_tick_rel_pos = []
    _major_tick_rel_pos = []
    _major_tick_labels = []

    for _i, _m, _M, _o in zip(range(n_axes), minor, major, offset):
        ax.yaxis.set_major_locator(MultipleLocator(_M[2]))
        ax.yaxis.set_minor_locator(MultipleLocator(_m[2]))
        ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))
        ax.yaxis.set_minor_locator(MultipleLocator(_m[-1]))

        # _minor_base = np.linspace(_m[0], _m[1], int((_m[1]-_m[0])/_m[2])+1)
        # _major_base = np.linspace(_M[0], _M[1], int((_M[1]-_M[0])/_M[2])+1)
        _minor_base = np.round(np.linspace(*_m), decimals=6)
        _major_base = np.round(np.linspace(*_M), decimals=6)
        _minor_tick_rel_pos += (_minor_base - _i * sep + _o).tolist()
        _major_tick_rel_pos += (_major_base - _i * sep + _o).tolist()
        _major_tick_labels += _major_base.tolist()

    ax.set_yticks(_minor_tick_rel_pos, minor=True)
    ax.set_yticks(_major_tick_rel_pos, minor=False)
    ax.set_yticklabels(_major_tick_labels)

    # --- auxiliary axis for plot design
    if auxiliary_axis:
        aux_ax = ax.twinx()
        aux_ax.set_ylim(ax.get_ylim())
        return aux_ax


def multiplot(
             ax, x_a, y_a,
             std_a=None,
             bool_a=True,  # False to deactivate plot in list
             color_a=['mediumblue', 'crimson', 'green', 'goldenrod', 'purple'],
             style_a='-',
             alpha_a=1.0,
             lw_a=3,
             fill_alpha_a=0.25,
             fill_color_a=None,
             exp=None,
             style_exp='-',
             alpha_exp=1.0,
             lw_exp=3,
             stack_plots=True,
             pile_up=False,  # fill space between plots (additive)
             fill_between=False,  # fill space between plots (subtractive)
             hatch_a=None,  # pattern for pile_up or fill_between
             offset_a=0.0,  # shift y
             sep=5,  # in %
             hspace=None,  # maually define shift between plots, overrrides sep
             xlim=None,
             ylim=None,
             **kwargs):
    '''Make a nice plot of data in list.
       Arguments with _a denote list of values corresponding to data list y_a
       kwargs contains argument for pyplot
       '''

    if not isinstance(y_a, (list, tuple)):
        raise TypeError('Expected list or tuple for y_a!')

    global _shift  # unique variable also used by pub_label class

    n_plots = len(y_a)

    if exp is not None:
        e,  xe = exp[:,  1],  exp[:,  0]

    if xlim is None:
        xlim = (np.amin(np.array(np.hstack(x_a))),
                np.amax(np.array(np.hstack(x_a))))

    # --- ToDo: create class attributes
    def _listify(xx):
        if not isinstance(xx, (list, tuple)):
            # return [np.array([_x for _x in xx])] * n_plots
            return [xx] * n_plots
        else:
            return xx

    # if not isinstance(x_a, (list, tuple)):  # x_a is different
    #     x_a = [np.array([_x for _x in x_a])] * n_plots
    x_a, \
        bool_a, \
        color_a, \
        style_a, \
        alpha_a, \
        lw_a, \
        fill_alpha_a, \
        fill_color_a, \
        hatch_a, \
        offset_a, \
        = map(_listify, [
                         x_a,
                         bool_a,
                         color_a,
                         style_a,
                         alpha_a,
                         lw_a,
                         fill_alpha_a,
                         fill_color_a,
                         hatch_a,
                         offset_a,
                         ])

    _fill = False
    if std_a is not None:
        _fill = True

    if pile_up and any([stack_plots,  _fill, fill_between]):
        warnings.warn('pile_up set: disabling stack_plots and/or fill_between'
                      ' arguments, respectively!', stacklevel=2)
        stack_plots = False
        _fill = False
        fill_between = False

    if fill_between and std_a is not None:
        warnings.warn('fill_between set: disabling std_a'
                      ' argument', stacklevel=2)
        _fill = False

    if any(len(_a) != n_plots for _a in [y_a,  bool_a]):
        raise ValueError('Inconsistent no. of plots in lists!')

    _slc = [slice(*sorted(np.argmin(np.abs(_x_a-_x)) for _x in xlim))
            for _x_a in x_a]
    if exp is not None:
        _slce = slice(*sorted(np.argmin(np.abs(xe-_x)) for _x in xlim))

    # --- Calculate hspace per plot and ylim
    if hspace is None:
        try:
            _shift = max([np.amax(_y[_s]) - np.amin(_y[_s])
                          for _y, _s in zip(y_a, _slc)])
            if exp is not None:
                _shift = max(np.amax(e[_slce]) - np.amin(e[_slce]), _shift)
            _shift *= (1 + sep / 100)

        except ValueError:
            warnings.warn('Could not calculate plot shift. Good luck!',
                          RuntimeWarning,
                          stacklevel=2)
    else:
        _shift = hspace

    if stack_plots:
        print(_shift)
        if not fill_between:
            _y_a = [_y-_shift*_i for _i, _y in enumerate(y_a)]
            if exp is not None:
                _e = e - n_plots*_shift
        else:
            _y_a = y_a
            if exp is not None:
                _e = e - _shift
    else:
        _y_a = y_a
        if exp is not None:
            _e = e

    if ylim is None:  # add routine for pile_up option
        ylim = (min([np.amin(_y[_s]) for _y, _s in zip(_y_a, _slc)]),
                max([np.amax(_y[_s]) for _y, _s in zip(_y_a, _slc)]))
        if exp is not None:
            ylim = (min(ylim[0], np.amin(_e[_slce])),
                    max(ylim[1], np.amax(_e[_slce])))
        ylim = (ylim[0] - 0.25*_shift, ylim[1] + 0.25*_shift)

    # --- plot reference (experiment)
    if exp is not None:
        ax.plot(xe, _e, style_exp, alpha=alpha_exp, lw=lw_exp, color='black',
                label='exp.')

    # --- plot data
    if _fill:
        for _b, _x, _y, _st, _c, _al, _lw, _s, _fal, _fc, _o in zip(
                                                               bool_a,
                                                               x_a,
                                                               _y_a,
                                                               style_a,
                                                               color_a,
                                                               alpha_a,
                                                               lw_a,
                                                               std_a,
                                                               fill_alpha_a,
                                                               fill_color_a,
                                                               offset_a):
            if _b:
                if _fc is None:
                    _fc = _c
                ax.fill_between(_x, _y+_o+_s, _y+_o-_s, color=_fc, alpha=_fal)
                ax.plot(_x, _y+_o, _st, lw=_lw, color=_c, alpha=_al, **kwargs)

    elif fill_between:
        if n_plots <= 1:
            raise ValueError('fill_between requires at least two data sets')
        if not np.allclose(np.unique(x_a),  x_a[0]):
            raise ValueError('fill_between requires identical x attributes')

        _y_1 = None
        _o_1 = None
        for _iset, (_b, _x, _y, _st, _ha, _c, _al, _lw, _fal, _fc, _o) in \
            enumerate(
                                                           zip(bool_a,
                                                               x_a,
                                                               _y_a,
                                                               style_a,
                                                               hatch_a,
                                                               color_a,
                                                               alpha_a,
                                                               lw_a,
                                                               fill_alpha_a,
                                                               fill_color_a,
                                                               offset_a)):
            if _b:
                if _fc is None:
                    _fc = _c
                if _iset > 0:
                    ax.fill_between(_x, _y+_o, _y_1+_o_1, color=_fc,
                                    alpha=_fal, lw=0, hatch=_ha)
                ax.plot(_x, _y+_o, _st, lw=_lw, color=_c, alpha=_al, **kwargs)
            _y_1 = copy.deepcopy(_y)
            _o_1 = copy.deepcopy(_o)

    elif pile_up:
        if not np.allclose(np.unique(x_a),  x_a[0]):
            raise ValueError('pile_up requires identical x attributes')
        _last = np.zeros_like(_y_a[0])
        for _b, _x, _y, _st, _ha, _c, _al, _lw, _fal, _fc, _o in zip(
                                                                bool_a,
                                                                x_a,
                                                                _y_a,
                                                                style_a,
                                                                hatch_a,
                                                                color_a,
                                                                alpha_a,
                                                                lw_a,
                                                                fill_alpha_a,
                                                                fill_color_a,
                                                                offset_a):
            if _b:
                if _fc is None:
                    _fc = _c
                ax.fill_between(_x, _last, _last + _y+_o,
                                lw=0, color=_fc, alpha=_fal,  hatch=_ha)
                _last += _y+_o
                ax.plot(_x, _last, _st, lw=_lw, color=_c, alpha=_al, **kwargs)
    else:
        for _b, _x, _y, _st, _c, _al, _lw, _o in zip(bool_a,
                                                     x_a,
                                                     _y_a,
                                                     style_a,
                                                     color_a,
                                                     alpha_a,
                                                     lw_a,
                                                     offset_a):
            if _b:
                ax.plot(_x, _y+_o, _st, lw=_lw, color=_c, alpha=_al, **kwargs)

    # --- layout
    make_nice_ax(ax)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    if stack_plots:
        LB = [pub_label(
                        ax,
                        color=_c,
                        X=np.mean(xlim),
                        Y=-_shift * _i,
                        alpha=_al,
                        pad=0.2 * _shift
                    ) for _i,  (_c, _al) in enumerate(zip(color_a, alpha_a))] \
                 + [pub_label(
                        ax,
                        color='black',
                        X=np.mean(xlim),
                        Y=-n_plots * _shift,
                        pad=0.2 * _shift,
                        stancil=r'\emph{%s}'
                    )] * (exp is not None)

    else:
        LB = [pub_label(ax,
                        color=_c,
                        X=np.mean(xlim),
                        Y=np.mean(ylim),
                        alpha=_al) for _c, _al in zip(color_a, alpha_a)] \
             + [pub_label(ax,
                          color='black',
                          X=np.mean(xlim),
                          Y=np.mean(ylim),
                          stancil=r'\emph{%s}')] * (exp is not None)

    return LB


def histogram(ax_a, data_a,
              color_a=['#607c8e', '#c85a53', '#7ea07a', '#c4a661', '#3c4142'],
              alpha_a=1.0,
              bool_a=True,
              sum_to_one=False,  # exclusive with density
              edges=False,  # exclusive with density
              bins=None,
              ylim=None,
              weights_a=None,
              **kwargs):
    '''Create a beautiful histogram plot.
       Requires list of ax'''
    global _shift
    _shift = 0
    n_plots = len(data_a)
    xlim = kwargs.get('range')

    if color_a.__class__ is not list:
        color_a = [color_a] * n_plots
    if alpha_a.__class__ is not list:
        alpha_a = [alpha_a] * n_plots
    if bool_a.__class__ is not list:
        bool_a = [bool_a] * n_plots

    if sum_to_one:
        weights_a = [np.ones_like(_d) / _d.shape[0] for _d in data_a]
    else:
        if weights_a is None:
            weights_a = [np.ones_like(_d) for _d in data_a]

    # ToDo: missing keys: facecolor, edgecolor

    # _bin_width=(xlim[1]-xlim[0])/bins #use system variable?
    for _i, (ax, _b, _d, _c, _al, _wg) in enumerate(zip(ax_a,
                                                        bool_a,
                                                        data_a,
                                                        color_a,
                                                        alpha_a,
                                                        weights_a)):
        if _b and not edges:
            _h, _b_e = np.histogram(_d, bins=bins, **kwargs)
#            _b_e+=_i*sep*_bin_width
            # a = ax.hist(_d, bins=_b_e, color=_c, alpha=_al, weights=_wg,
            # bottom=-_i*sep, **kwargs)
#        elif edges: a = ax.hist(_h, bins=_b_e, facecolor='None',
        # edgecolor=_c, alpha=_al, **kwargs)

        make_nice_ax(ax)
        ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

    # --- export label object (beta)

    LB = [pub_label(ax, color=_c, Y=0.0, X=np.mean(_d), alpha=_al)
          for ax, _c, _al, _d in zip(ax_a, color_a, alpha_a, data_a)]

    return LB
