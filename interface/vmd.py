#!/usr/bin/env python
# ------------------------------------------------------
#
#  ChirPy
#
#    A buoyant python package for analysing supramolecular
#    and electronic structure, chirality and dynamics.
#
#
#  Developers:
#    2010-2016  Arne Scherrer
#    since 2014 Sascha Jähnigen
#
#  https://hartree.chimie.ens.fr/sjaehnigen/chirpy.git
#
# ------------------------------------------------------


import numpy as np
import copy

from scipy.interpolate import UnivariateSpline

from ..physics import constants

# ToDo: NEEDS Routine to include arrow tip in total vector length


class VMDPaths():
    def __init__(self, positions_aa, auto_smooth=True):
        '''Read a set of paths defines by node positions in angstrom with
           shape (n_points p. path[, n_paths], 3)'''

        if len(positions_aa.shape) == 2:
            self.pos_aa = np.array(positions_aa[:, None])
        elif len(positions_aa.shape) == 3:
            self.pos_aa = positions_aa
        else:
            raise ValueError('Expected positions with 2 or 3 axes, got'
                             f' {positions_aa.shape}')

        self.n_points, self.n_paths, three = self.pos_aa.shape

        if auto_smooth:
            self.smooth()
        print(self.pos_aa.shape)

    def smooth(self):
        '''Smoothing with k=3 spline'''
        def spline(points):
            x = np.arange(points.shape[0])
            spl0 = UnivariateSpline(x, points[:, 0])
            spl1 = UnivariateSpline(x, points[:, 1])
            spl2 = UnivariateSpline(x, points[:, 2])
            return np.array([spl0(x), spl1(x), spl2(x)]).swapaxes(0, 1)

        self.pos_aa = np.array(
                [spline(points) for points in self.pos_aa.swapaxes(0, 1)]
                ).swapaxes(0, 1)

    def reduce(self, cutoff_aa=0.0):
        '''Keep paths whose length is above given cutoff'''
        ind = np.linalg.norm(np.abs(
                                np.diff(self.pos_aa[:, :, :3], axis=0)
                                ).sum(axis=0),
                             axis=-1) >= cutoff_aa  # integrate curved path
        self.pos_aa = self.pos_aa[:, ind]


    @staticmethod
    def tmp_normalise(obj, norm=None, thresh=1.E-8, **kwargs):
        from ..classes.volume import ScalarField
        '''Norm has to be a ScalarField object (can be of different shape) or
           float.
           If no norm is given, the method uses np.linalg.norm of vector field
           (give axis in kwargs).'''

        # --- create empty object with the correct grid
        _N = ScalarField.from_object(obj, data=obj.grid())
        if norm is None:
            _N.data = np.linalg.norm(obj.data, **kwargs)
        elif isinstance(norm, float):
            _N.data += norm
        else:
            # --- __add__ interpolates different grids
            _N += norm

        with np.errstate(divide='ignore'):
            _N_inv = np.where(_N.data < thresh, 0.0, np.divide(1.0, _N.data))

        obj.data *= _N_inv
        return obj


    @classmethod
    def from_vector_field(cls, obj,
                          sparse=1,
                          scale=1,
                          length=1,
                          normalise=None,
                          thresh=0.0
                          ):
        '''Generate VMD object from VectorField object
           normalise ... None/max/local
           thresh ... exclude regions of where vector norm is smaller than
                      value (before normalisation)
           '''

        _obj = obj.sparse(sparse)
        _ind = np.linalg.norm(_obj.data, axis=0) > thresh

        if normalise is not None:
            if normalise == 'max':
                _obj = cls.tmp_normalise(_obj, np.amax(np.linalg.norm(_obj.data, axis=0)))
            elif normalise == 'local':
                _obj = cls.tmp_normalise(_obj, axis=0)

        _p = _obj.pos_grid()[:, _ind].reshape((3, -1))
        print(f"Seeding {_p.shape[-1]} points.")

        pos_au = obj.streamlines(  # NB: using original obj
                    _p.T,
                    sparse=1,  # sparsity of interpolation
                    forward=True,
                    backward=True,
                    length=length,
                    timestep=scale
                    )['streamlines'][:, :, :3]

        return cls(pos_au*constants.l_au2aa, auto_smooth=False)

    @staticmethod
    def _draw_bit(p0, p1, tool='line', options='', overlap=0.0):
        '''p0, p1 ... path bits of shape (n_paths, 3)'''
        return ''.join([
            f"draw {tool} " +
            "{%16.9f %16.9f %16.9f} " % tuple(t0) +
            "{%16.9f %16.9f %16.9f} " % tuple(t0 + (1.0+overlap) * (t1-t0)) +
            options +
            "\n"
            for t0, t1 in zip(p0, p1)])

    def _draw(self, sparse=5, **kwargs):
        block = ''
        p0 = None
        for ip, p1 in enumerate(self.pos_aa[::sparse]):
            if ip > 0:
                block += self._draw_bit(p0, p1, **kwargs)
            p0 = copy.deepcopy(p1)
        return block

    @staticmethod
    def _draw_arrow_tip(positions_aa,
                        radius=0.075,
                        length=None,
                        resolution=30):
        '''Add cone to the ends of paths given through positions of
        shape (n_points p. path[, n_paths], 3)'''
        if length is None:
            length = 5 * radius

        def arr_head_sense(p, depth):
            '''unit vector of pointing cone'''
            backtrace = p[-1, None] - p[-depth:-1]
            with np.errstate(divide='ignore'):
                _N = np.linalg.norm(backtrace, axis=-1)
                _N_inv = np.where(_N < 1.E-16, 0.0, np.divide(1.0, _N))

            return (backtrace * _N_inv[:, :, None]).sum(axis=0) / depth

        pos = positions_aa[-1]
        sense = arr_head_sense(positions_aa, 5)

        return ''.join(["draw cone " +
                        "{%16.9f %16.9f %16.9f} " % tuple(t0) +
                        "{%16.9f %16.9f %16.9f} " % tuple(t0 + length * t1) +
                        f"radius {radius} resolution {resolution}"
                        "\n"
                        for t0, t1 in zip(pos, sense)])

    def draw_line(self, fn,
                  cutoff_aa=0.1,
                  sparse=5,
                  style='solid',
                  width=1,
                  rgb=(0.5, 0.5, 0.5),
                  arrow=False,
                  arrow_radius=0.075,
                  arrow_length=None,
                  arrow_resolution=30,
                  ):
        '''sparsity: skip every <sparse>th point in positions; <0 for reversed
        order'''
        tool = 'line'
        options = f'width {width} style {style}'
        self.reduce(cutoff_aa=cutoff_aa)

        with open(fn, 'w') as f:
            f.write("color change rgb tan %f %f %f\n" % rgb)
            f.write("draw color tan\n")
            f.write("draw materials off\n")
            f.write(self._draw(tool=tool, options=options, sparse=sparse,
                    overlap=0.0))
            if arrow:
                f.write(self._draw_arrow_tip(self.pos_aa[::sparse],
                                             radius=arrow_radius,
                                             length=arrow_length,
                                             resolution=arrow_resolution))

    def draw_tube(self, fn,
                  cutoff_aa=0.1,
                  sparse=5,
                  radius=0.025,
                  resolution=10,
                  material='AOShiny',
                  rgb=(0.0, 0.4, 0.0),
                  arrow=False,
                  arrow_radius=None,
                  arrow_length=None,
                  arrow_resolution=30,
                  ):
        '''sparsity: skip every <sparse>th point in positions; <0 for reversed
        order'''
        tool = 'cylinder'
        options = f'radius {radius} resolution {resolution}'
        self.reduce(cutoff_aa=cutoff_aa)
        if arrow_radius is None:
            arrow_radius = 3 * radius

        with open(fn, 'w') as f:
            f.write("color change rgb tan %f %f %f\n" % rgb)
            f.write("draw color tan\n")
            f.write("draw materials on\n")
            f.write("draw material %s\n" % material)
            f.write(self._draw(tool=tool, options=options, sparse=sparse,
                    overlap=0.0))
            if arrow:
                f.write(self._draw_arrow_tip(self.pos_aa[::sparse],
                                             radius=arrow_radius,
                                             length=arrow_length,
                                             resolution=arrow_resolution))

# [f.write("draw sphere {%8.2f %8.2f %8.2f} radius %f resolution %d\n"
# %(t1[0], t1[1], t1[2], rad, res)) for t1 in p1]

# draw color black
# draw materials off
# draw cylinder {6 6 4} {6 6 4.2} radius 7.2 resolution 50
# color change rgb tan 0.0 0.4 0.0
# draw color tan
# draw materials off
# draw cylinder {6 6 4} {6 6 10} radius 0.1 resolution 50
# draw cone {6 6 10} {6 6 11} radius 0.3 resolution 50
# draw materials off
# draw text {5.3 6 11.2} "r" size 2 thickness 5
# draw text {5.0 6 11.5} "z" size 1 thickness 3
# draw color orange
# draw materials off
# draw cylinder {6 6 4} {10.243 10.243 4} radius 0.1 resolution 50
# draw cone {10.243 10.243 4} {10.950 10.950 4} radius 0.3 resolution 50
# draw materials off
# draw text {11.332 11.232 3.0} "r" size 2 thickness 5
# draw text {11.032 11.232 3.3} "xy" size 1 thickness 3
