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
import copy

from scipy.interpolate import UnivariateSpline


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

    @classmethod
    def from_vector(cls, origins_aa, vectors_aa,
                    scale=1,
                    thresh=0.0
                    ):
        '''Generate VMD object from vector array
           origins_aa/vectors_aa ... numpy arrays of shape (n_vectors, 3)
           scale ... int or numpy array of shape (n_vectors) to multiply
                     vectors with
           thresh ... exclude regions of where vector norm is smaller than
                      value (before scaling)
           '''
        _ind = np.linalg.norm(vectors_aa, axis=-1) > thresh
        _p0 = origins_aa.T
        _p1 = _p0 + vectors_aa.T * scale

        _p = np.array([_p0.T, _p1.T])

        return cls(_p[:, _ind], auto_smooth=False)

    @classmethod
    def from_vector_field(cls, obj,
                          sparse=1,
                          scale=1,
                          length=1,
                          normalise=None,
                          thresh=0.0,
                          verbose=False
                          ):
        '''Generate VMD object from VectorField object
           normalise ... None/max/local
           thresh ... exclude regions of where vector norm is smaller than
                      value (before normalisation)
           '''

        # -- apply sparse and thresh
        _obj = obj.sparse(sparse)
        _ind = np.linalg.norm(_obj.data, axis=0) > thresh
        _p = _obj.pos_grid()[:, _ind].reshape((3, -1))

        # --- continue with original obj
        if normalise is not None:
            if normalise == 'max':
                obj.normalise(np.amax(np.linalg.norm(obj.data, axis=0)))
            elif normalise == 'local':
                obj.normalise(axis=0)

        if verbose:
            print(f"Seeding {_p.shape[-1]} points.")

        pos_aa = obj.streamlines(
                    _p.T,
                    sparse=1,  # sparsity of interpolation
                    forward=True,
                    backward=True,
                    length=length,
                    timestep_fs=scale
                    )['streamlines'][:, :, :3]

        return cls(pos_aa, auto_smooth=False)

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
            length = 8 * radius

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

    @staticmethod
    def _get_color_id():
        '''picks randonly one of VMD's colors (57 < id < 1057 to preserve solid
        colors). Warning: this may interfere with other
        represantations/drawings'''
        return int(np.random.random() * 1000 + 57)

    def draw_line(self, fn,
                  cutoff_aa=0.1,
                  sparse=5,
                  style='solid',
                  width=1,
                  rgb=(0.5, 0.5, 0.5),
                  arrow=False,
                  arrow_radius=None,
                  arrow_length=None,
                  arrow_resolution=30,
                  ):
        '''sparsity: skip every <sparse>th point in positions; <0 for reversed
        order'''
        tool = 'line'
        options = f'width {width} style {style}'
        self.reduce(cutoff_aa=cutoff_aa)
        if arrow_length is None:
            arrow_length = 0.064 * width
        if arrow_radius is None:
            arrow_radius = arrow_length / 8

        with open(fn, 'w') as f:
            _col = self._get_color_id()
            f.write("mol new\n")
            f.write("color change rgb %d %f %f %f\n" % tuple((_col,) + rgb))
            f.write("draw color %d\n" % _col)
            f.write("draw materials off\n")
            f.write(self._draw(tool=tool, options=options, sparse=sparse,
                    overlap=0.0))
            if arrow:
                f.write(self._draw_arrow_tip(self.pos_aa[::sparse],
                                             radius=arrow_radius,
                                             length=arrow_length,
                                             resolution=arrow_resolution))
            f.write("display resetview\n")

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
        if arrow_length is None:
            arrow_length = 8 * arrow_radius

        with open(fn, 'w') as f:
            _col = self._get_color_id()
            f.write("mol new\n")
            f.write("color change rgb %d %f %f %f\n" % tuple((_col,) + rgb))
            f.write("draw color %d\n" % _col)
            f.write("draw materials on\n")
            f.write("draw material %s\n" % material)
            f.write(self._draw(tool=tool, options=options, sparse=sparse,
                    overlap=0.0))
            if arrow:
                f.write(self._draw_arrow_tip(self.pos_aa[::sparse],
                                             radius=arrow_radius,
                                             length=arrow_length,
                                             resolution=arrow_resolution))
            f.write("display resetview\n")
