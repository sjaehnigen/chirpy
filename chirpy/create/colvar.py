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
#  Copyright (c) 2010-2022, The ChirPy Developers.
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

import numpy as _np
# import copy as _copy
#
from ..classes.core import CORE as _CORE
from ..topology.mapping import vector_pbc, angle_pbc, dihedral_pbc
from ..mathematics.algebra import cross, dot, angle
# from ..physics import classical_electrodynamics as ed
# from .. import constants


class _COLVAR(_CORE):
    def __repr__(self):
        return self.label

    def __mul__(self, other):
        if not isinstance(other, (int, float, complex)):
            raise TypeError('unsupported operand type(s) for *: '
                            f'{type(other).__name__}')
        return Combination([self], weights=[other])

    def __add__(self, other):
        self._isinstance(other, info='*')
        return Combination([self, other], weights=(1., 1.))

    def __sub__(self, other):
        self._isinstance(other, info='*')
        return Combination([self, other], weights=(1., -1.))

    @staticmethod
    def _isinstance(item, info='operation'):
        if not item.__class__.__mro__[1].__name__ == '_COLVAR':
            # if not isinstance(item, _COLVAR):
            raise TypeError('unsupported type(s) for {info}: '
                            f'{type(item).__name__}')

    def value(self, configuration, cell=None):
        '''Evaluate collective variable for given configuration.

           arguments:
               configuration ... input positions,
                                 array of shape ([n_frames], n_atoms, 3)
               cell ... 6-dimensional cell vector: [a, b, c, al, be, ga]
                        (optional)

           returns:
               colvar value or array shape (n_frames,)
           '''

        return self._eval(configuration, cell=cell)

    def derivative(self,  configuration, cell=None):
        '''Evaluate collective variable derivative with respect to
           atomic displacements in the given configuration.

           arguments:
               configuration ... input positions,
                                 array of shape ([n_frames], n_atoms, 3)
               cell ... 6-dimensional cell vector: [a, b, c, al, be, ga]
                        (optional)

           returns:
               colvar derivative, array of shape ([n_frames], n_atoms, 3)
           '''

        return self._eval_der(configuration, cell=cell)


class Bond(_COLVAR):
    '''i0 --> i1'''
    def __init__(self, i0, i1):
        self.label = f'{self.__class__.__name__}({i0}, {i1})'
        self.i0 = i0
        self.i1 = i1

    def _eval(self, configuration, cell=None):
        vector = vector_pbc(
                configuration[..., self.i0, :],
                configuration[..., self.i1, :],
                cell=cell
                )
        norm = _np.linalg.norm(vector, axis=-1)
        return norm

    def _eval_der(self, configuration, cell=None):
        vector = vector_pbc(
                configuration[..., self.i0, :],
                configuration[..., self.i1, :],
                cell=cell
                )
        norm = _np.linalg.norm(vector, axis=-1)

        derivative = _np.zeros_like(configuration)
        derivative[..., self.i0, :] = -vector / norm[..., None]
        derivative[..., self.i1, :] = -derivative[..., self.i0, :]

        return derivative


class Angle(_COLVAR):
    '''i0 <-- i1 --> i2'''
    def __init__(self, i0, i1, i2):
        self.label = f'{self.__class__.__name__}({i0}, {i1}, {i2})'
        self.i0 = i0
        self.i1 = i1
        self.i2 = i2

    def _eval(self, configuration, cell=None):
        angle = angle_pbc(
                configuration[..., self.i0, :],
                configuration[..., self.i1, :],
                configuration[..., self.i2, :],
                cell=cell
                )
        return angle

    def _eval_der(self, configuration, cell=None):
        vector0 = vector_pbc(
                configuration[..., self.i1, :],
                configuration[..., self.i0, :],
                cell
                )
        vector1 = vector_pbc(
                configuration[..., self.i1, :],
                configuration[..., self.i2, :],
                cell
                )
        angle = angle_pbc(
                configuration[..., self.i0, :],
                configuration[..., self.i1, :],
                configuration[..., self.i2, :],
                cell=cell
                )
        _sin = _np.sin(angle)[..., None]
        _cos = _np.cos(angle)[..., None]
        norm0 = _np.linalg.norm(vector0, axis=-1)[..., None]
        norm1 = _np.linalg.norm(vector1, axis=-1)[..., None]
        u0 = vector0 / norm0
        u1 = vector1 / norm1

        derivative = _np.zeros_like(configuration)
        derivative[..., self.i0, :] = (u0*_cos - u1) / norm0 / _sin
        derivative[..., self.i2, :] = (u1*_cos - u0) / norm1 / _sin
        derivative[..., self.i1, :] = - derivative[..., self.i0, :] \
                                      - derivative[..., self.i2, :]

        return derivative


class Dihedral(_COLVAR):
    '''i0 <-- i1 --> i2 --> i3'''
    def __init__(self, i0, i1, i2, i3):
        self.label = f'{self.__class__.__name__}({i0}, {i1}, {i2}, {i3})'
        self.i0 = i0
        self.i1 = i1
        self.i2 = i2
        self.i3 = i3

    def _eval(self, configuration, cell=None):
        dihedral = dihedral_pbc(
                configuration[..., self.i0, :],
                configuration[..., self.i1, :],
                configuration[..., self.i2, :],
                configuration[..., self.i3, :],
                cell=cell
                )
        return dihedral

    def _eval_der(self, configuration, cell=None):
        # see also: https://salilab.org/modeller/9v6/manual/node436.html
        vector0 = vector_pbc(
                configuration[..., self.i1, :],
                configuration[..., self.i0, :],
                cell
                )
        vector1 = vector_pbc(
                configuration[..., self.i1, :],
                configuration[..., self.i2, :],
                cell
                )
        vector2 = vector_pbc(
                configuration[..., self.i2, :],
                configuration[..., self.i3, :],
                cell=cell
                )
        norm1 = _np.linalg.norm(vector1, axis=-1)

        plane0 = cross(vector0, vector1)
        plane1 = cross(vector1, vector2)
        norm_plane0 = _np.linalg.norm(plane0, axis=-1)
        norm_plane1 = _np.linalg.norm(plane1, axis=-1)

        scalar0 = (dot(vector0, vector1) / norm1**2)[..., None]
        scalar1 = (dot(vector2, vector1) / norm1**2)[..., None]

        derivative = _np.zeros_like(configuration)
        derivative[..., self.i0, :] = \
            (norm1 / norm_plane0**2)[..., None] * plane0
        derivative[..., self.i3, :] = \
            (norm1 / norm_plane1**2)[..., None] * plane1
        derivative[..., self.i1, :] = \
            (scalar0 - 1) * derivative[..., self.i0, :] + \
            scalar1 * derivative[..., self.i3, :]
        derivative[..., self.i2, :] = \
            (-scalar1 - 1) * derivative[..., self.i3, :] - \
            scalar0 * derivative[..., self.i0, :]

        return derivative


class Outplane(_COLVAR):
    '''
    i0 <-- i3 --> i2
           ¦
           V
           i1'''
    def __init__(self, i0, i1, i2, i3):
        self.label = f'{self.__class__.__name__}({i0}, {i1}, {i2}, {i3})'
        self.i0 = i0
        self.i1 = i1
        self.i2 = i2
        self.i3 = i3

    def _eval(self, configuration, cell=None):
        vector0 = vector_pbc(
                configuration[..., self.i3, :],
                configuration[..., self.i0, :],
                cell
                )
        vector1 = vector_pbc(
                configuration[..., self.i3, :],
                configuration[..., self.i1, :],
                cell
                )
        vector2 = vector_pbc(
                configuration[..., self.i3, :],
                configuration[..., self.i2, :],
                cell=cell
                )
        norm0 = _np.linalg.norm(vector0, axis=-1)[..., None]
        norm1 = _np.linalg.norm(vector1, axis=-1)[..., None]
        norm2 = _np.linalg.norm(vector2, axis=-1)[..., None]

        unit0 = vector0 / norm0
        unit1 = vector1 / norm1
        unit2 = vector2 / norm2

        plane1 = cross(unit1, unit2)

        phi1 = angle(vector1, vector2)  # no cell here

        scalar = dot(plane1, unit0) / _np.sin(phi1)

        return _np.arcsin(scalar)

    def _eval_der(self, configuration, cell=None):
        vector0 = vector_pbc(
                configuration[..., self.i3, :],
                configuration[..., self.i0, :],
                cell
                )
        vector1 = vector_pbc(
                configuration[..., self.i3, :],
                configuration[..., self.i1, :],
                cell
                )
        vector2 = vector_pbc(
                configuration[..., self.i3, :],
                configuration[..., self.i2, :],
                cell=cell
                )
        norm0 = _np.linalg.norm(vector0, axis=-1)[..., None]
        norm1 = _np.linalg.norm(vector1, axis=-1)[..., None]
        norm2 = _np.linalg.norm(vector2, axis=-1)[..., None]

        unit0 = vector0 / norm0
        unit1 = vector1 / norm1
        unit2 = vector2 / norm2

        plane0 = cross(unit0, unit1)
        plane1 = cross(unit1, unit2)
        plane2 = cross(unit2, unit0)

        phi1 = angle(vector1, vector2)[..., None]  # no cell here
        sin_phi1 = _np.sin(phi1)
        cos_phi1 = _np.cos(phi1)

        sinus = dot(plane1, unit0)[..., None] / _np.sin(phi1)
        cosinus = _np.cos(_np.arcsin(sinus))

        derivative = _np.zeros_like(configuration)
        derivative[..., self.i0, :] = \
            1./norm0/cosinus * (
                plane1/sin_phi1 - sinus*unit0
                )
        derivative[..., self.i1, :] = \
            1./norm1/cosinus * (
                plane2/sin_phi1 - sinus/sin_phi1**2 * (unit1 - cos_phi1*unit2)
                )
        derivative[..., self.i2, :] = \
            1./norm2/cosinus * (
                plane0/sin_phi1 - sinus/sin_phi1**2 * (unit2 - cos_phi1*unit1)
                )
        derivative[..., self.i3, :] = \
            - derivative[..., self.i0, :] \
            - derivative[..., self.i1, :] \
            - derivative[..., self.i2, :]

        return derivative


class Coord(_COLVAR):
    '''i0 along axis'''
    def __init__(self, i0, axis):
        self.label = f'{self.__class__.__name__}({i0}[{axis}])'
        self.i0 = i0
        self.axis = axis

    def _eval(self, configuration, cell=None):
        return configuration[..., self.i0, self.axis]

    def _eval_der(self, configuration, cell=None):
        derivative = _np.zeros_like(configuration)
        derivative[..., self.i0, self.axis] = 1.0
        return derivative


class Combination(_COLVAR):
    '''Collective variable as linear combination of other collective
       variables with optional weights'''
    def __init__(self, colvar_array, weights=None, label=None):
        n_colvars = len(colvar_array)
        if weights is None:
            weights = n_colvars * (1.,)

        self.colvar_array = tuple(colvar_array)
        self.weights = tuple(weights)
        self._clean()

    def _clean(self):
        colvar_array = ()
        weights = ()
        for _c, _w in zip(self.colvar_array, self.weights):
            if hasattr(_c, 'colvar_array'):
                colvar_array += _c.colvar_array
                weights += tuple([_w * _ww for _ww in _c.weights])
            else:
                colvar_array += (_c,)
                weights += (_w,)
        self.label = f'{self.__class__.__name__}('
        self.label += ' + '.join([f'({_w:+})*{_c.label}'
                                  for _c, _w in zip(colvar_array, weights)])
        self.label += ')'
        self.colvar_array = colvar_array
        self.weights = weights

    def _eval(self, configuration, cell=None):
        return _np.sum([_w * _c._eval(configuration, cell=None)
                        for _c, _w in zip(self.colvar_array, self.weights)],
                       axis=0)

    def _eval_der(self, configuration, cell=None):
        return _np.sum([_w * _c._eval_der(configuration, cell=None)
                        for _c, _w in zip(self.colvar_array, self.weights)],
                       axis=0)


class InternalCoordinates(list):
    '''       '''
    def __init__(self, items, *args):
        [_COLVAR._isinstance(_a, info='InternalCoordinates') for _a in items]
        super().__init__(items, *args)

    def __repr__(self):
        return 'InternalCoordinates: ' \
                + super().__repr__()

    def __setitem__(self, index, value):
        _COLVAR._isinstance(value, info='InternalCoordinates')
        super().__setitem__(index, value)

    def __add__(self, other, *args):
        [_COLVAR._isinstance(_a, info='InternalCoordinates') for _a in other]
        super().__iadd__(other, *args)
        return self

    def append(self, value):
        _COLVAR._isinstance(value, info='InternalCoordinates')
        super().append(value)

    def extend(self, items, *args):
        [_COLVAR._isinstance(_a, info='InternalCoordinates') for _a in items]
        super().extend(items, *args)

    def convert(self, positions, cell=None):
        '''Convert Cartesian data into Internal Coordinates.

           arguments:
               positions ... Cartesian positions,
                                 array of shape ([n_frames], n_atoms, 3)
               cell ... 6-dimensional cell vector: [a, b, c, al, be, ga]
                        (optional)

           returns:
               data array shape ([n_frames,], n_colvars)
            '''

        data = _np.array([_colvar.value(positions, cell=cell)
                          for _colvar in self])
        return data.T

    def Bmatrix_test(self, positions, cell=None):
        '''Return Jacobi matrix of the derivatives of all collective variables
           with respect to the Cartesian degrees of freedom given in
           configuration.

           arguments:
               positions ... Cartesian positions,
                                 array of shape ([n_frames], n_atoms, 3)
               cell ... 6-dimensional cell vector: [a, b, c, al, be, ga]
                        (optional)
           returns:
               array of shape ([n_frames,], n_colvars, n_atoms*3)
            '''

        n_atoms, three = positions.shape[-2:]
        n_colvars = len(self)
        data = _np.array([
            _colvar.derivative(positions, cell=cell) for _colvar in self
            ]).reshape((n_colvars, -1, n_atoms, 3))

        return _np.moveaxis(data, 1, 0)

    # def BMatrix(self):
    #     pass
