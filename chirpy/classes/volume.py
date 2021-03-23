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
#  Copyright (c) 2010-2020, The ChirPy Developers.
#
#
#  Released under the GNU General Public Licence, v3
#
#   ChirPy is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published
#   by the Free Software Foundation, either version 3 of the License.
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
import copy as _copy
from scipy.interpolate import interpn as _interpn
from scipy.integrate import simps as _simps
from scipy.ndimage import gaussian_filter1d
import warnings as _warnings

from .core import _CORE
from .. import extract_keys
from ..config import ChirPyWarning
from ..read.grid import cubeReader
from ..write.grid import cubeWriter
from ..physics.kspace import k_potential as _k_potential
from ..physics import constants
from ..mathematics.algebra import rotate_griddata, rotate_vector
from ..mathematics.analysis import divrot
from ..topology import mapping as mp
from ..visualise import print_info


class ScalarField(_CORE):
    def __init__(self, *args, **kwargs):
        self._print_info = [print_info.print_cell]
        if len(args) > 1:
            raise TypeError("File reader of %s takes at most 1 argument!"
                            % self.__class__.__name__)

        elif len(args) == 1:
            fn = args[0]
            self._fn = fn
            fmt = kwargs.get('fmt', str(fn).split('.')[-1])
            if fmt == 'bz2':
                kwargs.update({'bz2': True})
                fmt = fn.split('.')[-2]

            self._fmt = fmt
            if fmt == "cube":
                data, self.origin_aa, self.cell_vec_aa, pos_aa, self.numbers, \
                    comments = cubeReader(fn, **kwargs)
                if data.shape[0] > 1:
                    _warnings.warn(
                        'Volume class does not (yet) support trajectory data!',
                        ChirPyWarning,
                        stacklevel=2)
                # No support of multiple frames for now
                self.data = data[0]
                self.pos_aa = pos_aa[0]
                self.comments = comments[0]

            elif fmt == 'npy':
                test = ScalarField.from_data(data=_np.load(fn), **kwargs)
                self.comments = test.comments
                self.origin_aa = test.origin_aa
                self.cell_vec_aa = test.cell_vec_aa
                self.pos_aa = test.pos_aa
                self.numbers = test.numbers
                self.data = test.data

            elif fmt == 'wfn':
                raise NotImplementedError('Format wfn not supported.')

            else:
                try:
                    self.__dict__ = self.__class__.from_object(
                                                   args[0], **kwargs).__dict__
                except TypeError:
                    try:
                        self.__dict__ = self.__class__.load(args[0]).__dict__
                    except (TypeError, AttributeError):
                        raise ValueError('Unknown format.')

        elif len(args) == 0:
            self.__dict__ = self.__class__.from_data(**kwargs).__dict__

        self._sync_class()
        if (sparse := kwargs.get('sparsity', 1)) != 1:
            self = self.sparse(sparse)

    def _sync_class(self):
        # --- backward compatibility to version < 0.17.1
        for _a in ['pos', 'cell_vec', 'origin']:
            self._print_info = [print_info.print_cell]
            if not hasattr(self, _a+'_aa') and hasattr(self, _a+'_au'):
                _warnings.warn('Since version 0.17.1. the default length unit '
                               'in ChirPy is '
                               f'angstrom. \'{_a}_au\' is '
                               'deprecated and should no longer be '
                               f'used; use \'{_a}_aa\' instead.',
                               FutureWarning, stacklevel=2)
                setattr(self, _a+'_aa',
                        getattr(self, _a+'_au')*constants.l_au2aa)
                delattr(self, _a+'_au')

        self.n_x = self.data.shape[-3]
        self.n_y = self.data.shape[-2]
        self.n_z = self.data.shape[-1]
        try:
            self.n_atoms = self.pos_aa.shape[0]
            if self.n_atoms != len(self.numbers):
                raise ValueError('List of atom positions and numbers do '
                                 'not match: %d, %d'
                                 % (self.n_atoms, len(self.numbers)))

            self.symbols = constants.numbers_to_symbols(self.numbers)
            self.voxel = _np.dot(self.cell_vec_aa[0],
                                 _np.cross(self.cell_vec_aa[1],
                                           self.cell_vec_aa[2]))
            self.cell_aa_deg = mp.get_cell_l_deg(
                                     self.cell_vec_aa,
                                     multiply=self.data.shape[-3:]
                                     )
        except AttributeError:
            pass

    @classmethod
    def from_object(cls, obj, **kwargs):
        '''Use kwargs to transfer new attribute values'''
        nargs = extract_keys(_copy.deepcopy(vars(obj)),
                             data=None, cell_vec_aa=None,
                             origin_aa=None, pos_aa=None, numbers=None)
        nargs.update(kwargs)
        return cls.from_data(**nargs)

    @classmethod
    def from_domain(cls, domain, **kwargs):
        return cls.from_data(data=domain.expand(), **kwargs)

    @classmethod
    def from_data(cls, data, cell_vec_aa,
                  origin_aa=None, pos_aa=None, numbers=None):
        obj = cls.__new__(cls)
        # --- quick workaround to find out if vectorfield
        if data.shape[0] == 3:
            obj.comments = 3 * [('no_comment', 'no_comment')]
        else:
            obj.comments = ('no_comment', 'no_comment')

        if origin_aa is None:
            origin_aa = _np.zeros((3))
        if pos_aa is None:
            pos_aa = _np.zeros((0, 3))
        if numbers is None:
            numbers = _np.zeros((0, ))

        obj.origin_aa = origin_aa
        obj.pos_aa = pos_aa
        obj.numbers = numbers
        obj.cell_vec_aa = cell_vec_aa
        obj.data = data

        # Check for optional data
        # for key, value in kwargs.items():
        #     if not hasattr(obj, key): setattr(obj, key, value)

        obj._sync_class()
        return obj

    def __add__(self, other, factor=1):
        '''Supports different data shapes.
           For equal grids use the faster way:
              new = obj.__class__.from_object(obj)
              new.data + data1 + data2 etc.
        '''
        new = _copy.deepcopy(self)
        points = (_np.arange(0, other.n_x),
                  _np.arange(0, other.n_y),
                  _np.arange(0, other.n_z))
        values = _np.moveaxis(
                     _np.moveaxis(
                         _np.moveaxis(
                             other.data, -1, 0
                             ), -1, 0
                         ), -1, 0
                     )
        p_trans = other._rtransform(_np.moveaxis(new.pos_grid(), 0, -1)
                                    - other.origin_aa)
        i_values = _np.moveaxis(
                     _np.moveaxis(
                         _np.moveaxis(
                             _interpn(
                                 points,
                                 values,
                                 p_trans,
                                 method='nearest',
                                 bounds_error=False,
                                 fill_value=None,
                                 ), 0, -1
                             ), 0, -1
                         ), 0, -1
                     )

        new.data += i_values * factor

        return new

    def __sub__(self, other):
        self = self.__add__(other, factor=-1)
        return self

    def _is_similar(self, other, strict=1, return_false=False):
        '''level of strictness: 1...similar, 2...very similar, 3...equal'''
        def _f_check(a):
            return [_BOOL for _BOOL in (
                (getattr(self, a) != getattr(other, a), )
                if isinstance(getattr(self, a), int)
                else (not _np.allclose(getattr(self, a), getattr(other, a)), )
                )][0]

        err_keys = [
                'origin_aa',
                'cell_vec_aa',
                'voxel',
                ]
        wrn_keys = [
                'n_atoms',
                'numbers',
                ]
        equ_keys = [
                'data',
                ]
        _ERR = list(map(_f_check, err_keys))
        _WRN = list(map(_f_check, wrn_keys))

        if any(_ERR):
            if return_false:
                return False
            raise ValueError('\n'.join(
                'Objects dissimilar in %s!'
                % _e for (_e, _B) in zip(err_keys, _ERR) if _B)
               )

        if any(_WRN):
            if strict == 1:
                with _warnings.catch_warnings():
                    _warnings.warn('\n'.join(
                        'Objects dissimilar in %s!'
                        % _e for (_e, _B) in zip(wrn_keys, _WRN) if _B),
                        ChirPyWarning, stacklevel=2
                       )
            else:
                if return_false:
                    return False
                raise ValueError('\n'.join(
                    'Objects dissimilar in %s!'
                    % _e for (_e, _B) in zip(wrn_keys, _WRN) if _B)
                   )

        if strict == 3:
            _EQU = list(map(_f_check, equ_keys))
            if any(_EQU):
                if return_false:
                    return False
                raise ValueError('\n'.join(
                    'Objects differing in %s!'
                    % _e for (_e, _B) in zip(equ_keys, _EQU) if _B)
                   )
        return True

    def integral(self):
        return self.voxel*_simps(_simps(_simps(self.data)))

    def normalise(self, norm=None, thresh=1.E-8, **kwargs):
        '''Norm has to be a ScalarField object (can be of different shape) or
           float.
           If no norm is given, the method uses np.linalg.norm of vector field
           (give axis in kwargs).'''

        # --- create empty object with the correct grid
        _N = ScalarField.from_object(self, data=self.grid())
        if norm is None:
            _N.data = _np.linalg.norm(self.data, **kwargs)
        elif isinstance(norm, float):
            _N.data += norm
        else:
            # --- __add__ interpolates different grids
            _N += norm

        with _np.errstate(divide='ignore'):
            _N_inv = _np.where(_N.data < thresh, 0.0, _np.divide(1.0, _N.data))

        self.data *= _N_inv

    def grid(self):
        '''Return an empty copy of grid'''
        return _np.zeros(self.data.shape)

    def _rtransform(self, p):
        '''transform position (relative to origin) into grid index'''
        return mp.get_cell_coordinates(
                p,
                mp.get_cell_l_deg(self.cell_vec_aa)
                )

    def _ltransform(self, i):
        '''transform grid index into position'''
        return mp.get_cartesian_coordinates(
                i,
                mp.get_cell_l_deg(self.cell_vec_aa)
                )

    def ind_grid(self):
        '''Return grid point indices'''
        xaxis = _np.arange(0, self.n_x)
        yaxis = _np.arange(0, self.n_y)
        zaxis = _np.arange(0, self.n_z)

        return _np.array(_np.meshgrid(xaxis, yaxis, zaxis, indexing='ij'))

    def pos_grid(self):
        '''Return grid point coordinates.
           Slow.'''
        pos_grid = self.ind_grid()

        return _np.einsum(
                'inmo, ji -> jnmo',
                pos_grid,
                self.cell_vec_aa) + self.origin_aa[:, None, None, None]

    def smoothen(self, sigma):
        '''Apply a sequence of 1D Gaussian filters to grid data'''
        self.data = gaussian_filter1d(self.data, sigma, axis=-1)
        self.data = gaussian_filter1d(self.data, sigma, axis=-2)
        self.data = gaussian_filter1d(self.data, sigma, axis=-3)

    def sparse(self, sp, dims='xyz'):
        '''Returns a new object with sparse grid according to sp (integer).'''

        new = _copy.deepcopy(self)

        def _apply(_i):
            new.data = _np.moveaxis(_np.moveaxis(new.data, _i, 0)[::sp],
                                    0,
                                    _i)
            new.cell_vec_aa[_i] *= sp

        if 'x' in dims:
            _apply(-3)
        if 'y' in dims:
            _apply(-2)
        if 'z' in dims:
            _apply(-1)

        new._sync_class()

        return new

    def crop(self, r, dims='xyz'):
        '''r tuple of len=len(dims)'''
        def _apply(_i, _r):
            self.data = _np.moveaxis(_np.moveaxis(
                                        self.data,
                                        _i,
                                        0
                                        )[_r[0]:_r[1]],
                                     0,
                                     _i)
            self.origin_aa = _copy.deepcopy(
                    self.origin_aa + self._ltransform(
                        _np.roll(_np.array([[1.0, 0.0, 0.0]]), _i) * _r[0])[0]
                    )

        for _r, _dim in zip(r, dims):
            if 'x' in _dim:
                _apply(-3, _r)
            if 'y' in _dim:
                _apply(-2, _r)
            if 'z' in _dim:
                _apply(-1, _r)

        self._sync_class()

    def auto_crop(self, thresh=1.E-3, dims='xyz', dry_run=False):
        '''Crop after threshold (only for closed surfaces).
           dry run: return crop tuple without actually cropping
           '''
        def _get_ab(d, axis):
            if thresh < _np.amax(d):
                upper = _np.array(d.shape)[axis] -\
                        _np.amin(_np.array(d.shape) - _np.argwhere(d > thresh),
                                 axis=0)[axis]
                lower = _np.amin(_np.argwhere(d > thresh), axis=0)[axis]
                return lower, upper
            else:
                return 0, _np.array(d.shape)[axis]

        r = []
        if 'x' in dims:
            r.append(_get_ab(_np.abs(self.data), -3))
        if 'y' in dims:
            r.append(_get_ab(_np.abs(self.data), -2))
        if 'z' in dims:
            r.append(_get_ab(_np.abs(self.data), -1))

        if not dry_run:
            self.crop(tuple(r), dims=dims)

        return tuple(r)

    def rotate(self, R, rotate_grid=False, rot_origin_aa=None):
        '''Rotate entire object including atomic positions either by
           interpolation of data keeping grid points and cell vectors
           unchanged (default: rotate_grid=False), or by rotating cell
           vectors and origin (rotate_grid=True).
           R ... rotation matrix of shape (3, 3)
           '''
        if (_o := rot_origin_aa) is None:
            _o = self.origin_aa

        self.pos_aa = rotate_vector(self.pos_aa, R, origin=_o)

        if not rotate_grid:
            self.data = rotate_griddata(self.pos_grid(), self.data, R,
                                        origin=_o)

        else:
            _warnings.warn('Rotating grid may lead to problems with the '
                           'Gaussian Cube format convention!',
                           ChirPyWarning, stacklevel=2)
            self.cell_vec_aa = rotate_vector(self.cell_vec_aa, R)
            self.origin_aa = rotate_vector(self.origin_aa, R, origin=_o)

    def write(self, fn, **kwargs):
        fmt = kwargs.get('fmt', fn.split('.')[-1])
        attr = kwargs.get('attribute', 'data')
        if not hasattr(self, attr):
            raise AttributeError(
                              'Attribute %s not (yet) part of object!' % attr)
        if fmt == "cube":
            comments = kwargs.get('comments', self.comments)
            pos_aa = kwargs.get('pos_aa', self.pos_aa)
            n_atoms = pos_aa.shape[0]
            numbers = kwargs.get('numbers', self.numbers)
            # insert routine to get numbers from symbols: in derived class
            # constants.symbols_to_numbers(symbols)
            if len(numbers) != n_atoms:
                raise ValueError('Given numbers inconsistent with positions')
            data = getattr(self, attr)
            cubeWriter(fn,
                       comments,
                       numbers,
                       pos_aa,
                       self.cell_vec_aa,
                       data,
                       origin_aa=self.origin_aa)
        else:
            raise ValueError('Unknown format (Not implemented).')


class VectorField(ScalarField):
    def __init__(self, *args, **kwargs):
        self._print_info = [print_info.print_cell]
        if len(args) not in [0, 1, 3]:
            raise TypeError(
                    "File reader of %s takes only 0, 1, or 3 arguments!"
                    % self.__class__.__name__)
        elif len(args) == 3:
            buf_x = ScalarField(args[0], **kwargs)
            buf_y = ScalarField(args[1], **kwargs)
            buf_z = ScalarField(args[2], **kwargs)
            self._join_scalar_fields(buf_x, buf_y, buf_z)
            del buf_x, buf_y, buf_z

        elif len(args) == 1:
            try:
                self.__dict__ = self.__class__.load(args[0]).__dict__
            except (TypeError, AttributeError):
                try:
                    self.__dict__ = self.__class__.from_object(
                                            args[0], **kwargs).__dict__
                except TypeError:
                    raise ValueError('Unknown format.')

        elif len(args) == 0:
            self.__dict__ = self.__class__.from_data(**kwargs).__dict__
        self._sync_class()

    def _join_scalar_fields(self, x, y, z):
        '''x, y, z ... ScalarField objects'''
        if x._is_similar(y, strict=2) and x._is_similar(z, strict=2):
            self._fn1, self._fn2, self._fn3 = x._fn, y._fn, z._fn
            self.comments = _np.array([x.comments, y.comments, z.comments])
            self.data = _np.array([x.data, y.data, z.data])
            for _a in [
                    'origin_aa',
                    'cell_vec_aa',
                    'pos_aa',
                    'n_atoms',
                    'numbers',
                    'voxel',
                    ]:
                setattr(self, _a, getattr(x, _a))

    @staticmethod
    def _ip2ind(ip, F):
        return _np.unravel_index(ip, F.shape[1:])

    @staticmethod
    def _read_vec(F, ip):
        _slc = (slice(None), ) \
               + tuple([_i for _i in VectorField._ip2ind(ip, F)])
        return F[_slc]

    @staticmethod
    def _write_vec(F, ip, V):
        _slc = (slice(None), ) \
               + tuple([_i for _i in VectorField._ip2ind(ip, F)])
        F[_slc] = V

    def grid(self):
        '''Return an empty copy of grid'''
        return _np.zeros(self.data.shape[1:])

    def rotate(self, *args, **kwargs):
        raise NotImplementedError(
                'Use the ScalarField method for each component!')

    def streamlines(self, starting_points_aa,
                    sparse=4,
                    forward=True,
                    backward=True,
                    length=400,
                    timestep_fs=0.5,
                    external_object=False,
                    ext_pos_aa=None,
                    ext_vel_au=None,
                    # unit_vel='au',
                    ):
        '''Compute streamlines from given starting points.

           vector field data and velocities in "velocity atomic units"
           timestep in fs

           starting_points_aa ... array of shape (n_points, 3) in angstrom
           Output velocities are in atomic units.
           '''
        def get_value(p):
            return self._rtransform(_interpn(points,
                                             values,
                                             (p[0], p[1], p[2]),
                                             method='nearest',
                                             bounds_error=False,
                                             fill_value=None))

        # --- Todo: Centralise
        # if unit_vel == 'au':
        _uv = constants.v_au2aaperfs
        # else:
        #     raise NotImplementedError('Does not support velocity unit other'
        #                              'than \'au\'')

        p0 = _copy.deepcopy(starting_points_aa)
        dt = timestep_fs
        ext = external_object
        if ext:
            if any([ext_pos_aa is None, ext_vel_au is None]):
                _warnings.warn('Missing external object for set keyword! '
                               'Please give ext_p and ext_v.',
                               ChirPyWarning, stacklevel=2)
                ext = False

            if ext_pos_aa.shape != ext_vel_au.shape:
                _warnings.warn('External object with inconsistent ext_p and '
                               'ext_v! Skippping.',
                               ChirPyWarning, stacklevel=2)
                ext = False

        if ext:
            ext_p0 = _copy.deepcopy(ext_pos_aa)
            ext_v = _copy.deepcopy(ext_vel_au)

        v_field = self.data[:, ::sparse, ::sparse, ::sparse]

        points = (_np.arange(0, self.n_x, sparse),
                  _np.arange(0, self.n_y, sparse),
                  _np.arange(0, self.n_z, sparse))
        values = _np.moveaxis(v_field, 0, -1)

        traj = list()
        ext_t = list()

        if backward:
            pn = self._rtransform(_copy.deepcopy(p0) - self.origin_aa)
            vn = get_value(pn.swapaxes(0, 1))
            traj.append(_np.concatenate((self._ltransform(pn),
                                         self._ltransform(vn)),
                                        axis=-1))
            if ext:
                ext_p = _copy.deepcopy(ext_p0)
                ext_t.append(_np.concatenate((_copy.deepcopy(ext_p),
                                              _copy.deepcopy(ext_v)),
                                             axis=-1))

            for t in range(length):
                pn -= vn * _uv * dt
                vn = get_value(pn.swapaxes(0, 1))
                traj.append(_np.concatenate((self._ltransform(pn),
                                             self._ltransform(vn)),
                                            axis=-1))
                if ext:
                    ext_p -= ext_v * _uv * dt
                    ext_t.append(_np.concatenate((_copy.deepcopy(ext_p),
                                                  _copy.deepcopy(ext_v)),
                                                 axis=-1))
            if forward:
                traj = traj[1:][::-1]
                if ext:
                    ext_t = ext_t[1:][::-1]
            else:
                traj = traj[::-1]
                if ext:
                    ext_t = ext_t[::-1]

        if forward:
            pn = self._rtransform(_copy.deepcopy(p0) - self.origin_aa)
            vn = get_value(pn.swapaxes(0, 1))
            traj.append(_np.concatenate((self._ltransform(pn),
                                         self._ltransform(vn)),
                                        axis=-1))
            if ext:
                ext_p = _copy.deepcopy(ext_p0)
                ext_t.append(_np.concatenate((_copy.deepcopy(ext_p),
                                              _copy.deepcopy(ext_v)),
                                             axis=-1))

            for t in range(length):
                pn += vn * _uv * dt
                vn = get_value(pn.swapaxes(0, 1))
                traj.append(_np.concatenate((self._ltransform(pn),
                                             self._ltransform(vn)),
                                            axis=-1))
                if ext:
                    ext_p += ext_v * _uv * dt
                    ext_t.append(_np.concatenate((_copy.deepcopy(ext_p),
                                                  _copy.deepcopy(ext_v)),
                                                 axis=-1))

        result = {}
        result['streamlines'] = _np.array(traj)
        result['streamlines'][:, :, :3] += self.origin_aa
        if ext:
            result['particles'] = _np.array(ext_t)

        return result

    def streamtubes(self):
        '''See notebook 24b'''
        pass

    @staticmethod
    def _helmholtz_components(data, cell_vec_aa):
        div, rot = divrot(data, cell_vec_aa)
        V = _k_potential(div, _np.array(cell_vec_aa))[1]/(4*_np.pi)
        A1 = _k_potential(rot[0], _np.array(cell_vec_aa))[1]
        A2 = _k_potential(rot[1], _np.array(cell_vec_aa))[1]
        A3 = _k_potential(rot[2], _np.array(cell_vec_aa))[1]
        A = _np.array([A1, A2, A3])/(4*_np.pi)
        irrotational_field = -_np.array(_np.gradient(V,
                                                     cell_vec_aa[0][0],
                                                     cell_vec_aa[1][1],
                                                     cell_vec_aa[2][2]))
        solenoidal_field = divrot(A, cell_vec_aa)[1]

        return irrotational_field, solenoidal_field, div, rot

    def divergence_and_rotation(self):
        self.div, self.rot = divrot(self.data, self.cell_vec_aa)

    def helmholtz_decomposition(self):
        irr = self.__class__.from_object(self)
        sol = self.__class__.from_object(self)
        hom = self.__class__.from_object(self)
        irr.data, sol.data, div, rot = self._helmholtz_components(
                                                   self.data, self.cell_vec_aa)

        self.div = ScalarField(self, data=div)
        self.rot = VectorField(self, data=rot)
        hom.data = self.data - irr.data - sol.data
        self.irrotational_field = irr
        self.solenoidal_field = sol
        self.homogeneous_field = hom

    def write(self, fn1, fn2, fn3, **kwargs):
        '''Generalise this routine with autodetection for scalar and velfield,
        since div j is a scalar field, but attr of vec field class.'''
        fmt = kwargs.get('fmt', fn1.split('.')[-1])
        attr = kwargs.get('attribute', 'data')
        if fmt == "cube":
            comments = kwargs.get('comments', self.comments)
            pos_aa = kwargs.get('pos_aa', self.pos_aa)
            numbers = kwargs.get('numbers', self.numbers)
            cell_vec_aa = kwargs.get('cell_vec_aa', self.cell_vec_aa)
            origin_aa = kwargs.get('origin_aa', self.origin_aa)
            data = getattr(self, attr)
            cubeWriter(fn1,
                       comments[0],
                       numbers,
                       pos_aa,
                       cell_vec_aa,
                       data[0],
                       origin_aa=origin_aa)
            cubeWriter(fn2,
                       comments[1],
                       numbers,
                       pos_aa,
                       cell_vec_aa,
                       data[1],
                       origin_aa=origin_aa)
            cubeWriter(fn3,
                       comments[2],
                       numbers,
                       pos_aa,
                       cell_vec_aa,
                       data[2],
                       origin_aa=origin_aa)
        else:
            raise ValueError(f'Unknown format {fmt}.')
