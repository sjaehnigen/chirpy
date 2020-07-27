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

import numpy as _np
import copy as _copy
from scipy.interpolate import interpn as _interpn
from scipy.integrate import simps as _simps
import warnings as _warnings

from .core import _CORE
from ..read.grid import cubeReader
from ..write.grid import cubeWriter
from ..physics.kspace import k_potential as _k_potential
from ..physics.classical_electrodynamics import _get_divrot
from ..physics import constants
from ..mathematics.algebra import rotate_griddata, rotate_vector
from ..mathematics.algebra import change_euclidean_basis as ceb


class ScalarField(_CORE):
    def __init__(self, *args, **kwargs):

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
                data, self.origin_au, self.cell_vec_au, pos_au, self.numbers, \
                    comments = cubeReader(fn, **kwargs)
                if data.shape[0] > 1:
                    _warnings.warn(
                        'Volume class does not (yet) support trajectory data!',
                        stacklevel=2)
                # No support of multiple frames for now
                self.data = data[0]
                self.pos_au = pos_au[0]
                self.comments = comments[0]

            elif fmt == 'npy':
                test = ScalarField.from_data(data=_np.load(fn), **kwargs)
                self.comments = test.comments
                self.origin_au = test.origin_au
                self.cell_vec_au = test.cell_vec_au
                self.pos_au = test.pos_au
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
        if kwargs.get('sparsity', 1) != 1:
            # python 3.8: use walrus
            self = self.sparse(kwargs.get('sparsity'))

    def _sync_class(self):
        self.n_x = self.data.shape[-3]
        self.n_y = self.data.shape[-2]
        self.n_z = self.data.shape[-1]
        try:
            self.n_atoms = self.pos_au.shape[0]
            if self.n_atoms != len(self.numbers):
                raise ValueError('List of atom numbers and atom positions do '
                                 'not match!')

            self.symbols = constants.numbers_to_symbols(self.numbers)
            self.voxel = _np.dot(self.cell_vec_au[0],
                                 _np.cross(self.cell_vec_au[1],
                                           self.cell_vec_au[2]))
        except AttributeError:
            pass

    def print_info(self):
        print('')
        print(77 * '–')
        print('%-12s' % self.__class__.__name__)
        print(77 * '–')
        print(' x '.join(map('{:d}'.format, self.data.shape[-3:])))
        print('%d Atoms' % self.n_atoms)
        # print('\n'.join(self.comments))
        print('\n')
        print(77 * '–')
        print('Origin (a.u.)'
              + ' '.join(map('{:10.5f}'.format, self.origin_au)))
        print(77 * '-')
        print(' grid vector (A) (a.u.) '
              + ' '.join(map('{:10.5f}'.format, self.cell_vec_au[0])))
        print(' grid vector (B) (a.u.) '
              + ' '.join(map('{:10.5f}'.format, self.cell_vec_au[1])))
        print(' grid vector (C) (a.u.) '
              + ' '.join(map('{:10.5f}'.format, self.cell_vec_au[2])))
        print(77 * '–')
        print('')

    @classmethod
    def from_object(cls, obj, **kwargs):
        '''Use kwargs to transfer new attribute values'''
        nargs = {}
        nargs.update(_copy.deepcopy(vars(obj)))
        nargs.update(kwargs)
        return cls.from_data(**nargs)

    @classmethod
    def from_domain(cls, domain, **kwargs):
        return cls.from_data(data=domain.expand(), **kwargs)

    @classmethod
    def from_data(cls, **kwargs):
        cell_vec_au = kwargs.get('cell_vec_au')
        data = kwargs.get('data')
        if any([cell_vec_au is None, data is None]):
            raise TypeError('Please give both, cell_vec_au and data!')
        obj = cls.__new__(cls)
        # quick workaround to find out if vectorfield
        if data.shape[0] == 3:
            obj.comments = 3 * [('no_comment', 'no_comment')]
        else:
            obj.comments = ('no_comment', 'no_comment')
        obj.origin_au = kwargs.get('origin_au', _np.zeros((3)))
        obj.pos_au = kwargs.get('pos_au', _np.zeros((0, 3)))
        obj.numbers = kwargs.get('numbers', _np.zeros((0, )))
        obj.cell_vec_au = cell_vec_au
        obj.data = data
        # Check for optional data
        # for key, value in kwargs.items():
        #     if not hasattr(obj, key): setattr(obj, key, value)
        obj._sync_class()
        return obj

    def __add__(self, other, factor=1):
        '''supports different data shapes'''
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
                                    - other.origin_au)
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
                'origin_au',
                'cell_vec_au',
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
                        RuntimeWarning, stacklevel=2
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
        '''If no norm is given, the method uses np.linalg.norm
        (give axis in kwargs).'''

        _N = norm
        if _N is None:
            _N = _np.linalg.norm(self.data, **kwargs)

        with _np.errstate(divide='ignore'):
            _N_inv = _np.where(_N < thresh, 0.0, _np.divide(1.0, _N))

        self.data *= _N_inv

    def grid(self):
        '''Return an empty copy of grid'''
        return _np.zeros(self.data.shape)

    def _rtransform(self, p):
        '''transform position (relative to origin) into grid index'''
        return ceb(_copy.deepcopy(p), self.cell_vec_au)

    def _ltransform(self, i):
        '''transform grid index into position'''
        return _np.einsum('ni, ji -> nj',
                          _copy.deepcopy(i),
                          self.cell_vec_au
                          )

    def pos_grid(self):
        '''Generate grid point coordinates (only for tetragonal cells)'''
        xaxis = _np.arange(0, self.n_x)
        yaxis = _np.arange(0, self.n_y)
        zaxis = _np.arange(0, self.n_z)

        pos_grid = _np.array(_np.meshgrid(xaxis, yaxis, zaxis, indexing='ij'))
        return _np.einsum(
                'inmo, ji -> jnmo',
                pos_grid,
                self.cell_vec_au) + self.origin_au[:, None, None, None]

    def sparse(self, sp, dims='xyz'):
        '''Returns a new object with sparse grid according to sp (integer).'''

        new = _copy.deepcopy(self)

        def _apply(_i):
            new.data = _np.moveaxis(_np.moveaxis(new.data, _i, 0)[::sp],
                                    0,
                                    _i)
            new.cell_vec_au[_i] *= sp

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
            self.origin_au = _copy.deepcopy(
                    self.origin_au + self._ltransform(
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

    def rotate(self, R, rotate_grid=False, **kwargs):
        '''Rotate entire object including atomic positions either by
           interpolation of data keeping grid points and cell vectors
           unchanged (default: rotate_grid=False), or by rotating cell
           vectors and origin (rotate_grid=True).
           R ... rotation matrix of shape (3, 3)
           '''
        _o = kwargs.get('rot_origin_au', self.origin_au)

        self.pos_au = rotate_vector(self.pos_au, R, origin=_o)

        if not rotate_grid:
            self.data = rotate_griddata(self.pos_grid(), self.data, R,
                                        origin=_o)

        else:
            _warnings.warn('Rotating grid may lead to problems with the '
                           'Gaussian Cube format convention!', stacklevel=2)
            self.cell_vec_au = rotate_vector(self.cell_vec_au, R)
            self.origin_au = rotate_vector(self.origin_au, R, origin=_o)

    def write(self, fn, **kwargs):
        fmt = kwargs.get('fmt', fn.split('.')[-1])
        attr = kwargs.get('attribute', 'data')
        if not hasattr(self, attr):
            raise AttributeError(
                              'Attribute %s not (yet) part of object!' % attr)
        if fmt == "cube":
            comments = kwargs.get('comments', self.comments)
            pos_au = kwargs.get('pos_au', self.pos_au)
            n_atoms = pos_au.shape[0]
            numbers = kwargs.get('numbers', self.numbers)
            # insert routine to get numbers from symbols: in derived class
            # constants.symbols_to_numbers(symbols)
            if len(numbers) != n_atoms:
                raise ValueError('Given numbers inconsistent with positions')
            data = getattr(self, attr)
            cubeWriter(fn,
                       comments,
                       numbers,
                       pos_au,
                       self.cell_vec_au,
                       data,
                       origin_au=self.origin_au)
        else:
            raise ValueError('Unknown format (Not implemented).')


class VectorField(ScalarField):
    def __init__(self, *args, **kwargs):
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
                    'origin_au',
                    'cell_vec_au',
                    'pos_au',
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

    def streamlines(self, p0,
                    sparse=4,
                    forward=True,
                    backward=True,
                    length=400,
                    timestep=0.5,
                    external_object=False,
                    **kwargs):
        '''pn...starting points of shape (n_points, 3)'''
        def get_value(p):
            return self._rtransform(_interpn(points,
                                             values,
                                             (p[0], p[1], p[2]),
                                             method='nearest',
                                             bounds_error=False,
                                             fill_value=None))

        dt = timestep
        ext = external_object
        if ext:
            ext_p0, ext_v = kwargs.get('ext_p'), kwargs.get('ext_v')

            if any([ext_p0 is None, ext_v is None]):
                _warnings.warn('Missing external object for set keyword! '
                               'Please give ext_p and ext_v.',
                               RuntimeWarning, stacklevel=2)
                ext = False

            if ext_p0.shape != ext_v.shape:
                _warnings.warn('External object with inconsistent ext_p and '
                               'ext_v! Skippping.',
                               RuntimeWarning, stacklevel=2)
                ext = False

        v_field = self.data[:, ::sparse, ::sparse, ::sparse]

        points = (_np.arange(0, self.n_x, sparse),
                  _np.arange(0, self.n_y, sparse),
                  _np.arange(0, self.n_z, sparse))
        values = _np.moveaxis(v_field, 0, -1)

        traj = list()
        ext_t = list()

        if backward:
            pn = self._rtransform(_copy.deepcopy(p0) - self.origin_au)
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
                pn -= vn * dt
                vn = get_value(pn.swapaxes(0, 1))
                traj.append(_np.concatenate((self._ltransform(pn),
                                             self._ltransform(vn)),
                                            axis=-1))
                if ext:
                    ext_p -= ext_v * dt
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
            pn = self._rtransform(_copy.deepcopy(p0) - self.origin_au)
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
                pn += vn * dt
                vn = get_value(pn.swapaxes(0, 1))
                traj.append(_np.concatenate((self._ltransform(pn),
                                             self._ltransform(vn)),
                                            axis=-1))
                if ext:
                    ext_p += ext_v * dt
                    ext_t.append(_np.concatenate((_copy.deepcopy(ext_p),
                                                  _copy.deepcopy(ext_v)),
                                                 axis=-1))

        result = {}
        result['streamlines'] = _np.array(traj)
        result['streamlines'][:, :, :3] += self.origin_au
        if ext:
            result['particles'] = _np.array(ext_t)

        return result

    def streamtubes(self):
        '''See notebook 24b'''
        pass

    @staticmethod
    def _helmholtz_components(data, cell_vec_au):
        div, rot = _get_divrot(data, cell_vec_au)
        V = _k_potential(div, _np.array(cell_vec_au))[1]/(4*_np.pi)
        A1 = _k_potential(rot[0], _np.array(cell_vec_au))[1]
        A2 = _k_potential(rot[1], _np.array(cell_vec_au))[1]
        A3 = _k_potential(rot[2], _np.array(cell_vec_au))[1]
        A = _np.array([A1, A2, A3])/(4*_np.pi)
        irrotational_field = -_np.array(_np.gradient(V,
                                                     cell_vec_au[0][0],
                                                     cell_vec_au[1][1],
                                                     cell_vec_au[2][2]))
        solenoidal_field = _get_divrot(A, cell_vec_au)[1]

        return irrotational_field, solenoidal_field, div, rot

    def divergence_and_rotation(self):
        self.div, self.rot = _get_divrot(self.data, self.cell_vec_au)

    def helmholtz_decomposition(self):
        irr = _copy.deepcopy(self)
        sol = _copy.deepcopy(self)
        hom = _copy.deepcopy(self)
        irr.data, sol.data, div, rot = self._helmholtz_components(
                                                   self.data, self.cell_vec_au)

        self.div = ScalarField(self, data=div)
        self.rot = ScalarField(self, data=rot)
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
            pos_au = kwargs.get('pos_au', self.pos_au)
            numbers = kwargs.get('numbers', self.numbers)
            cell_vec_au = kwargs.get('cell_vec_au', self.cell_vec_au)
            origin_au = kwargs.get('origin_au', self.origin_au)
            data = getattr(self, attr)
            cubeWriter(fn1,
                       comments[0],
                       numbers,
                       pos_au,
                       cell_vec_au,
                       data[0],
                       origin_au=origin_au)
            cubeWriter(fn2,
                       comments[1],
                       numbers,
                       pos_au,
                       cell_vec_au,
                       data[1],
                       origin_au=origin_au)
            cubeWriter(fn3,
                       comments[2],
                       numbers,
                       pos_au,
                       cell_vec_au,
                       data[2],
                       origin_au=origin_au)
        else:
            raise ValueError(f'Unknown format {fmt}.')
