#!/usr/bin/env python
# ------------------------------------------------------
#
#  ChirPy 0.1
#
#  https://hartree.chimie.ens.fr/sjaehnigen/ChirPy.git
#
#  2010-2016 Arne Scherrer
#  2014-2019 Sascha Jähnigen
#
#
# ------------------------------------------------------

import numpy as np
import copy
from scipy.interpolate import griddata
from scipy.integrate import simps

from ..readers.volume import cubeReader
from ..writers.volume import cubeWriter
from ..physics.kspace import k_potential
from ..physics.classical_electrodynamics import _get_divrot

# Convention Warning:
# cell_au is cell_vec_au


class ScalarField():
    def __init__(self, *args, **kwargs):

        if len(args) > 1:
            raise TypeError("File reader of %s takes at most 1 argument!"
                            % self.__class__.__name__)

        elif len(args) == 1:
            self.fn = args[0]
            self.fmt = kwargs.get('fmt', self.fn.split('.')[-1])
            if self.fmt == "cube":
                buf = cubeReader(self.fn)
                self.comments = buf['comments'].strip()
                self.origin_au = np.array(buf['origin_au'])
                self.cell_au = np.array(buf['cell_au'])  # deprecated
                self.cell_vec_au = np.array(buf['cell_au'])
                self.pos_au = np.array(buf['coords_au'])
                self.numbers = np.array(buf['numbers'])
                self.data = buf['volume_data']

            elif self.fmt == 'npy':
                test = ScalarField.from_data(data=np.load(self.fn), **kwargs)
                # I do not know why, but this has to be explicit
                # otherwise self remains empty
                self.comments = test.comments
                self.origin_au = test.origin_au
                self.cell_au = test.cell_vec_au  # deprecated
                self.cell_vec_au = test.cell_vec_au
                self.pos_au = test.pos_au
                self.numbers = test.numbers
                self.data = test.data

            elif self.fmt == 'wfn':
                raise NotImplementedError('Format wfn not supported.')

            else:
                raise AttributeError('Unknown format.')

        elif len(args) == 0:
            if kwargs.get("fmt") is not None:  # deprecated but still in use
                raise NotImplementedError("Use %s.from_data()!"
                                          % self.__class__.__name__)

        self._sync_class()

    def _sync_class(self):
        try:
            self.n_atoms = self.pos_au.shape[0]
            if self.n_atoms != self.numbers.shape[0]:
                raise ValueError('ERROR: List of atom numbers and atom \
                                 positions do not match!')

            self.voxel = np.dot(self.cell_vec_au[0],
                                np.cross(self.cell_vec_au[1],
                                         self.cell_vec_au[2]))
        except AttributeError:
            pass

    def print_info(self):
        # Work in progress...
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
        print(' cell(A) (a.u.) '
              + ' '.join(map('{:10.5f}'.format, self.cell_vec_au[0])))
        print(' cell(B) (a.u.) '
              + ' '.join(map('{:10.5f}'.format, self.cell_vec_au[1])))
        print(' cell(C) (a.u.) '
              + ' '.join(map('{:10.5f}'.format, self.cell_vec_au[2])))
        print(77 * '–')
        print('')

    @classmethod
    def from_object(cls, obj):
        return cls.from_data(**vars(copy.deepcopy(obj)))

    @classmethod
    def from_domain(cls, domain, **kwargs):
        return cls.from_data(data=domain.expand(), **kwargs)

    @classmethod
    def from_data(cls, **kwargs):
        cell_vec_au = kwargs.get('cell_vec_au', np.empty((0)))
        data = kwargs.get('data', np.empty((0)))
        if any([cell_vec_au.size == 0, data.size == 0]):
            raise AttributeError('ERROR: Please give both, \
cell_vec_au and data!')
        obj = cls()
        # quick workaround to find out if vectorfield
        if data.shape[0] == 3:
            obj.comments = 3*['no_comment\nno_comment']
        else:
            obj.comments = 'no_comment\nno_comment'
        obj.origin_au = kwargs.get('origin_au', np.zeros((3)))
        obj.pos_au = kwargs.get('pos_au', np.zeros((0, 3)))
        obj.numbers = kwargs.get('numbers', np.zeros((0, )))
        obj.cell_au = cell_vec_au  # deprecated
        obj.cell_vec_au = cell_vec_au
        obj.data = data
        # Check for optional data
        # for key, value in kwargs.items():
        #     if not hasattr(obj, key): setattr(obj, key, value)
        obj._sync_class()
        return obj

    def __add__(self, other):
        self._is_similar(other)
        new = copy.deepcopy(self)
        new.data += other.data
        return new

    def __iadd__(self, other):
        self._is_similar(other)
        self.data += other.data
        return self

    def _is_similar(self, other, strict=1, return_false=False):
        '''level of strictness: 1...similar, 2...very similar, 3...equal'''
        def _f_check(a):
            return [_BOOL for _BOOL in (
                (getattr(self, a) != getattr(other, a), )
                if isinstance(getattr(self, a), int)
                else (not np.allclose(getattr(self, a), getattr(other, a)), )
                )][0]

        err_keys = [
                'origin_au',
                'cell_au',
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
        _EQU = list(map(_f_check, equ_keys))

        if any(_ERR):
            if return_false:
                return False
            raise ValueError('\n'.join(
                'ERROR: objects dissimilar in %s!'
                % _e for (_e, _B) in zip(err_keys, _ERR) if _B)
               )

        if any(_WRN):
            if strict == 1:
                print('\n'.join(
                    'WARNING: objects dissimilar in %s!'
                    % _e for (_e, _B) in zip(wrn_keys, _WRN) if _B)
                   )
            else:
                if return_false:
                    return False
                raise ValueError('\n'.join(
                    'ERROR: objects dissimilar in %s!'
                    % _e for (_e, _B) in zip(wrn_keys, _WRN) if _B)
                   )

        if any(_EQU) and strict == 3:
            if return_false:
                return False
            raise ValueError('\n'.join(
                'ERROR: objects differing in %s!'
                % _e for (_e, _B) in zip(equ_keys, _EQU) if _B)
               )

        return True

    def integral(self):
        return self.voxel*simps(simps(simps(self.data)))

    def normalise(self, **kwargs):
        '''If no norm is given, the method uses np.linalg.norm
        (give axis in kwargs).'''
        # Do we need an upper threshold?
        _N = kwargs.pop("norm")
        if _N is None:
            _N = np.linalg.norm(self.data, **kwargs)
        thresh = kwargs.pop("tresh", 1.E-8)

        with np.errstate(divide='ignore'):
            _N_inv = np.where(_N < thresh, 0.0, np.divide(1.0, _N))

        self.data *= _N_inv

    def grid(self):
        '''Return an empty copy of grid'''
        return np.zeros(self.data.shape)

    def pos_grid(self):
        '''Generate grid point coordinates (only for tetragonal cells)'''
        self.n_x = self.data.shape[-3]
        self.n_y = self.data.shape[-2]
        self.n_z = self.data.shape[-1]
        xaxis = self.cell_au[0, 0]*np.arange(0, self.n_x) + self.origin_au[0]
        yaxis = self.cell_au[1, 1]*np.arange(0, self.n_y) + self.origin_au[1]
        zaxis = self.cell_au[2, 2]*np.arange(0, self.n_z) + self.origin_au[2]

        return np.array(np.meshgrid(xaxis, yaxis, zaxis, indexing='ij'))

    # copy to new object?
    def sparsity(self, sp, **kwargs):
        '''sp int'''
        dims = kwargs.get('dims', 'xyz')

        def _apply(_i):
            self.data = np.moveaxis(np.moveaxis(self.data, _i, 0)[::sp], 0, _i)
            self.cell_au[_i] *= sp
            self.cell_vec_au[_i] *= sp

        if 'x' in dims:
            _apply(-3)
        if 'y' in dims:
            _apply(-2)
        if 'z' in dims:
            _apply(-1)
        self.voxel = np.dot(self.cell_vec_au[0],
                            np.cross(self.cell_vec_au[1],
                                     self.cell_vec_au[2]))

    # only symmetric crop; ToDo: routine for centering + crop
    def crop(self, r, **kwargs):
        dims = kwargs.get('dims', 'xyz')

        def _apply(_i):
            self.data = np.moveaxis(np.moveaxis(self.data, _i, 0)[r:-r], 0, _i)
            self.origin_au[_i] += self.cell_au[_i, _i] * r

        if 'x' in dims:
            _apply(-3)
        if 'y' in dims:
            _apply(-2)
        if 'z' in dims:
            _apply(-1)

    def auto_crop(self, **kwargs):
        '''crop after threshold (default: ...)'''
        thresh = kwargs.get('thresh', 1.E-3)
        a = np.amin(np.array(self.data.shape)
                    - np.argwhere(np.abs(self.data) > thresh))
        b = np.amin(np.argwhere(np.abs(self.data) > thresh))
        self.crop(min(a, b))

        return min(a, b)

    def write(self, fn, **kwargs):
        '''Generalise this routine with autodetection for scalar and velfield,
        since div j is a scalar field, but attr of vec field class.'''

        fmt = kwargs.get('fmt', fn.split('.')[-1])
        attr = kwargs.get('attribute', 'data')
        if not hasattr(self, attr):
            raise Exception('ERROR: Attribute %s not (yet) part of object!'
                            % attr)
        if fmt == "cube":
            comment1 = kwargs.get('comment1', self.comments.split('\n')[0])
            comment2 = kwargs.get('comment2', self.comments.split('\n')[1])
            pos_au = kwargs.get('pos_au', self.pos_au)
            n_atoms = pos_au.shape[0]
            numbers = kwargs.get('numbers', self.numbers)
            # insert routine to get numbers from symbols: in derived class
            # constants.symbols_to_numbers(symbols)
            if numbers.shape[0] != n_atoms:
                raise Exception('ERROR: Given numbers inconsistent \
                                with positions')
            cell_au = kwargs.get('cell_au', self.cell_au)
            origin_au = kwargs.get('origin_au', self.origin_au)
            data = getattr(self, attr)
            cubeWriter(fn,
                       comment1,
                       comment2,
                       numbers,
                       pos_au,
                       cell_au,
                       data,
                       origin=origin_au)
        else:
            raise Exception('Unknown format (Not implemented).')


class VectorField(ScalarField):
    def __init__(self, *args, **kwargs):
        if len(args) not in [0, 3]:
            raise TypeError(
                    "File reader of %s takes exactly zero or three arguments!"
                    % self.__class__.__name__)
        elif len(args) == 3:
            buf_x = ScalarField(args[0], **kwargs)
            buf_y = ScalarField(args[1], **kwargs)
            buf_z = ScalarField(args[2], **kwargs)
            self._join_scalar_fields(buf_x, buf_y, buf_z)
            del buf_x, buf_y, buf_z
        elif len(args) == 0:
            if hasattr(self, "fmt"):  # deprecated but still in use
                self = self.__class__.from_data(**kwargs)

    def _join_scalar_fields(self, x, y, z):
        '''x, y, z ... ScalarField objects'''
        if x._is_similar(y, strict=2) and x._is_similar(z, strict=2):
            self.fn1, self.fn2, self.fn3 = x.fn, y.fn, z.fn
            self.comments = np.array([x.comments, y.comments, z.comments])
            self.data = np.array([x.data, y.data, z.data])
            for _a in [
                    'origin_au',
                    'cell_au',  # deprecated
                    'cell_vec_au',
                    'pos_au',
                    'n_atoms',
                    'numbers',
                    'voxel',
                    ]:
                setattr(self, _a, getattr(x, _a))

    @staticmethod
    def _ip2ind(ip, F):
        return np.unravel_index(ip, F.shape[1:])

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

    def streamlines(self, p0, **kwargs):
        '''pn...starting points of shape (n_points, 3)'''
        def get_value(p):
            return griddata(points,
                            values,
                            (p[0], p[1], p[2]),
                            method='nearest')

        sparse = kwargs.get('sparse', 4)
        fw = kwargs.get('forward', True)
        bw = kwargs.get('backward', True)
        L = kwargs.get('length', 400)
        dt = kwargs.get('timestep', 0.5)
        ext = kwargs.get('external_object', False)
        if ext:
            ext_p0, ext_v = kwargs.get('ext_p'), kwargs.get('ext_v')

            if any([ext_p0 is None, ext_v is None]):
                print('WARNING: Missing external object for set keyword! \
                       Please give ext_p and ext_v.')
                ext = False

            if ext_p0.shape != ext_v.shape:
                print('WARNING: External object with inconsistent \
                       ext_p and ext_v! Skippping.')
                ext = False

        pos_grid = self.pos_grid()[:, ::sparse, ::sparse, ::sparse]
        v_field = self.data[:, ::sparse, ::sparse, ::sparse]
        gl_norm = np.amax(np.linalg.norm(v_field, axis=0))
        ds = np.linalg.norm(self.cell_au, axis=1)

        points = np.array(
                [pos_grid[0].ravel(), pos_grid[1].ravel(), pos_grid[2].ravel()]
                ).swapaxes(0, 1)

        values = np.array(
                [v_field[0].ravel(), v_field[1].ravel(), v_field[2].ravel()]
                ).swapaxes(0, 1)

        traj = list()
        ext_t = list()

        if bw:
            pn = copy.deepcopy(p0)
            vn = get_value(p0.swapaxes(0, 1))
            traj.append(np.concatenate((copy.deepcopy(pn), copy.deepcopy(vn)),
                                       axis=-1))
            if ext:
                ext_p = copy.deepcopy(ext_p0)
                ext_t.append(np.concatenate((copy.deepcopy(ext_p),
                                             copy.deepcopy(ext_v)),
                                            axis=-1))

            for t in range(L):
                pn -= vn/gl_norm*ds[None]*dt
                vn = get_value(pn.swapaxes(0, 1))
                traj.append(np.concatenate((copy.deepcopy(pn),
                                            copy.deepcopy(vn)),
                                           axis=-1))
                if ext:
                    ext_p -= ext_v/gl_norm*ds*dt
                    ext_t.append(np.concatenate((copy.deepcopy(ext_p),
                                                 copy.deepcopy(ext_v)),
                                                axis=-1))
            if fw:
                traj = traj[1:][::-1]
                if ext:
                    ext_t = ext_t[1:][::-1]
            else:
                traj = traj[::-1]
                if ext:
                    ext_t = ext_t[::-1]

        if fw:
            pn = copy.deepcopy(p0)
            vn = get_value(pn.swapaxes(0, 1))
            traj.append(np.concatenate((copy.deepcopy(pn),
                                        copy.deepcopy(vn)),
                                       axis=-1))
            if ext:
                ext_p = copy.deepcopy(ext_p0)
                ext_t.append(np.concatenate((copy.deepcopy(ext_p),
                                             copy.deepcopy(ext_v)),
                                            axis=-1))

            for t in range(L):
                pn += vn/gl_norm*ds[None]*dt
                vn = get_value(pn.swapaxes(0, 1))
                traj.append(np.concatenate((copy.deepcopy(pn),
                                            copy.deepcopy(vn)),
                                           axis=-1))
                if ext:
                    ext_p += ext_v/gl_norm*ds*dt
                    ext_t.append(np.concatenate((copy.deepcopy(ext_p),
                                                 copy.deepcopy(ext_v)),
                                                axis=-1))

        if ext:
            return np.array(traj), np.array(ext_t)
        else:
            return np.array(traj)

    def streamtubes(self):
        '''See notebook 24b'''
        pass

    @staticmethod
    def _get_helmholtz_components(data, cell_au):
        div, rot = _get_divrot(data, cell_au)
        V = k_potential(div, np.array(cell_au))[1]/(4*np.pi)
        A1 = k_potential(rot[0], np.array(cell_au))[1]
        A2 = k_potential(rot[1], np.array(cell_au))[1]
        A3 = k_potential(rot[2], np.array(cell_au))[1]
        A = np.array([A1, A2, A3])/(4*np.pi)
        irrotational_field = -np.array(np.gradient(V,
                                                   cell_au[0][0],
                                                   cell_au[1][1],
                                                   cell_au[2][2]))
        solenoidal_field = _get_divrot(A, cell_au)[1]

        return irrotational_field, solenoidal_field, div, rot

    def divergence_and_rotation(self):
        self.div, self.rot = _get_divrot(self.data, self.cell_au)

    def helmholtz_decomposition(self):
        self.irrotational_field,\
                self.solenoidal_field,\
                self.div,\
                self.rot\
                =\
                self._get_helmholtz_components(self.data, self.cell_au)

    def write(self, fn1, fn2, fn3, **kwargs):
        '''Generalise this routine with autodetection for scalar and velfield,
        since div j is a scalar field, but attr of vec field class.'''
        fmt = kwargs.get('fmt', fn1.split('.')[-1])
        attr = kwargs.get('attribute', 'data')
        if fmt == "cube":
            comment1 = kwargs.get('comment1',
                                  [c.split('\n')[0] for c in self.comments])
            comment2 = kwargs.get('comment2',
                                  [c.split('\n')[1] for c in self.comments])
            pos_au = kwargs.get('pos_au', self.pos_au)
            numbers = kwargs.get('numbers', self.numbers)
            cell_au = kwargs.get('cell_au', self.cell_au)
            origin_au = kwargs.get('origin_au', self.origin_au)
            data = getattr(self, attr)
            cubeWriter(fn1,
                       comment1[0],
                       comment2[0],
                       numbers,
                       pos_au,
                       cell_au,
                       data[0],
                       origin=origin_au)
            cubeWriter(fn2,
                       comment1[1],
                       comment2[1],
                       numbers,
                       pos_au,
                       cell_au,
                       data[1],
                       origin=origin_au)
            cubeWriter(fn3,
                       comment1[2],
                       comment2[2],
                       numbers,
                       pos_au,
                       cell_au,
                       data[2],
                       origin=origin_au)
        else:
            raise Exception('Unknown format (Not implemented).')
