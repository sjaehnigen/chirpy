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

import copy as _copy
import numpy as _np
import warnings as _warnings
import itertools as _itertools

from .core import _CORE
from ..snippets import extract_keys as _extract_keys
from ..read.modes import xvibsReader
from ..read.coordinates import xyzReader, cpmdReader, pdbReader
from ..read.coordinates import xyzIterator as _xyzIterator
from ..read.coordinates import cpmdIterator as _cpmdIterator
from ..read.coordinates import pdbIterator as _pdbIterator
from ..write.coordinates import cpmdWriter, xyzWriter, pdbWriter

from ..topology.mapping import align_atoms as _align_atoms
from ..topology.mapping import dec as _dec
from ..topology.mapping import dist_crit_aa as _dist_crit_aa
from ..topology.mapping import wrap as _wrap
from ..topology.mapping import cowt as _cowt
from ..topology.mapping import join_molecules as _join_molecules
from ..topology.mapping import distance_matrix as _distance_matrix
from ..topology.mapping import distance_pbc as _distance_pbc

from ..physics import constants
from ..physics.statistical_mechanics import kinetic_energies as \
        _kinetic_energies

from ..mathematics import algebra as _algebra

# NB: data is accessed from behind (axis_pointer):
#   frame is (N,X)
#   trajectory is (F,N,X),
#   list of modes is (M,F,N,X)


class _FRAME(_CORE):
    def _labels(self):
        self._type = 'frame'
        self._labels = ('symbols',  '')

    def __init__(self, *args, **kwargs):
        self._labels()
        self._read_input(*args, **kwargs)
        self._sync_class()

    def _read_input(self,  *args, **kwargs):
        self.comments = kwargs.get('comments', [])
        self.symbols = kwargs.get('symbols', ())
        self.data = kwargs.get('data', _np.zeros((0, 0)))

    def _sync_class(self):
        self.axis_pointer = -2

        self.n_atoms, self.n_fields = self.data.shape
        if self.n_atoms != len(self.symbols):
            raise ValueError('Data shape inconsistent '
                             'with symbols attribute!\n')

    def __add__(self, other):
        new = _copy.deepcopy(self)
        new.data = _np.concatenate((self.data, other.data),
                                   axis=self.axis_pointer)
        _l = new._labels[self.axis_pointer]
        setattr(new, _l, getattr(self, _l) + getattr(other, _l))
        new._sync_class()
        return new

    def tail(self, n, **kwargs):
        axis = kwargs.get("axis", self.axis_pointer)
        new = _copy.deepcopy(self)
        new.data = self.data.swapaxes(axis, 0)[-n:].swapaxes(0, axis)
        try:
            _l = new._labels[axis]
            setattr(new, _l, getattr(self, _l)[-n:])
        except (KeyError, AttributeError):
            pass
        new._sync_class()
        return new

    def sort(self, *args, **kwargs):
        _symbols = _np.array(self.symbols)

        def get_slist():
            elem = {s: _np.where(_symbols == s)[0]
                    for s in _np.unique(_symbols)}
            return [i for k in sorted(elem) for i in elem[k]]

        if len(args) == 1:
            _slist = list(args[0])
        else:
            _slist = kwargs.get('order', get_slist())

        self.data = self.data.swapaxes(0, -2)[_slist].swapaxes(0, -2)
        self.symbols = tuple(_symbols[_slist])
        self._sync_class()

        return _slist

    def _is_similar(self, other):
        ie = list(map(lambda a: getattr(self, a) == getattr(other, a),
                      ['_type', 'n_atoms', 'n_fields']))
        ie.append(bool(_np.prod([a == b
                                for a, b in zip(_np.sort(self.symbols),
                                                _np.sort(other.symbols))])))
        return _np.prod(ie), ie

    def split(self, mask, select=None, join_molecules=True):
        '''select ... list or tuple of ids'''
        _data = [_np.moveaxis(_d, 0, -2)
                 for _d in _dec(_np.moveaxis(self.data, -2, 0), mask)]
        _symbols = _dec(self.symbols, mask)

        def create_obj(_d, _s):
            nargs = {}
            nargs.update(self.__dict__)
            nargs.update({'data': _d, 'symbols': _s})
            _obj = self._from_data(**nargs)
            if join_molecules:
                _obj.wrap_molecules(_np.ones_like(_s).astype(int))
            return _obj

        if select is None:
            _new = []
            for _d, _s in zip(_data, _symbols):
                _new.append(create_obj(_d, _s))
            return _new
        else:
            if isinstance(select, list):
                _new = create_obj(_data[select[0]], _symbols[select[0]])
                for _id in select[1:]:
                    _new += create_obj(_data[_id], _symbols[_id])
            else:
                _new = create_obj(_data[select], _symbols[select])
            self.__dict__.update(_new.__dict__)
            self._sync_class()

    @staticmethod
    def map_frame(obj1, obj2, **kwargs):
        '''obj1, obj2 ... Frame objects.
           Returns indices that would sort obj2 to match obj1.
           '''
        ie, tmp = obj1._is_similar(obj2)
        if not ie:
            raise TypeError('''The two Molecule objects are not similar!
                     n_atoms: %s
                     n_fields: %s
                     symbols: %s
                  ''' % tuple(tmp))

        if obj1._type != 'frame':
            raise NotImplementedError('map supports only FRAME objects!')

        com1 = _cowt(obj1.data, obj1.masses_amu, axis=-2)
        com2 = _cowt(obj2.data, obj2.masses_amu, axis=-2)

        assign = _np.zeros((obj1.n_atoms,)).astype(int)
        for s in _np.unique(obj1.symbols):
            i1 = _np.array(obj1.symbols) == s
            i2 = _np.array(obj2.symbols) == s
            ass = _np.argmin(_distance_matrix(
                                obj1.data[i1, :3] - com1[None],
                                obj2.data[i2, :3] - com2[None],
                                **kwargs
                                ),
                             axis=0)
            assign[i1] = _np.arange(obj2.n_atoms)[i2][ass]

        if not len(_np.unique(assign)) == obj1.n_atoms:
            _warnings.warn('Ambiguities encountered when mapping frames!')

        return assign

    @classmethod
    def _from_data(cls, **kwargs):
        return cls(**kwargs)


class _TRAJECTORY(_FRAME):
    def _labels(self):
        self._type = 'trajectory'
        self._labels = ('comments', 'symbols', '')

    def _read_input(self,  *args, **kwargs):
        self.comments = kwargs.get('comments', [])
        self.symbols = kwargs.get('symbols', ())
        self.data = kwargs.get('data', _np.zeros((0, 0, 0)))

    def _sync_class(self):
        self.n_frames, self.n_atoms, self.n_fields = self.data.shape
        if self.n_atoms != len(self.symbols):
            raise ValueError('Data shape inconsistent with '
                             'symbols attribute!\n')
        if self.n_frames != len(self.comments):
            raise ValueError('Data shape inconsistent with '
                             'comments attribute!\n')
        self.axis_pointer = -2


class _MODES(_FRAME):
    def _labels(self):
        self._type = 'modes'
        self._labels = ('comments', 'symbols', '')

    def _sync_class(self):
        self.n_modes, self.n_atoms, self.n_fields = self.data.shape
        if self.n_atoms != len(self.symbols):
            raise ValueError('Data shape inconsistent with '
                             'symbols attribute!\n')
        if self.n_modes != len(self.comments):
            raise ValueError('Data shape inconsistent with '
                             'comments attribute!\n')
        self.axis_pointer = -2
        self.modes = self.data[:, :, 6:9]
        self.eival_cgs = _np.array(self.comments).astype(float)

    def _modelist(self, modelist):
        if not isinstance(modelist, list):
            raise TypeError('Please give a list of integers instead of %s!'
                            % modelist.__class__.__name__)
        self.data = self.data[modelist]
        self.comments = self.comments[modelist]
        self._sync_class()

    def _modes(self):
        '''see _vel_au'''
        pass

    def _check_orthonormality(self):
        self.eivec = self.modes * _np.sqrt(self.masses_amu)[None, :, None]
        norm = _np.linalg.norm(self.eivec, axis=(1, 2))

        # --- ToDo: Work Around for trans + rot
        norm[:6] = 1.0
        # ---
        self.eivec /= norm[:, None, None]

        atol = 5.E-5
        com_motion = _np.linalg.norm(_cowt(self.modes,
                                           self.masses_amu,
                                           axis=1),
                                     axis=-1)

        if _np.amax(com_motion) > atol:
            _warnings.warn('Significant motion of COM for certain modes!',
                           RuntimeWarning)

        test = self.modes.reshape(self.n_modes, self.n_atoms*3)
        a = _np.inner(test, test)
        if any([_np.allclose(a,
                             _np.identity(self.n_modes),
                             atol=atol),
                _np.allclose(a[6:, 6:],
                             _np.identity(self.n_modes-6),
                             atol=atol)]):
            raise ValueError('The given cartesian displacements are '
                             'orthonormal! Please try enabling/disabling '
                             'the -mw flag!')
        test = self.eivec.reshape(self.n_modes, self.n_atoms*3)
        a = _np.inner(test, test)
        if not any([_np.allclose(a,
                                 _np.identity(self.n_modes),
                                 atol=atol),
                    _np.allclose(a[6:, 6:],
                                 _np.identity(self.n_modes-6),
                                 atol=atol)
                    ]):
            print(a)
            print(_np.amax(_np.abs(a-_np.identity(self.n_modes))))
            raise ValueError('The eigenvectors are not orthonormal!')


class _XYZ():
    '''Convention (at the moment) of data attribute:
       col 1-3: pos in aa; col 4-6: vel in au.
       pos_aa/vel_au attributes are protected, use underscored
       attributes for changes.'''

    def _read_input(self, *args, **kwargs):
        align_coords = kwargs.get('align_coords')
        center_coords = kwargs.get('center_coords')
        wrap = kwargs.get('wrap', False)
        wrap_molecules = kwargs.get('wrap_molecules', False)
        self.cell_aa_deg = kwargs.get('cell_aa_deg')

        if len(args) > 1:
            raise TypeError("File reader of %s takes at most 1 argument!"
                            % self.__class__.__name__)

        elif len(args) == 1:
            fn = args[0]
            fmt = kwargs.get('fmt', fn.split('.')[-1])
            self._fmt = fmt
            self._fn = fn
            if self._type == 'frame':
                _fr = kwargs.get('frame', 0)
                _fr = _fr, 1, _fr+1
                kwargs.update({'range': _fr})
            elif self._type == 'trajectory':
                _fr = kwargs.get('frame_range', (0, 1, float('inf')))
            self.fn = fn

            if fmt == "xyz":
                data, symbols, comments = xyzReader(fn,
                                                    **_extract_keys(kwargs,
                                                                    range=_fr,
                                                                    )
                                                    )

            elif fmt == "pdb":
                data, types, symbols, residues, cell_aa_deg, title = \
                        pdbReader(fn)
                n_atoms = len(symbols)
                comments = kwargs.get('comments', ['pdb'] * data.shape[0])

                if cell_aa_deg is not None:
                    if self.cell_aa_deg is None:
                        self.cell_aa_deg = cell_aa_deg

                    elif not _np.allclose(self.cell_aa_deg, cell_aa_deg):
                        _warnings.warn('The given cell parametres are '
                                       'different from those of the '
                                       'PDB file! '
                                       'Ignoring the latter.',
                                       RuntimeWarning)

            elif fmt == "xvibs":
                n_atoms, numbers, pos_aa, \
                    n_modes, omega_cgs, modes = xvibsReader(fn, **kwargs)
                comments = _np.array(omega_cgs).astype(str)
                symbols = constants.numbers_to_symbols(numbers)
                data = _np.concatenate((
                           _np.tile(pos_aa, (n_modes, 1, 1)),
                           _np.tile(_np.zeros_like(pos_aa), (n_modes, 1, 1)),
                           modes
                          ),
                          axis=-1
                          )

            elif fmt == "cpmd":
                # NB: CPMD writes XYZ files with vel_aa
                if ('symbols' in kwargs or 'numbers' in kwargs):
                    numbers = kwargs.get('numbers')
                    symbols = kwargs.get('symbols')
                    if symbols is None:
                        symbols = constants.numbers_to_symbols(numbers)
                else:
                    raise TypeError("cpmdReader needs list of numbers or "
                                    "symbols.")

                data = _np.array(cpmdReader(fn,
                                            **_extract_keys(kwargs,
                                                            kinds=symbols,
                                                            filetype=fn,
                                                            range=_fr
                                                            )
                                            ))
                comments = kwargs.get('comments', ['cpmd'] * data.shape[0])
                data[:, :, :3] *= constants.l_au2aa

            else:
                raise ValueError('Unknown format: %s.' % fmt)

        elif len(args) == 0:
            if 'data' in kwargs and ('symbols' in kwargs
                                     or 'numbers' in kwargs):
                self.fn = ''
                numbers = kwargs.get('numbers')
                symbols = kwargs.get('symbols')
                if symbols is None:
                    symbols = [constants.symbols[z - 1] for z in numbers]
                data = kwargs.get('data')
                _sh = data.shape
                if len(_sh) == 2:
                    data = data.reshape((1, ) + _sh)
                comments = _np.array([kwargs.get('comments',
                                                 data.shape[0] * ['passed'])
                                      ]).flatten()
            else:
                raise TypeError('%s needs fn or data + symbols argument!' %
                                self.__class__.__name__)

        if self.cell_aa_deg is None:
            self.cell_aa_deg = _np.array([0.0, 0.0, 0.0, 90., 90., 90.])

        if self._type == 'frame':
            _f = kwargs.get("frame", 0)
            data = data[_f]
            comments = [comments[_f]]

        if self._type == 'modes':
            if 'omega_cgs' not in locals():
                raise TypeError('Could not retrieve modes data from input!')
            self.eival_cgs = omega_cgs
            comments = _np.array(omega_cgs).astype(str)

        self.symbols = tuple(symbols)
        self.comments = list(comments)
        self.data = data
        self._sync_class()

        if wrap:
            self.wrap_atoms()
        if wrap_molecules:
            try:
                self.wrap_molecules(kwargs['mol_map'])
            except KeyError:
                _warnings.warn('Could not find molecular map for wrapping!')
                self.wrap_atoms()

        self._sync_class()

    def _pos_aa(self, *args):
        if len(args) == 0:
            self.pos_aa = self.data.swapaxes(0, -1)[:3].swapaxes(0, -1)
        elif len(args) == 1:
            if args[0].shape != self.pos_aa.shape:
                raise ValueError(
                     'Cannot update attribute with values of different shape!')
            _tmp = self.data.swapaxes(0, -1)
            _tmp[:3] = args[0].swapaxes(0, -1)
            self.data = _tmp.swapaxes(0, -1)
            self._pos_aa()
        else:
            raise TypeError('Too many arguments for %s!'
                            % self._pos_aa.__name__)

    def _vel_au(self, *args):
        if len(args) == 0:
            self.vel_au = self.data.swapaxes(0, -1)[3:6].swapaxes(0, -1)
        elif len(args) == 1:
            if args[0].shape != self.vel_au.shape:
                raise ValueError(
                     'Cannot update attribute with values of different shape!')
            _tmp = self.data.swapaxes(0, -1)
            _tmp[3:6] = args[0].swapaxes(0, -1)
            self.data = _tmp.swapaxes(0, -1)
            self._vel_au()
        else:
            raise TypeError('Too many arguments for %s!'
                            % self._vel_au.__name__)

    def _sync_class(self):
        try:
            self.masses_amu = constants.symbols_to_masses(self.symbols)
        except KeyError:
            _warnings.warn('Could not find masses for all elements! '
                           'Centre of mass cannot be used.',
                           RuntimeWarning)
        self._pos_aa()
        self._vel_au()
        if self.vel_au.size == 0:
            self.vel_au = _np.zeros_like(self.pos_aa)
        self.cell_aa_deg = _np.array(self.cell_aa_deg)
        if not isinstance(self.comments[0], str):
            raise AttributeError('Missing comments line! Contact support!')

    def _is_equal(self, other, atol=1e-08, noh=True):
        '''atol adds up to dist_crit_aa from vdw radii'''
        _p, ie = self._is_similar(other)

        def f(a):
            if self._type == 'trajectory':
                raise TypeError('Trajectories cannot be tested for equality '
                                '(only similarity)!')
            _o_pos = _copy.deepcopy(getattr(other, a).reshape(
                                        (1, ) + other.data.shape
                                        ))
            _s_pos = getattr(self, a)

            _o_pos = _align_atoms(_o_pos, _np.array(self.masses_amu),
                                  ref=_s_pos)[0]
            _bool = []
            for _s in set(self.symbols):
                _ind = _np.array(self.symbols) == _s
                if _s != 'H' or not noh:
                    a = _distance_pbc(_s_pos[_ind],
                                      _o_pos[_ind],
                                      cell_aa_deg=self.cell_aa_deg)
                    _bool.append(_np.amax(a) <= _dist_crit_aa([_s])[0] + atol)

            return bool(_np.prod(_bool))

        if _p == 1:
            ie += list(map(f, ['data']))

        self._sync_class()
        return _np.prod(ie), ie

    # --- join the next two methods?
    def wrap_atoms(self, **kwargs):
        if self._type == 'frame':
            self._pos_aa(_wrap(self.pos_aa.reshape(1, self.n_atoms, 3),
                               self.cell_aa_deg)[0])
        else:
            self._pos_aa(_wrap(self.pos_aa, self.cell_aa_deg))

    def wrap_molecules(self, mol_map, **kwargs):
        mode = kwargs.get('mode', 'cog')
        w = _np.ones((self.n_atoms))
        if mode == 'com':
            w = self.masses_amu

        if self._type == 'frame':
            _p, mol_c_aa = _join_molecules(
                                self.pos_aa.reshape(1, self.n_atoms, 3),
                                mol_map,
                                self.cell_aa_deg,
                                weights=w,
                                )
            self._pos_aa(_p[0])
            del _p
        else:
            _p, mol_c_aa = _join_molecules(
                                self.pos_aa,
                                mol_map,
                                self.cell_aa_deg,
                                weights=w,
                                )
            self._pos_aa(_p)
        return mol_c_aa

    def align_coordinates(self, align_coords, **kwargs):
        if not isinstance(align_coords, list):
            if isinstance(align_coords, bool) and align_coords:
                align_coords = slice(None)
            else:
                raise TypeError('Expecting a bool or a list of atoms '
                                'for alignment!')

        self._aligned_coords = align_coords
        wt = _np.ones((self.n_atoms))
        if kwargs.get('use_com', False):
            wt = self.masses_amu

        if self._type == 'trajectory':
            self._align_ref = kwargs.get('align_ref', self.pos_aa[0])
        else:
            self._align_ref = kwargs.get('align_ref', self.pos_aa)

        self.align(weights=wt,
                   ref=self._align_ref,
                   subset=align_coords,
                   )

        if hasattr(self, '_centered_coords') and kwargs.get(
                'force_centre', False):
            _ref = _cowt(self.pos_aa,
                         wt,
                         subset=self._centered_coords,
                         axis=self.axis_pointer)
            self.center_position(_ref, self.cell_aa_deg)

    def align(self, **kwargs):
        _wt = kwargs.get('weights', _np.ones(self.n_atoms))

        if self._type == 'trajectory':
            self._pos_aa(_align_atoms(
                            self.pos_aa,
                            _wt,
                            **kwargs
                            )
                         )
        elif self._type == 'frame':
            self._pos_aa(_align_atoms(
                            self.pos_aa.reshape((1, ) + self.data.shape),
                            _wt,
                            **kwargs
                            )[0]
                         )

    def center_coordinates(self, center_coords, **kwargs):
        if not isinstance(center_coords, list):
            if isinstance(center_coords, bool) and center_coords:
                center_coords = slice(None)
            else:
                raise TypeError('Expecting a bool or a list of atoms '
                                'for centering!')
        self._centered_coords = center_coords
        wt = _np.ones((self.n_atoms))
        if kwargs.get('use_com', False):
            wt = self.masses_amu

        _ref = _cowt(self.pos_aa,
                     wt,
                     subset=center_coords,
                     axis=self.axis_pointer)

        self.center_position(_ref, self.cell_aa_deg)

    def center_position(self, pos, cell_aa_deg, **kwargs):
        '''pos reference in shape (n_frames, three)'''
        if self._type == 'frame':
            self._pos_aa(self.pos_aa + cell_aa_deg[None, :3] / 2
                         - pos[None, :])
        else:
            self._pos_aa(self.pos_aa + cell_aa_deg[None, None, :3] / 2
                         - pos[:, None, :])

        if kwargs.get("wrap", True):
            self.wrap_atoms()

    def align_to_vector(self, i0, i1, vec, **kwargs):
        '''
        Align a reference line pos[i1]-pos[i0] to vec (no pbc support)
        Center of rotation is  pos[i0]. '''

        if self._type == 'frame':
            _ref = self.pos_aa - self.pos_aa[i0, None]
            _R = _algebra.rotation_matrix(_ref[i1], vec)
            _pos = _np.tensordot(_R,
                                 _ref,
                                 axes=([1], [1])) + self.pos_aa[i0, None]
            _vel = _np.tensordot(_R,
                                 self.vel_au,
                                 axes=([1], [1])) * constants.l_aa2au

        else:
            _pos = []
            _vel = []
            for _p, _v in zip(
                    self.pos_aa - self.pos_aa[:, i0, None],
                    self.vel_au
                    ):
                _R = _algebra.rotation_matrix(_p[i1], vec)
                _pos.append(_np.tensordot(_R,
                                          _p,
                                          axes=([1], [1])).swapaxes(0, 1))
                _vel.append(_np.tensordot(_R,
                                          _v,
                                          axes=([1], [1])).swapaxes(0, 1))
            _pos = _np.array(_pos) + self.pos_aa[:, i0, None]
            _vel = _np.array(_vel)

        self._pos_aa(_pos)
        self._vel_au(_vel)

    def write(self, fn, **kwargs):
        attr = kwargs.get('attr', 'data')
        factor = kwargs.get('factor', 1.0)  # for velocities
        separate_files = kwargs.get('separate_files', False)

        loc_self = _copy.deepcopy(self)
        if self._type == "frame":
            loc_self.data = loc_self.data.reshape((1,
                                                   self.n_atoms,
                                                   self.n_fields))
            loc_self.n_frames = 1
            _XYZ._sync_class(loc_self)

        fmt = kwargs.get('fmt', fn.split('.')[-1])

        if fmt == "xyz":

            if separate_files:
                frame_list = kwargs.get('frames', range(loc_self.n_frames))
                [xyzWriter(''.join(fn.split('.')[:-1])
                           + '%03d' % fr
                           + '.'
                           + fn.split('.')[-1],
                           [getattr(loc_self, attr)[fr]],
                           loc_self.symbols,
                           [loc_self.comments[fr]],
                           **_extract_keys(kwargs, append=False)
                           )
                 for fr in frame_list]

            else:
                xyzWriter(fn,
                          getattr(loc_self, attr),
                          loc_self.symbols,
                          getattr(loc_self, 'comments',
                                            loc_self.n_frames * ['passed']),
                          **_extract_keys(kwargs, append=False)
                          )

        elif fmt == "pdb":
            mol_map = kwargs.get('mol_map')
            if mol_map is None:
                raise AttributeError('Could not find mol_map for PDB output!')
            cell_aa_deg = kwargs.get('cell_aa_deg')
            if cell_aa_deg is None:
                _warnings.warn("Missing cell parametres for PDB output!",
                               RuntimeWarning)
                cell_aa_deg = _np.array([0.0, 0.0, 0.0, 90., 90., 90.])
            pdbWriter(fn,
                      loc_self.pos_aa[0],  # only frame 0, vels are not written
                      types=loc_self.symbols,  # types not supported
                      symbols=loc_self.symbols,
                      residues=_np.vstack((
                                    _np.array(mol_map) + 1,
                                    _np.array(['MOL'] * loc_self.n_atoms)
                                    )).swapaxes(0, 1),
                      box=cell_aa_deg,
                      title='Generated from %s with Molecule Class' % self.fn
                      )

        elif fmt == 'cpmd':
            if kwargs.get('sort_atoms', True):
                print('CPMD WARNING: Output with sorted atomlist!')
                loc_self._sort()
            cpmdWriter(fn,
                       loc_self.pos_aa * constants.l_aa2au,
                       loc_self.symbols,
                       loc_self.vel_au * factor,
                       **kwargs)  # DEFAULTS pp='MT_BLYP', bs=''

        else:
            raise ValueError('Unknown format: %s.' % fmt)

    def get_atom_spread(self):
        '''pos_aa: _np.array of shape (n_frames, n_atoms, 3)'''
        dim_qm = _np.zeros((3))
        for i in range(3):
            imin = _np.min(_np.moveaxis(self.pos_aa, -1, 0)[i])
            imax = _np.max(_np.moveaxis(self.pos_aa, -1, 0)[i])
            dim_qm[i] = imax - imin
        print('Spread of QM atoms:       %s %s %s'
              % tuple([round(dim, 4) for dim in dim_qm]))

        if self._type == 'frame':
            amax = _np.amax(_distance_matrix(
                                    self.data[:, :3],
                                    cell_aa_deg=self.cell_aa_deg
                                    ))
        print('Max distance:             %s' % round(amax, 4))


class XYZFrame(_XYZ, _FRAME):
    def _sync_class(self):
        _FRAME._sync_class(self)
        _XYZ._sync_class(self)

    def make_trajectory(self, n_images=3, ts_fs=1, **kwargs):
        '''Create a XYZTrajectory object with <n_images>
           frames from velocities and a timestep ts.'''

        _img = _np.arange(-(n_images // 2), n_images // 2 + 1)
        _pos_aa = _np.tile(self.pos_aa, (n_images, 1, 1))
        _vel_aa = _np.tile(self.vel_au * constants.t_fs2au * constants.l_au2aa,
                           (n_images, 1, 1))
        _pos_aa += _vel_aa * _img[:, None, None] * ts_fs

        return XYZTrajectory(data=_np.dstack((_pos_aa, _vel_aa)),
                             symbols=self.symbols,
                             comments=[self.comments[0] + ' im ' + str(m)
                                       for m in _img]
                             )


class XYZIterator(_XYZ, _FRAME):
    '''A generator of XYZ frames (BETA).'''
    def __init__(self, *args, **kwargs):
        if len(args) != 1:
            raise TypeError("File reader of %s takes exactly 1 argument!"
                            % self.__class__.__name__)
        else:
            fn = args[0]
            self._fn = fn
            self._fmt = kwargs.get('fmt', fn.split('.')[-1])

            if self._fmt == "xyz":
                self._topology = XYZFrame(fn, **kwargs)
                self._gen = _xyzIterator(fn,
                                         **kwargs
                                         )

            elif self._fmt == "pdb":
                self._topology = XYZFrame(fn, **kwargs)
                self._gen = _pdbIterator(fn,
                                         # **kwargs
                                         )
            elif self._fmt == "cpmd":
                if ('symbols' in kwargs or 'numbers' in kwargs):
                    numbers = kwargs.get('numbers')
                    symbols = kwargs.get('symbols',
                                         [constants.symbols[z - 1]
                                          for z in numbers]
                                         )
                    kwargs.update({'symbols': symbols})
                else:
                    raise TypeError("cpmdReader needs list of numbers or "
                                    "symbols.")
                # comments = kwargs.get('comments', [''])

                self._topology = XYZFrame(fn,
                                          **kwargs
                                          )

                self._gen = _cpmdIterator(fn,
                                          **kwargs
                                          )

            else:
                raise ValueError('Unknown format: %s.' % self._fmt)

            self._kwargs = {}
            # initialise list of masks
            self._kwargs['_masks'] = []
            # keep kwargs for iterations
            self._kwargs.update(kwargs)

            self._kwargs['range'] = kwargs.get('range', (0, 1, float('inf')))
            self._fr, self._st, buf = self._kwargs['range']

            # --- Load first frame w/o consuming it
            self._fr -= self._st
            next(self)
            self._chaste = True

    def __iter__(self):
        return self

    def __next__(self):
        if hasattr(self, '_chaste'):
            # repeat first step of next() after __init__
            if self._chaste:
                self._chaste = False
                return self._fr

        frame = next(self._gen)

        if self._fmt == 'xyz':
            out = {
                    'data': frame[0],
                    'symbols': frame[1],
                    'comments': frame[2],
                    }

        if self._fmt == 'pdb':
            out = {
                    'data': frame[0],
                    'symbols': frame[2],
                    'comments': [frame[-1]],
                    'cell_aa_deg': frame[-2],
                    # 'res':
                    }

        if self._fmt == 'cpmd':
            frame[:, :3] *= constants.l_au2aa
            out = {
                    'data': frame,
                    'symbols': self._symbols,
                    'comments': self._comments,
                    }
        self._fr += self._st
        self._kwargs.update(out)

        self._frame = XYZFrame(**self._kwargs)

        # --- check for stored masks
        for _f, _f_args, _f_kwargs in self._kwargs['_masks']:
            getattr(self._frame, _f)(*_f_args, **_f_kwargs)

        self.__dict__.update(self._frame.__dict__)

        return self._fr

    @classmethod
    def _from_list(cls, LIST, **kwargs):
        a = cls(LIST[0], **kwargs)
        for _f in LIST[1:]:
            b = cls(_f, **kwargs)
            a += b
        return a

    def __add__(self, other):
        new = self
        if new._topology._is_similar(other._topology)[0] == 1:
            new._gen = _itertools.chain(new._gen, other._gen)
            return new
        else:
            raise ValueError('Cannot combine frames of different size!')

    def _to_trajectory(self):
        '''Perform iteration on remaining iterator
           (may take some time)'''
        if self._fmt == 'xyz':
            data, symbols, comments = zip(*self._gen)
            out = {
                    'data': _np.array(data),
                    'symbols': symbols[0],
                    'comments': list(comments)
                    }

        if self._fmt == 'cpmd':
            data = _np.array(tuple(self._gen))
            data[:, :, :3] *= constants.l_au2aa
            out = {
                    'data': data,
                    'symbols': self._symbols,
                    'comments': [self._comments] * data.shape[0],
                    }

        self._kwargs.update(out)
        return XYZTrajectory(**self._kwargs)

    def rewind(self):
        '''Reinitialises the Interator (BETA)'''
        self._chaste = False
        self.__init__(self._fn, **self._kwargs)

    @staticmethod
    def _loop(obj, func, events, *args, **kwargs):
        '''Unwinds the Iterator until it is exhausted constantly
           executing the given frame-owned function and passing
           through given arguments.
           Events are dictionaries with (relative) frames
           as keys and some action as argument that are only
           executed when the Iterator reaches the value of the
           key.'''
        _fr = 0
        for _ifr in obj:
            if isinstance(func, str):
                getattr(obj._frame, func)(*args, **kwargs)
            elif callable(func):
                func(obj, *args, **kwargs)
            if _fr in events:
                if isinstance(events[_fr], dict):
                    kwargs.update(events[_fr])
            _fr += 1

    def write(self, fn, **kwargs):
        self._loop(self,
                   'write',
                   {0: {'append': True}},
                   fn,
                   **kwargs
                   )
        self.rewind()

    def mask_duplicate_frames(self, verbose=True, **kwargs):
        def split_comment(comment):
            # --- ToDo: Could be a staticmethod of XYZ
            if 'i = ' in comment:
                return int(comment.split()[2].rstrip(','))
            elif 'Iteration:' in comment:
                return int(comment.split('_')[1].rstrip())
            else:
                raise ValueError('Cannot get frame info from comments!')

        def _func(obj, **kwargs):
            _skip = obj._kwargs.get('skip', [])
            _timesteps = obj._kwargs.get('_timesteps', [])
            _ts = split_comment(obj._frame.comments[0])
            if _ts not in _timesteps:
                _timesteps.append(_ts)
            else:
                _skip.append(obj._fr)
            obj._kwargs.update({'_timesteps': _timesteps})
            obj._kwargs.update({'skip': _skip})

        self._kwargs['_timesteps'] = []
        self._loop(self, _func, {}, **kwargs)
        if verbose:
            print('Duplicate frames in %s according to given range %s:' % (
                self._fn,
                self._kwargs['range']
                ), self._kwargs['skip'])
        self.rewind()

    @staticmethod
    def _mask(obj, func, *args, **kwargs):
        '''Adds a frame-owned function that is called with every __next__()
           before returning.'''
        obj._kwargs['_masks'].append(
                (func, args, kwargs),
                )
        if len(obj._kwargs['_masks']) > 10:
            _warnings.warn('Too many masks on iterator!')

    # These masks all follow the same logic (could be generalised, but python
    # does not support call of function name from within that function)
    def sort(self, *args, **kwargs):
        self._frame.sort(*args, **kwargs)
        self.__dict__.update(self._frame.__dict__)
        self._mask(self, 'sort', *args, **kwargs)

    def align_coordinates(self, *args, **kwargs):
        self._frame.align_coordinates(*args, **kwargs)
        self.__dict__.update(self._frame.__dict__)
        # remember reference
        kwargs.update({'align_ref': self._frame._align_ref})
        self._mask(self, 'align_coordinates', *args, **kwargs)

    def center_coordinates(self, *args, **kwargs):
        self._frame.center_coordinates(*args, **kwargs)
        self.__dict__.update(self._frame.__dict__)
        self._mask(self, 'center_coordinates', *args, **kwargs)

    def center_position(self, *args, **kwargs):
        self._frame.center_position(*args, **kwargs)
        self.__dict__.update(self._frame.__dict__)
        self._mask(self, 'center_position', *args, **kwargs)

    def wrap_molecules(self, *args, **kwargs):
        self._frame.wrap_molecules(*args, **kwargs)
        self.__dict__.update(self._frame.__dict__)
        self._mask(self, 'wrap_molecules', *args, **kwargs)

    def split(self, *args, **kwargs):
        self._frame.split(*args, **kwargs)
        self.__dict__.update(self._frame.__dict__)
        self._mask(self, 'split', *args, **kwargs)


class XYZTrajectory(_XYZ, _TRAJECTORY):
    def _sync_class(self):
        _TRAJECTORY._sync_class(self)
        _XYZ._sync_class(self)

    def calculate_nuclear_velocities(self, **kwargs):
        '''finite diff, linear (frame1-frame0, frame2-frame1, etc.)'''
        _warnings.warn("Using outdated method of XYZTrajectory. "
                       "Proceed with care!")
        ts = kwargs.get('ts', 0.5)

        if _np.linalg.norm(self.vel_au) != 0:
            _warnings.warn('Overwriting existing velocities in file %s'
                           % self.fn,
                           RuntimeWarning)
        self.vel_au[:-1] = _np.diff(self.pos_aa,
                                    axis=0) / (ts * constants.v_au2aaperfs)


class VibrationalModes(_XYZ, _MODES):
    def _sync_class(self):
        _MODES._sync_class(self)
        _XYZ._sync_class(self)
        _MODES._check_orthonormality(self)

    def calculate_nuclear_velocities(self, **kwargs):
        '''Occupation can be single, average, or random.'''
        occupation = kwargs.get('occupation', 'single')
        temperature = kwargs.get('temperature', 300)
        beta_au = 1./(temperature*constants.k_B_au)
        print('Calculating velocities for {} K.'.format(temperature))

        S = self.eivec /\
            _np.sqrt(beta_au) /\
            _np.sqrt(constants.m_amu_au) /\
            _np.sqrt(self.masses_amu)[None, :, None]

        if occupation == 'single':
            _VEL = S
            e_kin_au = _kinetic_energies(_VEL, self.masses_amu)
            scale = temperature / (_np.sum(e_kin_au) / constants.k_B_au /
                                   self.n_modes) / 2
            print(scale)
            _VEL *= scale

        elif occupation == 'average':
            _VEL = S.sum(axis=0)
            # atomic_ekin_au = traj_utils.CalculateKineticEnergies(avg,
            # _masses_amu)
            # scale = temperature/(_np.sum(atomic_ekin_au)/
            # constants.k_B_au/n_modes)/2
            # print(scale)
            # avg *= _np.sqrt(scale)
        elif occupation == 'random':  # NOT TESTED!
            phases = _np.random.rand(self.n_modes)*_np.pi
            _VEL = (S*_np.cos(phases)[:, None, None]).sum(axis=0)

            # random pos
#            avg = _np.zeros((1, n_atoms, 3))
#            omega_au = 2*_np.pi*_np.array(self.eival_cgs)
#            *constants.c_cgs*constants.t_au
#            for i_mode in range(self.n_modes):
#                avg += S[i_mode].reshape(1, self.n_atoms, 3)
#                *_np.sin(phases[i_mode])/omega_au[i_mode]
#            avg *= constants.l_au2aa
#            avg += self.pos_au*constants.l_au2aa
            print('Random seed not tested')
        else:
            print('Occupation mode %s not understood!' % occupation)

        self._vel_au(_VEL)

    def get_mode(self, mode, **kwargs):
        '''Returns a XYZFrame of given mode'''

        return XYZFrame(data=self.data[mode],
                        symbols=self.symbols,
                        comments=[self.comments[mode]],
                        **kwargs
                        )


class NormalModes(VibrationalModes):
    pass
