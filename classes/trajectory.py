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

import sys as _sys
import copy as _copy
import numpy as _np
import warnings as _warnings
import itertools as _itertools

from ..snippets import extract_keys as _extract_keys
from ..readers.modes import xvibsReader
from ..readers.coordinates import xyzReader, cpmdReader, pdbReader
from ..readers.coordinates import xyzIterator as _xyzIterator
from ..readers.coordinates import cpmdIterator as _cpmdIterator
from ..writers.coordinates import cpmdWriter, xyzWriter, pdbWriter

from ..topology.mapping import align_atoms as _align_atoms
from ..topology.mapping import dec as _dec
from ..topology.symmetry import wrap as _wrap
from ..topology.symmetry import cowt as _cowt
from ..topology.symmetry import join_molecules as _join_molecules

from ..physics import constants
from ..physics.constants import masses_amu as _masses_amu
# from ..physics.classical_electrodynamics import current_dipole_moment, \
# magnetic_dipole_shift_origin
# from ..physics.modern_theory_of_magnetisation import calculate_mic
from ..physics.statistical_mechanics import calculate_kinetic_energies as \
        _calculate_kinetic_energies

from ..mathematics import algebra as _algebra

# ToDo: write() still old in _TRAJECORY, _FRAME does not have any write method
# new class: Moments()
# Note: the object format is extended from behind (axis_pointer):
#   frame is (N,X)
#   trajectory is (F,N,X),
#   list of modes is (M,F,N,X)
#       --> Access data structures from behind!
#   add iterators
# get rid of ugly solution with _labels (use iterator of _FRAMES?)


class _FRAME():
    if int(_np.version.version.split('.')[1]) < 14:
        print('ERROR: You have to use a numpy version >= 1.14.0! '
              'You are using %s.' % _np.version.version)
        _sys.exit(1)

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
        # access from behind !
        self.axis_pointer = -2
        # --- should be in __init__ but it is not inherited

        self.n_atoms, self.n_fields = self.data.shape
        if self.n_atoms != len(self.symbols):
            raise ValueError('ERROR: Data shape inconsistent '
                             'with symbols attribute!\n')

    def __add__(self, other):
        new = _copy.deepcopy(self)
        new.data = _np.concatenate((self.data, other.data),
                                   axis=self.axis_pointer)
        _l = new._labels[self.axis_pointer + 1]
        setattr(new, _l, getattr(self, _l) + getattr(other, _l))
        new._sync_class()
        return new

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        self = self.__add__(other)
        return self

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

    def _sort(self):
        elem = {s: _np.where(self.symbols == s)[0]
                for s in _np.unique(self.symbols)}
        ind = [i for k in sorted(elem) for i in elem[k]]
        self.data = self.data.swapaxes(0, -2)[ind].swapaxes(0, -2)
        self.symbols = self.symbols[ind]
        self._sync_class()
        return ind

    def _is_similar(self, other):
        # topology.map_atoms_by_coordinates
        ie = list(map(lambda a: getattr(self, a) == getattr(other, a),
                      ['n_atoms', 'n_fields']))
        ie.append(bool(_np.prod([a == b
                                for a, b in zip(_np.sort(self.symbols),
                                                _np.sort(other.symbols))])))
        # if hasattr(data, 'cell_aa')
        return _np.prod(ie), ie

    def _split(self, mask):
        _data = [_np.moveaxis(_d, 0, -2)
                 for _d in _dec(_np.moveaxis(self.data, -2, 0), mask)]
        _symbols = _dec(self.symbols, mask)

        return [self._from_data(data=_d, symbols=_s, comment=self.comments)
                for _d, _s in zip(_data, _symbols)]

    @classmethod
    def _from_data(cls, **kwargs):
        return cls(**kwargs)


class _TRAJECTORY(_FRAME):
    def _labels(self):
        self._type = 'trajectory'
        self._labels = ('comments', 'symbols')

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
        # access from behind !
        self.axis_pointer = -2
        # --- should be in __init__ but it is not inherited


class _XYZ():
    '''Convention (at the moment) of data attribute:
       col 1-3: pos in aa; col 4-6: vel in au.
       pos_aa/vel_au attributes are protected, use underscored
       attributes for changes.'''

    def _read_input(self, *args, **kwargs):
        align_coords = kwargs.get('align_coords')
        center_coords = kwargs.get('center_coords')
        wrap = kwargs.get('wrap', False)
        self.cell_aa_deg = kwargs.get('cell_aa_deg')

        if len(args) > 1:
            raise TypeError("File reader of %s takes at most 1 argument!"
                            % self.__class__.__name__)

        elif len(args) == 1:
            fn = args[0]
            fmt = kwargs.get('fmt', fn.split('.')[-1])
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
                comments = ["xvibs"]
                n_atoms, numbers, pos_aa, \
                    n_modes, omega_invcm, modes = xvibsReader(fn)
                symbols = [constants.symbols[z - 1] for z in numbers]
                data = pos_aa.reshape((1, n_atoms, 3))

            elif fmt == "cpmd":
                # NB: CPMD writes XYZ files with vel_aa
                if ('symbols' in kwargs or 'numbers' in kwargs):
                    numbers = kwargs.get('numbers')
                    symbols = kwargs.get('symbols', [constants.symbols[z - 1]
                                                     for z in numbers])
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

        elif len(args) == 0:  # shift it to classmethod _from_data()
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
                raise TypeError('XYZData needs fn or data + symbols argument!')

        if self.cell_aa_deg is None:
            self.cell_aa_deg = _np.array([0.0, 0.0, 0.0, 90., 90., 90.])

        if self._type == 'frame':  # is it a frame?
            _f = kwargs.get("frame", 0)
            data = data[_f]
            comments = [comments[_f]]

        self.symbols = tuple(symbols)
        self.comments = list(comments)
        self.data = data
        self._sync_class()

        if center_coords is not None and center_coords:
            if not isinstance(center_coords, list):
                if isinstance(center_coords, bool):
                    center_coords = slice(None)
                else:
                    raise TypeError('Expecting a bool or a list of atoms '
                                    'for centering!')

            wt = _np.ones((self.n_atoms))
            if kwargs.get('use_com', False):
                wt = self.masses_amu

            _ref = _cowt(self.pos_aa,
                         wt,
                         subset=center_coords,
                         axis=self.axis_pointer)

            self._center_position(_ref, self.cell_aa_deg)
            wrap = True

        if wrap:
            self._wrap_atoms(self.cell_aa_deg)

        # --- actions without allowed wrapping after this line

        if align_coords is not None and align_coords:
            if not isinstance(align_coords, list):
                if isinstance(align_coords, bool):
                    align_coords = slice(None)
                else:
                    raise TypeError('Expecting a bool or a list of atoms '
                                    'for alignment!')

            wt = _np.ones((self.n_atoms))
            if kwargs.get('use_com', False):
                wt = self.masses_amu

            if self._type == 'trajectory':
                self._align_ref = kwargs.get('align_ref', self.pos_aa[0])
            else:
                self._align_ref = kwargs.get('align_ref', self.pos_aa)

            self._align(weights=wt,
                        ref=self._align_ref,
                        subset=align_coords,
                        )

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
            self.vel_au = self.data.swapaxes(0, -1)[3:].swapaxes(0, -1)
        elif len(args) == 1:
            if args[0].shape != self.vel_au.shape:
                raise ValueError(
                     'Cannot update attribute with values of different shape!')
            _tmp = self.data.swapaxes(0, -1)
            _tmp[3:] = args[0].swapaxes(0, -1)
            self.data = _tmp.swapaxes(0, -1)
            self._vel_au()
        else:
            raise TypeError('Too many arguments for %s!'
                            % self._vel_au.__name__)

    def _sync_class(self):
        try:
            self.masses_amu = _np.array([_masses_amu[s]
                                         for s in self.symbols])
        except KeyError:
            _warnings.warn('Could not find masses for all elements! '
                           'Centre of mass cannot be used.',
                           RuntimeWarning)
        # These are NOT pointers and any changes to pos/vel will be overwritte
        # by data! You have to change data instead or use _pos/_vel
        self._pos_aa()
        self._vel_au()
        # Why using pos_aa/vel_au arguments AT ALL?
        if self.vel_au.size == 0:
            self.vel_au = _np.zeros_like(self.pos_aa)

    def _is_equal(self, other, atol=1e-08):
        _p, ie = self._is_similar(other)

        def f(a):
            if self._type == 'trajectory':
                raise TypeError('Trajectories cannot be tested for equality '
                                '(only similarity)!')
            _o_pos = getattr(other, a).reshape((1, ) + other.data.shape)
            _s_pos = getattr(self, a)

            _o_pos = _align_atoms(_o_pos, self.masses_amu, ref=_s_pos)[0]
            return _np.allclose(_s_pos, _np.mod(_o_pos, _s_pos), atol=atol)

        if _p == 1:
            ie += list(map(f, ['data']))

        return _np.prod(ie), ie

    # join the next two methods?
    def _wrap_atoms(self, cell_aa_deg, **kwargs):
        if self._type == 'frame':
            self._pos_aa(_wrap(self.pos_aa.reshape(1, self.n_atoms, 3),
                               cell_aa_deg)[0])
        else:
            self._pos_aa(_wrap(self.pos_aa, cell_aa_deg))

        # PDB needs it
        # abc, albega = _np.split(cell_aa_deg, 2)
        # setattr(self, 'abc', abc)
        # setattr(self, 'albega', albega)

    def _wrap_molecules(self, mol_map, cell_aa_deg, **kwargs):
        mode = kwargs.get('mode', 'cog')
        w = _np.ones((self.n_atoms))
        if mode == 'com':
            w = self.masses_amu

        if self._type == 'frame':
            _p, mol_c_aa = _join_molecules(
                                self.pos_aa.reshape(1, self.n_atoms, 3),
                                mol_map,
                                cell_aa_deg,
                                weights=w,
                                )
            self._pos_aa(_p[0])
            del _p
        else:
            _p, mol_c_aa = _join_molecules(
                                self.pos_aa,
                                mol_map,
                                cell_aa_deg,
                                weights=w,
                                )
            self._pos_aa(_p)
        return mol_c_aa
        # #print('UPDATE WARNING: inserted "swapaxes(0, 1)" for mol_cog_aa
        # attribute (new shape: (n_frames, n_mols, 3))!')
        # setattr(self, 'mol_'+mode+'_aa', _np.array(mol_c_aa).swapaxes(0, 1))
        # setattr(self, 'mol_map', mol_map)

        # PDB needs it ToDo
        # abc, albega = _np.split(cell_aa_deg, 2)
        # setattr(self, 'abc', abc)
        # setattr(self, 'albega', albega)

    def _align(self, **kwargs):
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

    def _center_position(self, pos, cell_aa_deg, **kwargs):
        '''pos reference in shape (n_frames, three)'''
        if self._type == 'frame':
            self._pos_aa(self.pos_aa + cell_aa_deg[None, :3] / 2
                         - pos[None, :])
        else:
            self._pos_aa(self.pos_aa + cell_aa_deg[None, None, :3] / 2
                         - pos[:, None, :])

    def _align_to_vector(self, i0, i1, vec, **kwargs):
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
        attr = kwargs.get('attr', 'data')  # only for xyz format
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
                      loc_self.pos_aa[0],  # only frame 0 vels are not written
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

    # Some info prints
    def get_atom_spread(self):
        '''pos_aa: _np.array of shape (n_frames, n_atoms, 3)'''
        dim_qm = _np.zeros((3))
        for i in range(3):
            imin = _np.min(self.pos_aa[:, :, i])
            imax = _np.max(self.pos_aa[:, :, i])
            dim_qm[i] = imax - imin
        print('Spread of QM atoms:       %s %s %s'
              % tuple([round(dim, 4) for dim in dim_qm]))


class XYZFrame(_XYZ, _FRAME):
    def _sync_class(self):
        _FRAME._sync_class(self)
        _XYZ._sync_class(self)

    def _make_trajectory(self, **kwargs):
        n_images = kwargs.get('n_images', 3)
        ts_fs = kwargs.get('ts_fs', 1)
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


class XYZIterator():
    '''Testing. Work in progress...'''
    def __init__(self, *args, **kwargs):
        if len(args) != 1:
            raise TypeError("File reader of %s takes at exactly 1 argument!"
                            % self.__class__.__name__)
        else:
            fn = args[0]
            self._fmt = kwargs.get('fmt', fn.split('.')[-1])

            if self._fmt == "xyz":
                self._topology = XYZFrame(fn, **kwargs)
                self._gen = _xyzIterator(fn,
                                         **kwargs
                                         # **_extract_keys(
                                         #          kwargs,
                                         #          range=(0, 1, float('inf')),
                                         #          )
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
                                          # **_extract_keys(
                                          #           kwargs,
                                          #           kinds=symbols,
                                          #           filetype=fn,
                                          #           )
                                          )

                self._gen = _cpmdIterator(fn,
                                          **kwargs
                                          # **_extract_keys(
                                          #         kwargs,
                                          #         kinds=symbols,
                                          #         filetype=fn,
                                          #         range=(0, 1, float('inf')),
                                          #         )
                                          )

            else:
                raise ValueError('Unknown format: %s.' % self._fmt)

            # keep kwargs for iterations
            self._kwargs = kwargs
            # --- TESTING
            self._fr = kwargs.get('range', (0, float('inf')))[0]-1
            next(self)

    def __iter__(self):
        return self

    def __next__(self):
        frame = next(self._gen)

        if self._fmt == 'xyz':
            out = {
                    'data': frame[0],
                    'symbols': frame[1],
                    'comments': frame[2],
                    }

        if self._fmt == 'cpmd':
            frame[:, :3] *= constants.l_au2aa
            out = {
                    'data': frame,
                    'symbols': self._symbols,
                    'comments': self._comments,
                    }
        self._fr += 1
        self._kwargs.update(out)

        self._frame = XYZFrame(**self._kwargs)

        self.__dict__.update(self._frame.__dict__)

        # Get additional keyword-dependent updates
        if self._kwargs.get('align_coords') is not None:
            self._kwargs.update({'align_ref': self._align_ref})

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
            # Does this rewind the original _gen() ??
            return new
        else:
            raise ValueError('Cannot combine frames of different size!')

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        self = self.__add__(other)
        return self

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

    @staticmethod
    def _loop(obj, func, events, *args, **kwargs):
        _fr = 0
        while True:
            try:
                getattr(obj._frame, func)(*args, **kwargs)
                next(obj)
                if _fr in events:
                    if isinstance(events[_fr], dict):
                        kwargs.update(events[_fr])
                _fr += 1
            except StopIteration:
                break

    def write(self, fn, **kwargs):
        return self._loop(self,
                          'write',
                          {0: {'append': True}},
                          fn,
                          **kwargs
                          )


class XYZTrajectory(_XYZ, _TRAJECTORY):
    def _sync_class(self):
        _TRAJECTORY._sync_class(self)
        _XYZ._sync_class(self)

    def _to_frame(self, fr=0):
        return XYZFrame(data=self.data[fr],
                        symbols=self.symbols,
                        comments=[self.comments[fr]])

    def calculate_nuclear_velocities(self, **kwargs):
        '''finite diff, linear (frame1-frame0, frame2-frame1, etc.)'''
        ts = kwargs.get('ts', 0.5)

        if _np.linalg.norm(self.vel_au) != 0:
            _warnings.warn('Overwriting existing velocities in file %s'
                           % self.fn,
                           RuntimeWarning)
        self.vel_au[:-1] = _np.diff(self.pos_aa,
                                    axis=0) / (ts * constants.v_au2aaperfs)


# CLEAN UP and inherit TRAJECTORY


class VibrationalModes():
    # actually here and for the entire MD simulation we should choose one pure
    # isotope since mixed atomic masses are not valid for the determination for
    # "average" mode frequencies (are they?)
    # allow partial reading of modes, insert check for completness of
    # modes 3N-6/5

    def __init__(self, fn, **kwargs):
        fmt = kwargs.get('fmt', fn.split('.')[-1])
        center_coords = kwargs.get('center_coords', False)

        if set(['modes',
                'numbers',
                'omega_cgs',
                'coords_aa']).issubset(kwargs):
            self.fn = fn
            modes = kwargs.get('modes')
            numbers = kwargs.get('numbers')
            omega_cgs = kwargs.get('omega_cgs')
            coords_aa = kwargs.get('coords_aa')
            comments = ["passed"]
            symbols = [constants.symbols[z-1] for z in numbers]
            pos_au = coords_aa*constants.l_aa2au
            eival_cgs = omega_cgs

        elif fmt == "xvibs":
            self.fn = fn
            comments = ["xvibs"]
            n_atoms, numbers, coords_aa, n_modes, omega_cgs, modes =\
                xvibsReader(fn)
            symbols = [constants.symbols[z-1] for z in numbers]
            pos_au = coords_aa*constants.l_aa2au
            eival_cgs = omega_cgs

        else:
            raise ValueError('Unknown format: %s.' % fmt)

        self.pos_au = pos_au
        self.comments = _np.array(comments)
        self.symbols = _np.array(symbols)
        self.masses_amu = _np.array([_masses_amu[s] for s in self.symbols])

        # --- DEBUG corner
        # manually switch mw ON/OFF (it should be clear from the file format)
        mw = kwargs.get('mw', False)

        if mw:
            print('DEBUG info:'
                  'Assuming mass-weighted coordinates in XVIBS.')
            modes /= _np.sqrt(self.masses_amu)[None, :, None] *\
                _np.sqrt(constants.m_amu_au)
        else:
            print('DEBUG info:'
                  'Not assuming mass-weighted coordinates in XVIBS.')
        # ---

        self.eival_cgs = _np.array(eival_cgs)
        self.modes = modes
        # new_eivec = modes*_np.sqrt(self.masses_amu
        # *constants.m_amu_au)[None, :, None]
        new_eivec = modes*_np.sqrt(self.masses_amu)[None, :, None]
        self.eivec = new_eivec  # /_np.linalg.norm(new_eivec,
        # axis=(1, 2))[:, None, None]
        # usually modes have been normalized after mass-weighing,
        # so eivecs have to be normalized again
        self._sync_class()

        # use external function
        if center_coords:
            cell_aa = kwargs.get('cell_aa',
                                 _np.array([0.0, 0.0, 0.0, 90., 90., 90.]))
            if not all([cl == 90. for cl in cell_aa[3:]]):
                print('ERROR: only orthorhombic/cubic cells can be used with '
                      'center function!')
            P = self.pos_au
            M = self.masses_amu
            com_au = _np.sum(P*M[:, None], axis=-2)/M.sum()
            self.pos_au += cell_aa[None, :3] / 2 * constants.l_aa2au -\
                com_au[None, :]

        self._sync_class()
        self._check_orthonormality()

    def _check_orthonormality(self):
        # How to treat rot/trans exclusion? detect 5 or 6?
        atol = 5.E-5
        com_motion = _np.linalg.norm((
                                self.modes * self.masses_amu[None, :, None]
                                ).sum(axis=1), axis=-1)/self.masses_amu.sum()

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
            # DEPRECATED use of 'ignore__warnings'
            print('ERROR: The given cartesian displacements are orthonormal! '
                  'Please try to enable/disable the -mw flag!')
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
            print('ERROR: The eigenvectors are not orthonormal!')
            print(_np.amax(_np.abs(a-_np.identity(self.n_modes))))

    def _sync_class(self):
        self.masses_amu = _np.array([_masses_amu[s] for s in self.symbols])
        self.n_modes, self.n_atoms, three = self.modes.shape
        norm = _np.linalg.norm(self.eivec, axis=(1, 2))
        norm[:6] = 1.0  # trans+rot
        self.eivec /= norm[:, None, None]

    def __add__(self, other):
        new = _copy.deepcopy(self)
        new.pos_au = _np.concatenate((self.pos_au, other.pos_au), axis=0)
        new.symbols = _np.concatenate((self.symbols, other.symbols))
        new.eivec = _np.concatenate((self.eivec, other.eivec), axis=1)
        new.modes = _np.concatenate((self.modes, other.modes), axis=1)
        new._sync_class()
        return new

    def __iadd__(self, other):
        self.pos_au = _np.concatenate((self.pos_au, other.pos_au), axis=0)
        self.symbols = _np.concatenate((self.symbols, other.symbols))
        self.eivec = _np.concatenate((self.eivec, other.eivec), axis=1)
        self.modes = _np.concatenate((self.modes, other.modes), axis=1)
        self._sync_class()
        return self

    def _sort(self):
        elem = {s: _np.where(self.symbols == s)[0]
                for s in _np.unique(self.symbols)}
        ind = [i for k in sorted(elem) for i in elem[k]]
        self.pos_au = self.pos_au[ind, :]
        self.symbols = self.symbols[ind]
        self.masses_amu = self.masses_amu[ind]
        self.eivec = self.eivec[:, ind, :]
        self.modes = self.modes[:, ind, :]

        if hasattr(self, 'vel_au'):
            self.vel_au = self.vel_au[ind, :]
        self._sync_class()

#
#
# UNCOMMENTED BECAUSE OF INCOMPATIBLE IMPORT OF CPMD INTERFACE
# UPDATE THIS CODE USING THE BUILT-IN FUNCTIONS OF CHIRPY
#
#
#    def get_transition_moments(self, source, **kwargs):
#        if source == 'cpmd_nvpt_md':
#            fn_traj = kwargs.get('fn_traj')
#            fn_moms = kwargs.get('fn_moms')
#            modelist = kwargs.get('modelist', range(self.n_modes))
#            # if fn_traj is None or fn_moms is None: Does the next line work ?
#            if None in [fn_traj, fn_moms]:
#                raise Exception('ERROR: Please give fn_traj and fn_moms for '
# 'source "cpmd_nvpt_md"!')
#            cell_au = getattr(self, 'cell_au', None)
#
#            ZV = _np.array([valence_charges[s] for s in self.symbols])
#            n_atoms, n_moms = len(self.symbols), sum(ZV)//2
#
#            self.n_states = n_moms
#            self.c_au = _np.zeros((self.n_modes, 3))
#            self.m_au = _np.zeros((self.n_modes, 3))
#            self._transport_term_au = _np.zeros((self.n_modes, 3))
#            self._r_wc_au = _np.zeros((self.n_modes, self.n_states, 3))
#            self._sw_c_au = _np.zeros((self.n_modes, self.n_states, 3))
#            self._sw_m_dwc_au = _np.zeros((self.n_modes, self.n_states, 3))
#
#            if hasattr(self, 'mol_map'):
#                print('Using molecular gauge.')
#                coms = self.mol_com_au
#                n_map = self.mol_map
#                self.mol_c_au = _np.zeros((self.n_modes, self.n_mols, 3))
#                self.mol_m_au = _np.zeros((self.n_modes, self.n_mols, 3))
#            else:
#                coms=(_np.sum(self.pos_au*self.masses_amu[:,None],axis=0) /\
#        self.masses_amu.sum()).reshape((1, 3))
#                n_map = tuple(_np.zeros((self.n_atoms)).astype(int))
#
#            ZV = _dec(ZV, n_map)
#
#            for i_mode, (pos, vel, wc, c, m) in enumerate(
#              cpmd.get_frame_traj_and_mom(fn_traj, fn_moms, n_atoms, n_moms)):
#                if i_mode >= len(modelist):
#                    print('WARNING: Trajectory file contains more entries '
#               'than given modelist!')
#                    break
#                else:
#                    if not _np.allclose(self.pos_au, pos):
#                        test = _np.unique(_np.around(pos-self.pos_au, 6))
#                        if test.shape == (3, ):
#                            print('WARNING: fn_traj coordinates shifted by '
#                         'vector %s with respect to stored coordinates!'%test)
#                        else:
#                            print('ERROR: fn_traj not consistent with '
#                   'nuclear coordinates!')#, _np.around(pos-self.pos_au, 6))
#                            if not ignore__warnings:
#                                _sys.exit(1)
#                            else:
#                                print('IGNORED')
#
#                    # raw data (for mtm)
#                    self._sw_c_au[modelist[i_mode]] = c
#                    self._sw_m_dwc_au[modelist[i_mode]] = m
#                    self._r_wc_au[modelist[i_mode]] = wc
#
#                    # assign Wannier centers to molecules
#                    dists = pos[_np.newaxis, :, :] - wc[:, _np.newaxis, :]
#                    if hasattr(cell_au, 'shape'):
#                        dists -= _np.around(dists/cell_au)*cell_au
#                    e_map = [n_map[_np.argmin(state_distances)]
#                             for state_distances in _np.sum(dists**2, axis=2)]
#                    # decompose data into molecular contributions
#                    pos, vel = _dec(pos, n_map), _dec(vel, n_map)
#                    wc, c, m = _dec(wc, e_map), _dec(c, e_map), _dec(m, e_map)
#
#                    mol_c, mol_m = list(), list()
#                    for i_mol, com in enumerate(coms):  # wannier2molecules
#                    # calculate nuclear contributions to molecular moments
# and the molecular current dipole moment
#                        el_c = c[i_mol]
#                        el_m = m[i_mol].sum(axis=0)
#                        el_m += magnetic_dipole_shift_origin(wc[i_mol], el_c,
#                                   origin_au=com, cell_au=cell_au)
#
#                        nu_c = current_dipole_moment(vel[i_mol], ZV[i_mol])
#                        nu_m = _np.zeros(nu_c.shape).sum(axis=0)
#                        nu_m += magnetic_dipole_shift_origin(pos[i_mol], nu_c,
#                                    origin_au=com, cell_au=cell_au)
#
#                        mol_c.append(nu_c.sum(axis=0)+el_c.sum(axis=0))
#                        mol_m.append(nu_m + el_m)
#
#                    mol_c = _np.array(mol_c)
#                    mol_m = _np.array(mol_m)
#
#                    for i_mol, com in enumerate(coms):
#                        self.c_au[modelist[i_mode]] += mol_c.sum(axis=0)
#                        self.m_au[modelist[i_mode]] += mol_m.sum(axis=0)
#                           + magnetic_dipole_shift_origin(coms,
#                                    mol_c, origin_au=com, cell_au=cell_au)
#                        self._transport_term_au[modelist[i_mode]] += \
#        magnetic_dipole_shift_origin(coms, mol_c, origin_au=com,
#               cell_au=cell_au)
#
#                    if hasattr(self, 'mol_map'):
#                        self.mol_c_au[modelist[i_mode]] = mol_c
#                        self.mol_m_au[modelist[i_mode]] = mol_m
#            if not i_mode+1 == self.n_modes:
# print('WARNING: Did not find data for all modes. Read only %d modes.'%i_mode)
#
#        elif source == 'cpmd_nvpt_at':
#            fn_APT = kwargs.get('fn_APT')
#            fn_AAT = kwargs.get('fn_AAT')
#            if None in [fn_APT, fn_AAT]:
#               raise Exception('ERROR: Please give fn_APT and fn_AAT for '
# 'source "cpmd_nvpt_at"!')
#            self.c_au = _np.zeros((self.n_modes, 3))
#            self.m_au = _np.zeros((self.n_modes, 3))
#
#            self.APT = _np.loadtxt(fn_APT).astype(float).reshape(self.n_atoms,
# 3, 3)
#            self.AAT = _np.loadtxt(fn_AAT).astype(float).reshape(self.n_atoms,
# 3, 3)
#            sumrule = constants.e_si**2*constants.avog*_np.pi*\
#        _np.sum(self.APT**2/self.masses_amu[:, _np.newaxis, _np.newaxis])/\
#       (3*constants.c_si**2)/constants.m_amu_si
#            print(sumrule)
#            # modes means cartesian displacements
#            self.c_au = (self.modes[:, :, :, _np.newaxis]*\
#        self.APT[_np.newaxis, :, :, :]).sum(axis=2).sum(axis=1)
#            self.m_au = (self.modes[:, :, :, _np.newaxis]*\
#        self.AAT[_np.newaxis, :, :, :]).sum(axis=2).sum(axis=1)
#            # INSERT HERE MOLECULAR GAUGE
#
#        else:  # orca, ...
#            raise Exception('Unknown or unimplemented source: %s.' % source)
#
#    def mtm_calculate_mic_contribution(self, box_vec_aa, source, **kwargs):
#        '''results is origin-independent'''
#        if source == 'cpmd_nvpt_md':  # all data in a.u.
#            fn_e0 = kwargs.get('fn_e0')
#            fn_r1 = kwargs.get('fn_r1')
#            modelist = kwargs.get('modelist')
#            if None in [fn_e0, fn_r1, modelist]:
#                raise Exception('ERROR: Please give fn_e0, fn_r1, and '
# 'modelist for source "cpmd_nvpt_md"!')
#            E0, R1 = cpmd.extract_mtm_data_tmp(fn_e0, fn_r1, len(modelist),
# self.n_states)
#            com_au = _np.sum(self.pos_au*self.masses_amu[:, None], axis=0)/\
#        self.masses_amu.sum()
#            origin_aa = _np.zeros(box_vec_aa.shape)
#            self.m_ic_r_au = _np.zeros((self.n_modes, 3))
#            self.m_ic_t_au = _np.zeros((self.n_modes, 3))
#            for im, mode in enumerate(modelist):
#                r_aa = (self._r_wc_au[mode]-com_au[_np.newaxis, :])*\
#        constants.l_au2aa+origin_aa[_np.newaxis, :]
#                r_aa -= _np.around(r_aa/box_vec_aa)*box_vec_aa
#                m_ic_r, m_ic_t = calculate_mic(E0[im], R1[im],
#               self._sw_c_au[mode], self.n_states, r_aa, box_vec_aa)
#                self.m_ic_r_au[mode] = m_ic_r
#                self.m_ic_t_au[mode] = m_ic_t
#            self.m_lc_au = _copy.deepcopy(self.m_au)
#            self.m_au += self.m_ic_r_au  # +self.m_ic_t_au
#
#    def calculate_mtm_spectrum(self):
#        self.m_au = _copy.deepcopy(self.m_lc_au)
#        self.calculate_spectrum()
#        self.continuous_spectrum(self.n_modes*[1])
#        self.D_cgs_lc = _copy.deepcopy(self.D_cgs)
#        self.R_cgs_lc = _copy.deepcopy(self.R_cgs)
#        self.ira_spec_lc = _copy.deepcopy(self.ira_spec)
#        self.vcd_spec_lc = _copy.deepcopy(self.vcd_spec)
#        self.m_au += self.m_ic_r_au  # +self.m_ic_t_au
#        self.calculate_spectrum()
#        self.continuous_spectrum(self.n_modes*[1])

    def load_localised_power_spectrum(self, fn_spec):
        a = _np.loadtxt(fn_spec)
        self.nu_cgs = a[:, 0]  # unit already cm-1 if coming from molsim
        self.pow_loc = a[:, 1:].swapaxes(0, 1)  # unit?

    def calculate_spectral_intensities(self):  # SHOULDN'T BE METHOD OF CLASS
        self.D_cgs = (self.c_au*self.c_au).sum(axis=1)  # going to be cgs
        self.R_cgs = (self.c_au*self.m_au).sum(axis=1)

        # rot_str_p_p_trans = _np.zeros(n_modes)
        # rot_str_p_p_diff  = _np.zeros(n_modes)
        # NEW STUFF
        #    dip_intensity      = dip_str*IR_int_kmpmol
        #    dip_str           *= dip_str_cgs/omega_invcm
        #    rot_str_m_p       *= rot_str_cgs #**2
        # Understand units later
        atomic_mass_unit = constants.m_amu_au  # 1822.88848367
        ev2au = 1/_np.sqrt(atomic_mass_unit)  # eigenvalue to atomic units
        # ev2wn = ev2au/(2*_np.pi*constants.t_au*constants.c_si)/100
        au2wn = 1/(2*_np.pi*constants.t_au*constants.c_si)/100
        ##############################################################
        # Dipole Strength
        # D_n = APT^2 S_n^2  hbar/(2 omega_n)
        # [S_n^2]     = M^-1  -> (m_u' m_e)^-1
        # [APT^2]     = Q^2   -> e^2
        # [hbar/omega_n] = M L^2 -> m_e l_au^2  (via au2wn/omega)
        # altogether this yields e^2 l_au^2 / m_u'
        # -> needs to be converted to cgs
        e_cgs = constants.e_si*constants.c_si*1E1  # 4.80320427E-10
        l_au_cgs = constants.l_au*1E2
        dip_str_cgs = (e_cgs*l_au_cgs*ev2au)**2*au2wn*1E40/2
        # convert 1/omega back to au
        #################################################################
        # Rotational Strength
        # R_n = 1/c AAT APT S_n^2 omega_n hbar/(2 omega_n) = \
        #        hbar/(2 c) AAT APT S_n^2
        # [S_n^2]     = M^-1  -> (m_u' m_e)^-1
        # [APT]       = Q     -> e
        # [AAT]       = Q L   -> e l_au (NB! c_au is not yet included!)
        # altogether this yields hbar e^2 l_au/(2 c m_e) (/m_u'?!)
        c_cgs = constants.c_si*1E2
        m_e_cgs = constants.m_e_si*1E3
        hbar_cgs = constants.hbar_si*1E7
        # rot_str_cgs = (e_cgs*l_au_cgs*ev2au)**2/constants.c_au*1E44/2
        rot_str_cgs = (hbar_cgs*l_au_cgs*(e_cgs*ev2au)**2)/(c_cgs*m_e_cgs) *\
            1E44/2
        #####################################################################
        # IR Intensity in km/mol-1
        # prefactor (compare master A.97 or Eq. (13) in J. Chem. Phys. 134,
        # 084302 (2011)): Na beta/6 epsilon0 c
        # harmonic approximation A.98-A.99 yields a delta function*pi(*2?) ...
        # beta^-1 = hbar omega / 2 (A.89-A.91)
        # conversion to wavenumbers yields nu = omega/2 pi c a factor of 1/2
        # pi c ([delta(x)] = 1/[x])
        # APTs give (e**2/m_p)
        # conversion in km**-1 gives 1E-3
        # factor 1/3 is due to averaging
        # altogether this yields
        # (N_a e**2)/(6*epsilon0 c m_p)*1E-3
        epsilon0_si = 8.85418782000E-12  # 10(-12)C2.N-1.m-2
        IR_int_kmpmol = (constants.avog*constants.e_si**2)/(12*epsilon0_si *
                                                            constants.c_si**2 *
                                                            constants.m_p_si *
                                                            1000)

        IR_int_kmpmol = (constants.avog*constants.e_si**2)/(12*epsilon0_si *
                                                            constants.c_si**2 *
                                                            constants.m_p_si *
                                                            1000)
        dip_str_cgs = (e_cgs*l_au_cgs*ev2au)**2*au2wn*1E40/2
        # convert 1/omega back to au
        rot_str_cgs = (hbar_cgs*l_au_cgs*(e_cgs*ev2au)**2)/(c_cgs*m_e_cgs) * \
            1E44/2

        self.I_kmpmol = self.D_cgs*IR_int_kmpmol  # here, D_cgs is still au!
        self.D_cgs *= dip_str_cgs/self.eival_cgs
        self.R_cgs *= rot_str_cgs
        # rot_str_p_p_trans *= rot_str_cgs/2
        # rot_str_p_p_diff  *= rot_str_cgs/4
        # VCD intensity?

#    def discrete_spectrum():

    def continuous_spectrum(self,
                            widths=None,
                            nu_min_cgs=0,
                            nu_max_cgs=3800,
                            d_nu_cgs=2):
        def Lorentzian1(x, width, height, position):
            numerator = 1
            denominator = (x-position)**2 + width**2
            y = height*(numerator/denominator)/_np.pi
            return y

        def Lorentzian2(x, width, height, position):
            numerator = width
            denominator = (x-position)**2 + width**2
            y = height*(numerator/denominator)/_np.pi
            return y

        try:
            n_points = self.nu_cgs.shape[0]
            print('Found already loaded spectral data. Using now its '
                  'frequency range.')
        except AttributeError:
            self.nu_cgs = _np.arange(nu_min_cgs, nu_max_cgs, d_nu_cgs)
            n_points = self.nu_cgs.shape[0]

        self.ira_spec = _np.zeros((1+self.n_modes, n_points))
        self.vcd_spec = _np.zeros((1+self.n_modes, n_points))

        D_cgs = self.D_cgs*1E-40  # [esu2cm2]
        R_cgs = self.R_cgs*1E-44  # [esu2cm2]
        prefactor_cgs = (4 * _np.pi**2 * constants.avog) / \
            (3 * constants.c_cgs * constants.hbar_cgs)
        cm2_to_kmcm = 1E-5
        D_scaled = D_cgs*1*prefactor_cgs*cm2_to_kmcm
        R_scaled = R_cgs*4*prefactor_cgs*cm2_to_kmcm

        try:
            for k, nu_k in enumerate(self.eival_cgs):  # units?
                self.ira_spec[k+1, :] = self.pow_loc[k, :]*D_scaled[k]*nu_k
                self.vcd_spec[k+1, :] = self.pow_loc[k, :]*R_scaled[k]*nu_k
                self.ira_spec[0, :] += self.ira_spec[k+1, :]
                self.vcd_spec[0, :] += self.vcd_spec[k+1, :]
            print('Found localised spectral data. I will use it as basis for '
                  'ira and vcd calculation (instead of Lorentzian).')
        except AttributeError:
            if widths is None:
                widths = self.n_modes*[1]
            for k, nu_k in enumerate(self.eival_cgs):
                self.ira_spec[k+1, :] = Lorentzian2(self.nu_cgs,
                                                    widths[k],
                                                    D_scaled[k]*nu_k,
                                                    nu_k)
                self.vcd_spec[k+1, :] = Lorentzian2(self.nu_cgs,
                                                    widths[k],
                                                    R_scaled[k]*nu_k,
                                                    nu_k)
                self.ira_spec[0, :] += self.ira_spec[k+1, :]
                self.vcd_spec[0, :] += self.vcd_spec[k+1, :]

    # Needs more testing
    # The velocities are so small?
    # SHOULDN'T BE MATHOD OF CLASS

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
            self.vel_au = S
            e_kin_au = _calculate_kinetic_energies(self.vel_au,
                                                   self.masses_amu)
            scale = temperature / (_np.sum(e_kin_au) / constants.k_B_au /
                                   self.n_modes) / 2
            print(scale)
            self.vel_au *= scale

        elif occupation == 'average':
            self.vel_au = S.sum(axis=0)
            # atomic_ekin_au = traj_utils.CalculateKineticEnergies(avg,
            # _masses_amu)
            # scale = temperature/(_np.sum(atomic_ekin_au)/
            # constants.k_B_au/n_modes)/2
            # print(scale)
            # avg *= _np.sqrt(scale)
        elif occupation == 'random':  # NOT TESTED!
            phases = _np.random.rand(self.n_modes)*_np.pi
            self.vel_au = (S*_np.cos(phases)[:, None, None]).sum(axis=0)

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

    def write_nuclear_velocities(self, fn, **kwargs):
        fmt = kwargs.get('fmt', 'cpmd')  # fn.split('.')[-1])
        modelist = kwargs.get('modelist', range(self.n_modes))
        factor = kwargs.get('factor', 1.0)
        # loc_n_modes = len(modelist)
        if fmt == 'cpmd':
            print('CPMD WARNING: Output with sorted atomlist!')
            loc_self = _copy.deepcopy(self)
            loc_self._sort()
            pos_au = _np.tile(loc_self.pos_au, (loc_self.n_modes, 1, 1))
            try:
                cpmdWriter(fn,
                           pos_au[modelist],
                           loc_self.symbols,
                           factor*loc_self.vel_au[modelist],
                           offset=0,
                           **kwargs
                           )  # DEFAULTS pp='MT_BLYP', bs=''
            except AttributeError:
                loc_self.calculate_nuclear_velocities()
                cpmdWriter(fn,
                           pos_au[modelist],
                           loc_self.symbols,
                           factor*loc_self.vel_au[modelist],
                           offset=0,
                           **kwargs
                           )  # DEFAULTS pp='MT_BLYP', bs=''
            del loc_self

        elif fmt == "xyz":  # posvel, only vel not implemented
            pos_aa = _np.tile(self.pos_au * constants.l_au2aa,
                              (self.n_modes, 1, 1))
            try:
                xyzWriter(fn,
                          _np.concatenate((pos_aa, factor*self.vel_au),
                                          axis=-1)[modelist],
                          self.symbols,
                          [str(m) for m in modelist]
                          )
            except AttributeError:
                self.calculate_nuclear_velocities()
                xyzWriter(fn,
                          _np.concatenate((pos_aa, factor*self.vel_au),
                                          axis=-1)[modelist],
                          self.symbols,
                          [str(m) for m in modelist]
                          )

        else:
            raise ValueError('Unknown format: %s' % fmt)

    def write(self, fn, **kwargs):
        fmt = kwargs.get('fmt', 'xyz')
        # ToDo: compare this to _XYZ._make_trajectory in trajectory class
        if fmt == "xyz":
            modelist = kwargs.get('modelist', range(self.n_modes))
            n_images = kwargs.get('n_images', 3)
            img = _np.arange(-(n_images//2), n_images//2+1)
            ts_fs = kwargs.get('ts_fs', 1)
            pos_aa = _np.tile(self.pos_au*constants.l_au2aa, (n_images, 1, 1))
            self.calculate_nuclear_velocities()
            for mode in modelist:
                vel_aa = _np.tile(
                        self.vel_au[mode] * constants.t_fs2au *
                        constants.l_au2aa,
                        (n_images, 1, 1)
                        )*img[:, None, None]
                pos_aa += vel_aa*ts_fs
                xyzWriter('%03d' % mode+'-'+fn,
                          pos_aa,
                          self.symbols,
                          [str(m) for m in img]
                          )
#        elif fmt == 'cpmd': #not finished
#            cpmd.WriteTrajectoryFile(fn, pos_aa, self.vel_au[mode], offset=0)
#        elif fmt == "xvibs":
#            xvibsWriter(filename,
#                        n_atoms,
#                        numbers,
#                        pos_aa,
#                        freqs,
#                        modes
#                        )
#

            else:
                raise ValueError('Unknown format: %s' % fmt)


class NormalModes(VibrationalModes):
    # Hessian Things
    pass
