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

import copy as _copy
import numpy as _np
import warnings as _warnings

from . import _CORE, _ITERATOR
from .. import extract_keys as _extract_keys
from ..read.modes import xvibsReader
from ..read.coordinates import xyzReader, pdbReader, cifReader, arcReader
from ..read.grid import cubeReader
from ..read.coordinates import xyzIterator as _xyzIterator
from ..read.coordinates import cpmdIterator as _cpmdIterator
from ..read.coordinates import pdbIterator as _pdbIterator
from ..read.coordinates import arcIterator as _arcIterator
from ..write.coordinates import xyzWriter, pdbWriter, arcWriter
from ..write.modes import xvibsWriter

from ..interface.orca import orcaReader
from ..interface.cpmd import cpmdReader, cpmdWriter, cpmd_kinds_from_file
from ..interface.molden import write_moldenvib_file
from ..interface.gaussian import g09Reader

from ..topology import mapping, motion
from ..topology.dissection import read_topology_file

from ..physics import constants
from ..physics.statistical_mechanics import kinetic_energies as \
        _kinetic_energies
from ..physics.classical_electrodynamics import current_dipole_moment as \
        _current_dipole_moment
from ..physics.spectroscopy import absorption_from_transition_moment as \
        _absorption_from_transition_moment

from ..mathematics import algebra as _algebra

# NB: data is accessed from behind (axis_pointer):
#   frame is (N,X)
#   trajectory is (F,N,X),
#   list of modes is (M,F,N,X)


class _FRAME(_CORE):
    def _labels(self):
        self._type = 'frame'
        self._labels = ('symbols',  None)

    def __init__(self, *args, **kwargs):
        self._labels()
        self._read_input(*args, **kwargs)
        self._sync_class()

    def _read_input(self,  *args, **kwargs):
        self.symbols = kwargs.get('symbols', ())
        self.data = kwargs.get('data', _np.zeros((0, 0)))

    def _sync_class(self):
        self.axis_pointer = -2

        if not isinstance(self.symbols, tuple):
            raise TypeError('Expected tuple for symbols attribute. '
                            f'Got {self.symbols.__class__.__name__}')
        # --- optional attributes (symbols analogues)
        for key in ['names', 'residues']:
            if not isinstance((_oa := getattr(self, key, None)),
                              (tuple, type(None))):
                raise TypeError(f'Expected tuple for {key} attribute. '
                                f'Got {_oa.__class__.__name__}')
        self.n_atoms, self.n_fields = self.data.shape
        if self.n_atoms != len(self.symbols):
            raise ValueError('Data shape inconsistent '
                             'with symbols attribute! '
                             '{} {}'.format(self.data.shape,
                                            self.symbols))

    def __add__(self, other):
        new = _copy.deepcopy(self)
        new.data = _np.concatenate((self.data, other.data),
                                   axis=self.axis_pointer)
        _l = new._labels[self.axis_pointer]
        if _l is not None:
            setattr(new, _l, getattr(self, _l) + getattr(other, _l))

        # --- symbol analogues
        if _l == 'symbols':
            for _l in ['names', 'residues']:
                try:
                    setattr(new, _l, getattr(self, _l) + getattr(other, _l))
                except AttributeError:
                    pass

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

    def sort(self, *args):
        _symbols = _np.array(self.symbols)

        # --- scratch for in-depth sort that includes data
        # def get_slist():
        #     elem = {s: (lambda x: _np.lexsort((x[:, 0], x[:, 1], x[:, 2])))(
        #                self.data.swapaxes(0, -2)[_np.where(_symbols == s)[0]]
        #                ).tolist()
        #             for s in _np.unique(_symbols)}
        #     return [i for k in sorted(elem) for i in elem[k]]

        def get_slist():
            elem = {s: _np.where(_symbols == s)[0]
                    for s in _np.unique(_symbols)}
            return [i for k in sorted(elem) for i in elem[k]]

        if len(args) == 1:
            _slist = list(args[0])
        else:
            _slist = get_slist()

        self.data = self.data.swapaxes(0, -2)[_slist].swapaxes(0, -2)
        self.symbols = tuple(_symbols[_slist])
        # --- symbols analogues
        for _l in ['names', 'residues']:
            try:
                setattr(self, _l, tuple(_np.array(getattr(self,
                                        _l))[_slist].tolist()))
            except AttributeError:
                pass
        self._sync_class()

        return _slist

    def _is_similar(self, other):
        ie = list(map(lambda a: getattr(self, a) == getattr(other, a),
                      ['_type', 'n_atoms', 'n_fields']))
        ie.append(bool(_np.prod([a == b
                                for a, b in zip(_np.sort(self.symbols),
                                                _np.sort(other.symbols))])))
        return _np.prod(ie), ie

    def split(self, mask, select=None):
        '''select ... list or tuple of ids'''
        _data = [_np.moveaxis(_d, 0, -2)
                 for _d in mapping.dec(_np.moveaxis(self.data, -2, 0), mask)]
        _symbols = mapping.dec(self.symbols, mask)

        _DEC = (_data, _symbols)

        # --- ToDo: Generalise optional attributes
        for _l in ['names', 'residues']:
            if hasattr(self, _l):
                _dl = mapping.dec(getattr(self, _l), mask)
                _DEC += (_dl,)

        def create_obj(_d, _s, *optargs):
            nargs = {}
            nargs.update(self.__dict__)
            nargs.update({'data': _d, 'symbols': _s})
            for _oa, key in zip(optargs, ['names', 'residues']):
                nargs.update({key: _oa})
            _obj = self._from_data(**nargs)
            return _obj

        if select is None:
            _new = []
            # for _d, _s in zip(_data, _symbols):
            for _D in zip(*_DEC):
                _new.append(create_obj(*_D))
            return _new
        else:
            if isinstance(select, int):
                select = [select]
            elif not isinstance(select, list):
                raise TypeError('Expected list or integer for select '
                                'argument!')
            _iselect = [_i for _i, _m in enumerate(set(mask)) if _m in select]
            if len(_iselect) > 0:
                # _new = create_obj(_data[_iselect[0]], _symbols[_iselect[0]])
                _new = create_obj(*[_D[_iselect[0]] for _D in _DEC])
                for _id in _iselect[1:]:
                    # _new += create_obj(_data[_id], _symbols[_id])
                    _new += create_obj(*[_D[_id] for _D in _DEC])
            else:
                raise ValueError('Selection does not correspond to any '
                                 'mask entry!')
            self.__dict__.update(_new.__dict__)
            self._sync_class()

    @staticmethod
    def map_frame(obj1, obj2):
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

        com1 = mapping.cowt(obj1.pos_aa, obj1.masses_amu, axis=-2)
        com2 = mapping.cowt(obj2.pos_aa, obj2.masses_amu, axis=-2)

        assign = _np.zeros((obj1.n_atoms,)).astype(int)
        for s in _np.unique(obj1.symbols):
            i1 = _np.array(obj1.symbols) == s
            i2 = _np.array(obj2.symbols) == s
            ass = _np.argmin(mapping.distance_matrix(
                                obj1.pos_aa[i1] - com1[None],
                                obj2.pos_aa[i2] - com2[None],
                                cell=obj1.cell_aa_deg
                                ),
                             axis=0)
            assign[i1] = _np.arange(obj2.n_atoms)[i2][ass]

        with _warnings.catch_warnings():
            if not len(_np.unique(assign)) == obj1.n_atoms:
                _warnings.warn('Ambiguities encountered when mapping frames!',
                               RuntimeWarning, stacklevel=2)

        return assign

    @classmethod
    def _from_data(cls, **kwargs):
        return cls(**kwargs)


class _TRAJECTORY(_FRAME):
    def _labels(self):
        self._type = 'trajectory'
        self._labels = ('comments', 'symbols', None)

    def _read_input(self,  *args, **kwargs):
        self.comments = kwargs.get('comments', [])
        self.symbols = kwargs.get('symbols', ())
        self.data = kwargs.get('data', _np.zeros((0, 0, 0)))

    def _sync_class(self):
        self.n_frames, self.n_atoms, self.n_fields = self.data.shape
        if self.n_atoms != len(self.symbols):
            raise ValueError('Data shape inconsistent with '
                             'symbols attribute!'
                             '{} {}'.format(self.data.shape,
                                            self.symbols))
        if self.n_frames != len(self.comments):
            raise ValueError('Data shape inconsistent with '
                             'comments attribute!'
                             '{} {}'.format(self.data.shape,
                                            self.comments))
        self.axis_pointer = -2


class _MODES(_FRAME):
    def _labels(self):
        self._type = 'modes'
        self._labels = ('comments', 'symbols', None)

    def _sync_class(self, check_orthonormality=True):
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
        self._eivec()

        if not hasattr(self, 'etdm_au'):
            self.etdm_au = _np.zeros((self.n_modes, 3))

        if not hasattr(self, 'mtdm_au'):
            self.mtdm_au = _np.zeros((self.n_modes, 3))

        if not hasattr(self, 'IR_kmpmol'):
            # --- in km/mol
            _warnings.warn('No IR intensities found. Setting to one.',
                           stacklevel=2)
            self.IR_kmpmol = _np.ones((self.n_modes))

        if hasattr(self, 'APT_au'):
            # --- first step to generate eivec
            self._eivec()
            self.etdm_au = (
                    self.APT_au[None, :, :, :]
                    * (
                     self.eivec
                     / _np.sqrt(self.masses_amu)[None, :, None])[:, :, :, None]
                    ).sum(axis=(1, 2))

        if hasattr(self, 'AAT_au'):
            # --- first step to generate eivec
            self._eivec()
            self.mtdm_au = (
                    self.AAT_au[None, :, :, :]
                    * (
                     self.eivec
                     / _np.sqrt(self.masses_amu)[None, :, None])[:, :, :, None]
                    ).sum(axis=(1, 2))

        if check_orthonormality:
            self._check_orthonormality()

    def _modelist(self, modelist):
        if not isinstance(modelist, list):
            raise TypeError('Please give a list of integers instead of %s!'
                            % modelist.__class__.__name__)
        self.data = self.data[modelist]
        self.comments = [self.comments[_m] for _m in modelist]
        self._sync_class(check_orthonormality=False)

    def _modes(self, *args):
        if len(args) == 0:
            self.modes = self.data.swapaxes(0, -1)[6:9].swapaxes(0, -1)
            self._eivec()
        elif len(args) == 1:
            if args[0].shape != self.modes.shape:
                raise ValueError(
                     'Cannot update attribute with values of different shape!')
            _tmp = self.data.swapaxes(0, -1)
            _tmp[6:9] = args[0].swapaxes(0, -1)
            self.data = _tmp.swapaxes(0, -1)
            self._modes()
        else:
            raise TypeError('Too many arguments for %s!'
                            % self._modes.__name__)

    def _eivec(self):
        self.eivec = self.modes * _np.sqrt(self.masses_amu)[None, :, None]
        norm = _np.linalg.norm(self.eivec, axis=(1, 2))

        # --- ToDo: Work Around for trans + rot
        norm[:6] = 1.0
        # ---
        self.eivec /= norm[:, None, None]

    def _check_orthonormality(self):
        atol = 5.E-5
        com_motion = _np.linalg.norm(mapping.cowt(self.modes,
                                                  self.masses_amu,
                                                  axis=1),
                                     axis=-1)

        with _warnings.catch_warnings():
            if _np.amax(com_motion) > atol:
                _warnings.warn('Significant motion of COM for certain modes! '
                               'Are the eigenvectors orthonormal? '
                               'Try enabling/disabling the --mw flag!',
                               RuntimeWarning,
                               stacklevel=2)

        test = self.modes.reshape(self.n_modes, self.n_atoms*3)
        a = _np.inner(test, test)
        _np.set_printoptions(precision=2)
        if any([_np.allclose(a,
                             _np.identity(self.n_modes),
                             atol=atol),
                _np.allclose(a[6:, 6:],
                             _np.identity(self.n_modes-6),
                             atol=atol)]):
            # print(a[6:, 6:])
            _warnings.warn('The given cartesian displacements are '
                           'orthonormal! Please try enabling/disabling '
                           'the --mw flag!',
                           RuntimeWarning,
                           stacklevel=2)
            print(_np.amax(_np.abs(a[6:, 6:]-_np.identity(self.n_modes-6))))
        test = self.eivec.reshape(self.n_modes, self.n_atoms*3)
        a = _np.inner(test, test)
        if not any([_np.allclose(a,
                                 _np.identity(self.n_modes),
                                 atol=atol),
                    _np.allclose(a[6:, 6:],
                                 _np.identity(self.n_modes-6),
                                 atol=atol)
                    ]):
            # print(a[6:, 6:])
            _warnings.warn('The eigenvectors are not '
                           'orthonormal! Please try enabling/disabling '
                           'the --mw flag!',
                           RuntimeWarning,
                           stacklevel=2)
            print(_np.amax(_np.abs(a[6:, 6:]-_np.identity(self.n_modes-6))))
        _np.set_printoptions(precision=8)

    def _source_APT(self, fn):
        '''Requires file of APT in atomic units.
           BETA
           '''
        self.APT_au = _np.loadtxt(fn).astype(float).reshape(self.n_atoms, 3, 3)

        # ToDo: this into check_sumrules / sunc_class
        sumrule = constants.e_si**2 * constants.avog * _np.pi\
            * _np.sum(self.APT**2 / self.masses_amu[:, None, None])\
            / (3 * constants.c_si**2) / constants.m_amu_si
        print(sumrule)

        self._sync_class(check_orthonormality=False)

    def _source_AAT(self, fn):
        '''Requires file of AAT in atomic units.
           BETA
           '''
        self.AAT_au = _np.loadtxt(fn).astype(float).reshape(self.n_atoms, 3, 3)

        # ToDo: this into check_sumrules / sunc_class
        # sumrule = ? --> see old code
        # print(sumrule)

        self._sync_class(check_orthonormality=False)

    def _calculate_spectral_intensities(self, T_K=300):
        '''Calculate IR and VCD intensities from electronic and magnetic
           transition dipole moments.
           '''

        # --- use constants.Abs_au2per_M_cm_mol for conversion
        self.IR_au = _absorption_from_transition_moment(
                                        self.etdm,
                                        self.eival_cgs*constants.E_cm_12au,
                                        T_K=T_K
                                        )
        # self.VCD = (self.etdm_au * self.mtdm_au).sum(axis=-1)


class _XYZ():
    '''Convention (at the moment) of data attribute:
       col 1-3: pos in aa; col 4-6: vel in au.
       pos_aa/vel_au attributes are protected, use underscored
       attributes for changes.'''

    def _read_input(self, *args, **kwargs):
        align_coords = kwargs.pop('align_coords', False)
        center_coords = kwargs.pop('center_coords', False)
        wrap = kwargs.get('wrap', False)
        weight = kwargs.get('weight', 'mass')
        # wrap_molecules = kwargs.get('wrap_molecules', False)
        self.cell_aa_deg = kwargs.get('cell_aa_deg')
        names = kwargs.get('names')  # atom names
        residues = kwargs.get('residues')  # resid + num
        types = kwargs.get('types')
        connectivity = kwargs.get('connectivity')

        if len(args) > 1:
            raise TypeError("File reader of %s takes at most 1 argument!"
                            % self.__class__.__name__)

        elif len(args) == 1:
            fn = args[0]
            fmt = kwargs.get('fmt', fn.split('.')[-1])
            if fmt == 'bz2':
                kwargs.update({'bz2': True})
                fmt = fn.split('.')[-2]

            self._fmt = fmt
            self._fn = fn
            if self._type == 'frame':
                _fr = kwargs.get('frame', 0)
                _fr = _fr, 1, _fr+1
                kwargs.update({'range': _fr})
            elif self._type == 'trajectory':
                _fr = kwargs.get('range', (0, 1, float('inf')))

            if fmt == "xyz":
                data, symbols, comments = xyzReader(fn,
                                                    **_extract_keys(kwargs,
                                                                    range=_fr,
                                                                    bz2=False,
                                                                    )
                                                    )

            elif fmt in ["cube", "cub"]:
                data, origin_aa, cell_vec_aa, pos_aa, numbers, comments = \
                        cubeReader(fn, **_extract_keys(kwargs, bz2=False))
                _dims = data.shape[1:]
                del data
                # --- ToDo: to be discussed: adding origin
                data = (pos_aa + origin_aa)
                symbols = constants.numbers_to_symbols(numbers)
                cell_aa_deg = mapping.get_cell_l_deg(
                        cell_vec_aa,
                        multiply=_dims
                        )

                if cell_aa_deg is not None:
                    if self.cell_aa_deg is None:
                        self.cell_aa_deg = cell_aa_deg

                    elif not _np.allclose(self.cell_aa_deg, cell_aa_deg):
                        _warnings.warn('Overwriting cell parametres '
                                       'of CUBE file!',
                                       RuntimeWarning,
                                       stacklevel=2)

            elif fmt == "pdb":
                data, names, symbols, residues, cell_aa_deg, title = \
                        pdbReader(fn)
                n_atoms = len(symbols)
                comments = kwargs.get('comments', title)

                if cell_aa_deg is not None:
                    if self.cell_aa_deg is None:
                        self.cell_aa_deg = cell_aa_deg

                    elif not _np.allclose(self.cell_aa_deg, cell_aa_deg):
                        _warnings.warn('Overwriting cell parametres '
                                       'of PDB file!',
                                       RuntimeWarning,
                                       stacklevel=2)

            elif fmt == "cif":
                data, names, symbols, cell_aa_deg, title = cifReader(fn)
                comments = kwargs.get('comments', title)

                if self.cell_aa_deg is None:
                    self.cell_aa_deg = cell_aa_deg

                elif not _np.allclose(self.cell_aa_deg, cell_aa_deg):
                    _warnings.warn('Overwriting cell parametres '
                                   'of CIF file!',
                                   RuntimeWarning,
                                   stacklevel=2)

            elif fmt == "xvibs":
                nargs = _extract_keys(kwargs, mw=False, au=False)
                n_atoms, numbers, pos_aa, \
                    n_modes, omega_cgs, modes = xvibsReader(fn, **nargs)
                comments = _np.array(omega_cgs).astype(str)
                symbols = constants.numbers_to_symbols(numbers)
                data = _np.concatenate((
                           _np.tile(pos_aa, (n_modes, 1, 1)),
                           _np.tile(_np.zeros_like(pos_aa), (n_modes, 1, 1)),
                           modes
                          ),
                          axis=-1
                          )

            elif fmt == "cpmd" or any([_t in fn for _t in [
                                     'TRAJSAVED', 'GEOMETRY', 'TRAJECTORY']]):
                fmt = "cpmd"
                if 'symbols' not in kwargs:
                    kwargs.update({
                        'symbols': cpmd_kinds_from_file(fn)
                        })
                data_dict = cpmdReader(fn, **kwargs)
                self._data_dict = data_dict

                if 'symbols' in data_dict:
                    symbols = data_dict['symbols']
                    data = data_dict['data']
                    comments = data_dict['comments']
                else:
                    raise ValueError('File %s does not contain atom '
                                     'information!' % fn)

            elif fmt in ['tinker', 'arc', 'vel']:
                fmt = "tinker"
                data, symbols, indices, types, connectivity, comments =\
                    arcReader(fn,
                              **_extract_keys(kwargs,
                                              range=_fr,
                                              bz2=False,
                                              )
                              )

            elif fmt == "orca":
                data_dict = orcaReader(fn)
                self._data_dict = data_dict

                if 'symbols' in data_dict:
                    symbols = data_dict['symbols']
                    if 'modes' in data_dict:
                        modes = data_dict['modes']
                        n_modes = modes.shape[0]
                        pos_aa = data_dict['pos_aa']
                        omega_cgs = data_dict['omega_cgs']
                        comments = _np.array(omega_cgs).astype(str)
                        data = _np.concatenate((
                             _np.tile(pos_aa, (n_modes, 1, 1)),
                             _np.tile(_np.zeros_like(pos_aa), (n_modes, 1, 1)),
                             modes
                            ),
                            axis=-1
                            )
                        self.IR_kmpmol = data_dict['T**2']
                        self.APT_au = data_dict['APT_au']
                    else:
                        raise NotImplementedError('Cannot read file %s!' % fn)

                else:
                    raise ValueError('File %s does not contain atoms!' % fn)

            elif fmt in ["g09", 'gaussian']:
                nargs = _extract_keys(kwargs, run=1)
                data_dict = g09Reader(fn, **nargs)
                self._data_dict = data_dict

                if 'symbols' in data_dict:
                    symbols = data_dict['symbols']
                    if 'modes' in data_dict:
                        modes = data_dict['modes']
                        pos_aa = data_dict['pos_aa']
                        n_modes = modes.shape[0]
                        n_atoms = pos_aa.shape[0]
                        modes = modes.reshape((n_modes, n_atoms, 3))
                        omega_cgs = data_dict['omega_cgs']
                        comments = _np.array(omega_cgs).astype(str)

                        data = _np.concatenate((
                             _np.tile(pos_aa, (n_modes, 1, 1)),
                             _np.tile(_np.zeros_like(pos_aa), (n_modes, 1, 1)),
                             modes
                            ),
                            axis=-1
                            )
                        try:
                            # --- ToDo: add and debug these features
                            # self.APT_au = data_dict['Polar']
                            # self.AAT_au = data_dict['AAT']
                            # --- calculate it from tensors
                            self.IR_kmpmol = data_dict['T**2']

                        except KeyError:
                            pass

                    else:
                        raise NotImplementedError('Cannot read file %s!' % fn)

            elif fmt in ['cp2k', 'restart', 'inp']:
                # --- single frame only
                _dict = read_topology_file(fn)
                symbols = _dict['symbols']
                names = _dict['names']
                comments = _dict['comments_topo']
                self.cell_aa_deg = _dict['cell_aa_deg']
                data = _dict['data_topo']

            else:
                raise ValueError('Unknown format: %s.' % fmt)

        elif len(args) == 0:
            if 'data' in kwargs and ('symbols' in kwargs
                                     or 'numbers' in kwargs):
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

        comments = list(comments)
        if self._type == 'frame':
            # --- NB: frame selection has been made above
            data = data[0]
            comments = comments[0]

        if self._type == 'modes':
            if 'omega_cgs' not in locals():
                raise TypeError('Could not retrieve modes data from input!')
            self.eival_cgs = omega_cgs
            comments = _np.array(omega_cgs).astype(str)

        # --- external metadata (i.e. from topology file)
        #     wins over fn, but not in the case of data
        self.symbols = kwargs.get('symbols', tuple(symbols))
        self.comments = kwargs.get('comments', comments)
        self.data = data

        # --- optional
        if names is not None:
            self.names = tuple(names)
        if residues is not None:
            self.residues = tuple(residues)
        if types is not None:
            self.types = tuple(types)
        if connectivity is not None:
            self.connectivity = tuple(connectivity)

        self._sync_class(check_orthonormality=False)

        if bool(center_coords):
            selection = None
            if isinstance(center_coords, list):
                selection = center_coords
            self.center_coordinates(selection=selection,
                                    weight=weight, wrap=wrap)

        if wrap:
            self.wrap_atoms()
        # if wrap_molecules:
        #     try:
        #         self.wrap_molecules(kwargs['mol_map'])
        #     except KeyError:
        #         with _warnings.catch_warnings():
        #             _warnings.warn('Could not find mol_map for wrapping!',
        #                            RuntimeWarning, stacklevel=2)
        #         self.wrap_atoms()

        if bool(align_coords):
            selection = None
            if isinstance(align_coords, list):
                selection = align_coords
            if wrap:  # or wrap_molecules:
                _warnings.warn('Disabling wrapping for atom alignment!',
                               stacklevel=2)
                # unnecessary here
                wrap = False
                wrap_molecules = False
            self.align_coordinates(
                           selection=selection,
                           weight=weight,
                           align_ref=kwargs.get('align_ref'),
                           force_centering=kwargs.get('force_centering', False)
                           )

        # DISABLED 2020-10-20
        # self._sync_class(check_orthonormality=False)

    def _pos_aa(self, *args):
        '''Update positions'''
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
        '''Update velocities'''
        if len(args) == 0:
            self.vel_au = self.data.swapaxes(0, -1)[3:6].swapaxes(0, -1)
            if self.vel_au.size == 0:
                # ToDo: slow
                # --- vel_au has to be set!
                self.data = _np.concatenate((self.pos_aa,
                                             _np.zeros_like(self.pos_aa)),
                                            axis=-1)
                self._vel_au()
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
        # ToDo: kwargs for consistency with Modes
        # ToDo: slow (problem in iterator)
        try:
            self.masses_amu = constants.symbols_to_masses(self.symbols)
        except KeyError:
            _warnings.warn('Could not find masses for all elements! '
                           'Centre of mass cannot be used.',
                           RuntimeWarning, stacklevel=2)
        # DISABLED 2020-10-20
#        self._pos_aa()
#        self._vel_au()
        self.pos_aa = self.data.swapaxes(0, -1)[:3].swapaxes(0, -1)
        self.vel_au = self.data.swapaxes(0, -1)[3:6].swapaxes(0, -1)
        self.cell_aa_deg = _np.array(self.cell_aa_deg)

    def _is_equal(self, other, atol=1e-08, noh=True):
        '''atol adds up to dist_crit_aa from vdw radii'''
        _p, ie = self._is_similar(other)

        def f(a):
            if self._type == 'trajectory':
                raise TypeError('Trajectories cannot be tested for equality '
                                '(only similarity)!')
            _o_pos = _copy.deepcopy(getattr(other, a).reshape(
                                        (1, ) + other.pos_aa.shape
                                        ))
            _s_pos = getattr(self, a)

            _o_pos = mapping.align_atoms(_o_pos, _np.array(self.masses_amu),
                                         ref=_s_pos)[0]
            _bool = []
            for _s in set(self.symbols):
                _ind = _np.array(self.symbols) == _s
                if _s != 'H' or not noh:
                    a = mapping.distance_pbc(_s_pos[_ind],
                                             _o_pos[_ind],
                                             cell=self.cell_aa_deg)
                    a = _np.linalg.norm(a, axis=-1)
                    _bool.append(
                           _np.amax(a) <= mapping.dist_crit_aa([_s])[0] + atol)

            return bool(_np.prod(_bool))

        if _p == 1:
            ie += list(map(f, ['pos_aa']))

        self._sync_class()
        return _np.prod(ie), ie

    # --- join the next two methods?
    def wrap_atoms(self):
        if self._type == 'frame':
            self._pos_aa(mapping.wrap(self.pos_aa.reshape(1, self.n_atoms, 3),
                                      self.cell_aa_deg)[0])
        else:
            self._pos_aa(mapping.wrap(self.pos_aa, self.cell_aa_deg))

    def wrap_molecules(self, mol_map, weight='mass'):
        w = _np.ones((self.n_atoms))
        if weight == 'mass':
            w = self.masses_amu

        if self._type == 'trajectory':
            _p, mol_com_aa = mapping.join_molecules(
                                self.pos_aa,
                                mol_map,
                                self.cell_aa_deg,
                                weights=w,
                                )
            self._pos_aa(_p)
            self.mol_com_aa = mol_com_aa
        else:
            _p, mol_com_aa = mapping.join_molecules(
                                self.pos_aa,
                                mol_map,
                                self.cell_aa_deg,
                                weights=w,
                                )
            self._pos_aa(_p)
            self.mol_com_aa = mol_com_aa

    def align_coordinates(self, selection=None, weight='mass', align_ref=None,
                          force_centering=False):
        '''Aligns positions and rotates (but does not correct)
           velocities.
           '''
        if not isinstance(selection, list):
            if selection is None:
                selection = slice(None)
            else:
                raise TypeError('expected None or list of atoms as selection')

        self._aligned_coords = selection
        wt = _np.ones((self.n_atoms))
        if weight == 'mass':
            wt = self.masses_amu

        if self._type == 'frame':
            _p = self.pos_aa.reshape((1, ) + self.pos_aa.shape)
            _v = self.vel_au.reshape((1, ) + self.vel_au.shape)
        elif self._type == 'trajectory':
            _p = self.pos_aa
            _v = self.vel_au

        if align_ref is None:
            self._align_ref = _copy.deepcopy(_p[0, selection])
        else:
            self._align_ref = align_ref

        _p, _data = mapping.align_atoms(_p,
                                        wt,
                                        ref=self._align_ref,
                                        subset=selection,
                                        data=[_v],
                                        )

        if self._type == 'frame':
            self._pos_aa(_p[0])
            self._vel_au(_data[0][0])

        if self._type == 'trajectory':
            self._pos_aa(_p)
            self._vel_au(_data[0])

        if hasattr(self, '_centered_coords') and force_centering:
            _ref = mapping.cowt(self.pos_aa,
                                wt,
                                subset=self._centered_coords,
                                axis=self.axis_pointer)
            self.center_position(_ref, self.cell_aa_deg)

    def center_coordinates(self, selection=None, weight='mass', wrap=False):
        if not isinstance(selection, list):
            if selection is None:
                selection = slice(None)
            else:
                raise TypeError('expected None or list of atoms as selection')

        self._centered_coords = selection
        wt = _np.ones((self.n_atoms))
        if weight == 'mass':
            wt = self.masses_amu

        if self._type == 'frame':
            _p = self.pos_aa[selection]
        elif self._type == 'trajectory':
            _p = self.pos_aa[:, selection]

        # ---join subset (only for frame)
        if wrap and isinstance(selection, list):
            if self._type == 'frame':
                _p = _np.array([_p])
            _p = mapping.join_molecules(
                    _p,
                    _np.zeros((_p.shape[1])).astype(int),
                    self.cell_aa_deg,
                    )[0]
            if self._type == 'frame':
                _p = _p[0]

        _ref = mapping.cowt(_p,
                            wt[selection],
                            axis=self.axis_pointer)

        self.center_position(_ref, self.cell_aa_deg)

    def center_position(self, pos, cell_aa_deg, wrap=True):
        '''pos reference in shape (n_frames, three)'''
        if self._type == 'frame':
            self._pos_aa(self.pos_aa + cell_aa_deg[None, :3] / 2
                         - pos[None, :])
        else:
            self._pos_aa(self.pos_aa + cell_aa_deg[None, None, :3] / 2
                         - pos[:, None, :])

        if wrap:
            self.wrap_atoms()

    def rotate(self, R, origin_aa=_np.zeros(3)):
        '''Rotate atomic positions and velocities
           R ... rotation matrix of shape (3, 3)
           '''
        if self._type == 'frame':
            _pos = _algebra.rotate_vector(self.pos_aa, R, origin=origin_aa)
            _vel = _algebra.rotate_vector(self.vel_au, R)  # no origin needed
            self._vel_au(_vel)

        elif self._type == 'trajectory':
            _pos = []
            _vel = []
            for _p, _v in zip(self.pos_aa, self.vel_au):
                _pos.append(_algebra.rotate_vector(_p, R))
                _vel.append(_algebra.rotate_vector(_v, R))
            _pos = _np.array(_pos)
            _vel = _np.array(_vel)
            self._vel_au(_vel)

        elif self._type == 'modes':
            _mod = []
            for _p, _m in zip(self.pos_aa, self.modes):
                _mod.append(_algebra.rotate_vector(_m, R))
            _mod = _np.array(_mod)
            self._modes(_mod)

        self._pos_aa(_pos)
        self._vel_au(_vel)

    def align_to_vector(self, i0, i1, vec):
        '''
        Align a reference line pos[i1]-pos[i0] to vec (no pbc support)
        Center of rotation is  pos[i0]. '''

        if self._type == 'frame':
            _ref = self.pos_aa - self.pos_aa[i0, None]
            _R = _algebra.rotation_matrix(_ref[i1], vec)
            _pos = _algebra.rotate_vector(self.pos_aa, _R,
                                          origin=self.pos_aa[i0, None])
            _vel = _algebra.rotate_vector(self.vel_au, _R)
            self._vel_au(_vel)

        else:
            _pos = []
            _vel = []
            for _p, _v in zip(
                    self.pos_aa - self.pos_aa[:, i0, None],
                    self.vel_au
                    ):
                _R = _algebra.rotation_matrix(_p[i1], vec)
                _pos.append(_algebra.rotate_vector(_p, _R))
                _vel.append(_algebra.rotate_vector(_v, _R))

            _pos = _np.array(_pos) + self.pos_aa[:, i0, None]
            _vel = _np.array(_vel)
            self._vel_au(_vel)

        if self._type == 'modes':
            _mod = []
            for _p, _m in zip(
                    self.pos_aa - self.pos_aa[:, i0, None],
                    self.modes
                    ):
                _R = _algebra.rotation_matrix(_p[i1], vec)
                _mod.append(_algebra.rotate_vector(_m, _R))
            _mod = _np.array(_mod)
            self._modes(_mod)

        self._pos_aa(_pos)

    def clean_velocities(self, weights='mass'):
        '''Remove spurious linear and angular momenta from trajectory.
           Positions are not changed.
          '''
        wt = _np.ones((self.n_atoms))
        if weights == 'mass':
            wt = self.masses_amu

        _wt = wt
        if self._type == 'frame':
            _p = self.pos_aa.reshape((1, ) + self.pos_aa.shape)
            _v = self.vel_au.reshape((1, ) + self.vel_au.shape)
        elif self._type == 'trajectory':
            _p = self.pos_aa
            _v = self.vel_au

        _o = mapping.cowt(_p*constants.l_aa2au,
                          _wt,
                          axis=self.axis_pointer)

        _AV, _I = motion.angular_momenta(_p*constants.l_aa2au, _v, _wt,
                                         origin=_o, moI=True)
        _AV /= _I[:, None]
        _lever = _p*constants.l_aa2au - _o[:, None]
        _v -= _np.cross(_AV[:, None], _lever, axis=-1)

        _LV = motion.linear_momenta(_v, _wt / sum(_wt))
        _v -= _LV[:, None]

        if self._type == 'frame':
            self._vel_au(_v[0])
        elif self._type == 'trajectory':
            self._vel_au(_v)

    def write(self, fn, **kwargs):
        attr = kwargs.get('attr', 'data')  # not supported for all formats
        factor = kwargs.get('factor', 1.0)  # for velocities
        fmt = kwargs.get('fmt', fn.split('.')[-1])

        loc_self = _copy.deepcopy(self)

        if self._type == "modes":
            loc_self.n_frames = loc_self.n_modes

        if fmt == "xyz":
            xyzWriter(fn,
                      getattr(loc_self, attr),
                      loc_self.symbols,
                      getattr(loc_self, 'comments'),
                      **_extract_keys(kwargs, append=False)
                      )

        elif fmt in ["arc", "tinker"]:
            arcWriter(fn,
                      getattr(loc_self, attr),
                      loc_self.symbols,
                      comments=getattr(loc_self, 'comments'),
                      **_extract_keys(kwargs,
                                      append=False,
                                      types=getattr(loc_self,
                                                    'types',
                                                    []),
                                      connectivity=getattr(loc_self,
                                                           'connectivity',
                                                           [])
                                      )
                      )

        elif fmt == 'xvibs':
            if not hasattr(self, 'modes'):
                raise AttributeError('Cannot find modes for xvibs output!')
            xvibsWriter(fn,
                        len(loc_self.symbols),
                        constants.symbols_to_numbers(loc_self.symbols),
                        loc_self.pos_aa[1],
                        loc_self.eival_cgs,
                        loc_self.modes,
                        )

        elif fmt == 'molden':
            if not hasattr(self, 'modes'):
                raise AttributeError('Cannot find modes for molden output!')
            write_moldenvib_file(fn,
                                 loc_self.symbols,
                                 loc_self.pos_aa[1],
                                 loc_self.eival_cgs,
                                 loc_self.modes,
                                 )

        elif fmt == "pdb":
            if (mol_map := kwargs.get('mol_map')) is None:
                _warnings.warn('Could not find mol_map for PDB output!',
                               RuntimeWarning, stacklevel=2)
                mol_map = _np.zeros(loc_self.n_atoms).astype(int)

            if (cell_aa_deg := kwargs.get('cell_aa_deg')) is None:
                _warnings.warn("Missing cell parametres for PDB output!",
                               RuntimeWarning, stacklevel=2)
                cell_aa_deg = _np.array([0.0, 0.0, 0.0, 90., 90., 90.])

            pdbWriter(fn,
                      loc_self.pos_aa,
                      names=getattr(loc_self, 'names', loc_self.symbols),
                      symbols=loc_self.symbols,
                      residues=getattr(loc_self, 'residues', _np.vstack((
                                    _np.array(mol_map) + 1,
                                    _np.array(['MOL'] * loc_self.n_atoms)
                                    )).swapaxes(0, 1)),
                      box=cell_aa_deg,
                      title='Generated with ChirPy'
                      )

        elif fmt == 'cpmd':
            if sorted(loc_self.symbols) != list(loc_self.symbols):
                _warnings.warn('CPMD output with sorted atoms!', stacklevel=2)
                loc_self.sort()
            kwargs.update({'symbols': loc_self.symbols})
            loc_self.data = loc_self.data.swapaxes(0, -1)
            loc_self.data[3:] *= factor
            cpmdWriter(fn,
                       loc_self.data.swapaxes(0, -1),
                       **kwargs)

        else:
            raise ValueError('Unknown format: %s.' % fmt)

    def get_atom_spread(self):
        '''pos_aa: np.array of shape ([n_frames,] n_atoms, 3)'''
        dim_qm = _np.zeros((3))
        for i in range(3):
            imin = _np.min(_np.moveaxis(self.pos_aa, -1, 0)[i])
            imax = _np.max(_np.moveaxis(self.pos_aa, -1, 0)[i])
            dim_qm[i] = imax - imin
        print('Spread of QM atoms:       %s %s %s'
              % tuple([round(dim, 4) for dim in dim_qm]))

        if self._type == 'frame':
            amax = _np.amax(mapping.distance_matrix(
                                    self.data[:, :3],
                                    cell=self.cell_aa_deg
                                    ))
        print('Max distance:             %s' % round(amax, 4))


class _MOMENTS():
    '''Object that contains position and moment data very similar
       to _XYZ but more general.
       Supports CPMD input only.
       Everything in atomic units (including the positions and cell).
       '''
    def _read_input(self, *args, **kwargs):
        self.style = kwargs.get('style', 'CPMD 4.1')
        if self.style != 'CPMD 4.1':
            raise NotImplementedError('ChirPy supports only the CPMD 4.1 '
                                      'convention as moments style')

        self.cell_aa_deg = kwargs.get('cell_aa_deg')

        if len(args) > 1:
            raise TypeError("File reader of %s takes at most 1 argument!"
                            % self.__class__.__name__)

        elif len(args) == 1:
            fn = args[0]
            fmt = kwargs.get('fmt', fn.split('.')[-1])
            if fmt == 'bz2':
                kwargs.update({'bz2': True})
                fmt = fn.split('.')[-2]
            self._fmt = fmt
            self._fn = fn
            if self._type == 'frame':
                _fr = kwargs.get('frame', 0)
                _fr = _fr, 1, _fr+1
                kwargs.update({'range': _fr})
            elif self._type == 'trajectory':
                _fr = kwargs.get('range', (0, 1, float('inf')))

            if fmt == "xyz":
                _warnings.warn("XYZ format not tested for MOMENTS!",
                               stacklevel=2)
                data, symbols, comments = xyzReader(fn,
                                                    **_extract_keys(kwargs,
                                                                    range=_fr,
                                                                    bz2=False,
                                                                    )
                                                    )

            elif fmt == "cpmd" or any([_t in fn for _t in [
                                     'MOMENTS']]):
                fmt = "cpmd"
                if 'symbols' not in kwargs:
                    kwargs.update({
                        'symbols': cpmd_kinds_from_file(fn)
                        })
                data_dict = cpmdReader(fn, **kwargs)
                self._data_dict = data_dict

                symbols = data_dict['symbols']
                data = data_dict['data']

                comments = data_dict['comments']

            else:
                raise ValueError('Unknown format: %s.' % fmt)

        elif len(args) == 0:
            if 'data' in kwargs:
                numbers = kwargs.get('numbers', (1,))
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
                raise TypeError('%s needs file or data argument!' %
                                self.__class__.__name__)

        if self.cell_aa_deg is None:
            self.cell_aa_deg = _np.array([0.0, 0.0, 0.0, 90., 90., 90.])

        comments = list(comments)
        if self._type == 'frame':
            _f = kwargs.get("frame", 0)
            data = data[_f]
            comments = comments[_f]

        self.symbols = tuple(symbols)
        self.comments = comments
        self.data = data
        self._sync_class()

    def _sync_class(self):
        self._pos_aa()
        self._c_au()
        self._m_au()
        if self.m_au.size == 0:
            self.m_au = _np.zeros_like(self.pos_aa)
        # self.cell_aa_deg = _np.array(self.cell_aa_deg)

    def write(self, fn, **kwargs):
        attr = kwargs.get('attr', 'data')
        # loc_self = _copy.deepcopy(self)
        # fmt = kwargs.get('fmt', fn.split('.')[-1])
        fmt = kwargs.get('fmt', 'cpmd')

        if fmt == 'cpmd':
            kwargs.update({'symbols': self.symbols})
            cpmdWriter(fn,
                       getattr(self, attr),
                       write_atoms=False,
                       **kwargs)

        else:
            raise ValueError('Unknown format: %s.' % fmt)

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

    def _c_au(self, *args):
        '''Current dipole moments'''
        if len(args) == 0:
            self.c_au = self.data.swapaxes(0, -1)[3:6].swapaxes(0, -1)
        elif len(args) == 1:
            if args[0].shape != self.c_au.shape:
                raise ValueError(
                     'Cannot update attribute with values of different shape!')
            _tmp = self.data.swapaxes(0, -1)
            _tmp[3:6] = args[0].swapaxes(0, -1)
            self.data = _tmp.swapaxes(0, -1)
            self._c_au()
        else:
            raise TypeError('Too many arguments for %s!'
                            % self._c_au.__name__)

    def _m_au(self, *args):
        '''Magnetic dipole moments'''
        if len(args) == 0:
            self.m_au = self.data.swapaxes(0, -1)[6:9].swapaxes(0, -1)
        elif len(args) == 1:
            if args[0].shape != self.m_au.shape:
                raise ValueError(
                     'Cannot update attribute with values of different shape!')
            _tmp = self.data.swapaxes(0, -1)
            _tmp[6:9] = args[0].swapaxes(0, -1)
            self.data = _tmp.swapaxes(0, -1)
            self._m_au()
        else:
            raise TypeError('Too many arguments for %s!'
                            % self._m_au.__name__)


class XYZFrame(_XYZ, _FRAME):
    def _check_distances(self):
        _too_close = mapping.close_neighbours(self.data[:, :3],
                                              cell=self.cell_aa_deg,
                                              crit=0.5)

        for _i in _too_close:
            for _j in _i[1]:
                _warnings.warn(f'Found too close atoms {_i[0]} and {_j[0]} ('
                               f'{_np.round(_j[1], decimals=3)} Å)!',
                               RuntimeWarning, stacklevel=2)

    def _sync_class(self, **kwargs):
        _FRAME._sync_class(self)
        _XYZ._sync_class(self)

    def make_trajectory(self, n_images=3, ts_fs=1):
        '''Create a XYZTrajectory object with <n_images>
           frames from velocities and a timestep ts.'''

        _img = _np.arange(-(n_images // 2), n_images // 2 + 1)
        _pos_aa = _np.tile(self.pos_aa, (n_images, 1, 1))
        _vel_aa = _np.tile(self.vel_au * constants.v_au2aaperfs,
                           (n_images, 1, 1))
        _pos_aa += _vel_aa * _img[:, None, None] * ts_fs

        return _XYZTrajectory(data=_np.dstack((_pos_aa, _vel_aa)),
                              symbols=self.symbols,
                              comments=[self.comments + ' im ' + str(m)
                                        for m in _img]
                              )


class MOMENTSFrame(_MOMENTS, _FRAME):
    def _sync_class(self):
        _FRAME._sync_class(self)
        _MOMENTS._sync_class(self)

    @classmethod
    def from_classical_nuclei(cls, obj, **kwargs):
        '''Convert XYZFrame into _MOMENTS'''
        _pos = obj.data[:, :3]
        _vel = obj.data[:, 3:6]
        ZV = _np.array(constants.symbols_to_valence_charges(obj.symbols))
        _c = _current_dipole_moment(_vel, ZV)
        _m = _np.zeros_like(_c)

        return cls(
               data=_np.concatenate((_pos, _c, _m), axis=-1),
               symbols=obj.symbols,
               **kwargs
               )


class XYZ(_XYZ, _ITERATOR, _FRAME):
    '''A generator of XYZ frames (BETA).'''
    def __init__(self, *args, **kwargs):
        self._kernel = XYZFrame
        self._kwargs = {}
        # --- initialise list of masks
        self._kwargs['_masks'] = []

        if len(args) == 0:
            pass

        elif len(args) == 1:
            fn = args[0]
            self._fn = fn
            fmt = kwargs.get('fmt', fn.split('.')[-1])
            if fmt == 'bz2':
                kwargs.update({'bz2': True})
                fmt = fn.split('.')[-2]
            self._fmt = fmt

            if self._fmt == "xyz":
                self._gen = _xyzIterator(fn, **kwargs)

            elif self._fmt in ["arc", 'vel', 'tinker']:
                self._fmt = "tinker"
                self._gen = _arcIterator(fn, **kwargs)

            elif self._fmt == "pdb":
                self._gen = _pdbIterator(fn)  # **kwargs

            elif self._fmt == "cpmd" or any([_t in fn for _t in [
                                     'TRAJSAVED', 'GEOMETRY', 'TRAJECTORY']]):
                self._fmt = "cpmd"
                if 'symbols' not in kwargs:
                    kwargs.update({
                        'symbols': cpmd_kinds_from_file(fn)
                        })
                self._gen = _cpmdIterator(fn, **kwargs)

            else:
                raise ValueError('Unknown trajectory format: %s.' % self._fmt)

            self._topology = XYZFrame(fn, **kwargs)

            # keep kwargs for iterations
            self._kwargs.update(kwargs)

            self._kwargs['range'] = kwargs.get('range', (0, 1, float('inf')))
            self._fr, self._st, buf = self._kwargs['range']

            # --- Load first frame w/o consuming it
            self._fr -= self._st
            next(self)
            self._chaste = True
            # --- Store original skip as it is consumed by generator
            if 'skip' in self._kwargs:
                self._kwargs['_skip'] = self._kwargs['skip'].copy()

        else:
            raise TypeError("File reader of %s takes exactly 1 argument!"
                            % self.__class__.__name__)

    def __next__(self):
        if hasattr(self, '_chaste'):
            # repeat first step of next() after __init__
            # --- do nothing
            if self._chaste:
                # _warnings.warn("Currently loaded frame repeated after init.",
                #               stacklevel=2)
                self._chaste = False
                return self._fr

        frame = next(self._gen)

        def check_topo(k, f):
            return getattr(self._topology, k, f)

        if self._fmt == 'xyz':
            out = {
                    'data': frame[0],
                    'symbols': check_topo('symbols', frame[1]),
                    'comments': frame[-1],
                    }

        if self._fmt == 'tinker':
            out = {
                    'data': frame[0],
                    'symbols': check_topo('symbols', frame[1]),
                    'types': check_topo('types', frame[3]),
                    'connectivity': check_topo('connectivity', frame[4]),
                    'comments': frame[-1],
                    }

        if self._fmt == 'pdb':
            out = {
                    'data': frame[0],
                    'symbols': check_topo('symbols', frame[2]),
                    'comments': str(frame[-1]),  # if no title: 'None'
                    'cell_aa_deg': frame[-2],
                    'names': check_topo('names', frame[1]),
                    'residues': check_topo('residues', frame[3]),
                    }

        if self._fmt == 'cpmd':
            out = {
                    'data': frame,
                    'symbols': self._topology.symbols,
                    'comments': self._topology.comments,
                    }
        self._fr += self._st
        self._kwargs.update(out)

        self._frame = self._kernel(**self._kwargs)

        # ---check for memory (still a little awkward)
        if hasattr(self._frame, '_align_ref'):
            self._kwargs['align_ref'] = self._frame._align_ref

        # --- check for stored masks
        for _f, _f_args, _f_kwargs in self._kwargs['_masks']:
            if isinstance(_f, str):
                getattr(self._frame, _f)(*_f_args, **_f_kwargs)
            elif callable(_f):
                self._frame = _f(self._frame, *_f_args, **_f_kwargs)

        self.__dict__.update(self._frame.__dict__)

        return self._fr

    def expand(self):
        '''Perform iteration on remaining iterator and load
           entire trajectory
           (may take some time)
           '''
        self._chaste = True
        data, symbols, comments = zip(*[(self.data,
                                         self.symbols,
                                         self.comments) for _fr in self])
        out = {
                'data': _np.array(data),
                'symbols': symbols[0],
                'comments': list(comments)
                 }

        self._kwargs.update(out)

        return _XYZTrajectory(**self._kwargs)

    def write(self, fn, **kwargs):
        self._unwind(fn,
                     func='write',
                     events={0: {'append': True}},
                     **kwargs
                     )
        if kwargs.get('rewind', True):
            self.rewind()

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

    def clean_velocities(self, *args, **kwargs):
        self._frame.clean_velocities(*args, **kwargs)
        self.__dict__.update(self._frame.__dict__)
        self._mask(self, 'clean_velocities', *args, **kwargs)

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
        '''split is faster with fully loaded trajectory'''
        if 'select' not in kwargs:
            _warnings.warn('Splitting iterator without select argument has '
                           'no effect!', stacklevel=2)
        self._frame.split(*args, **kwargs)
        self.__dict__.update(self._frame.__dict__)
        self._mask(self, 'split', *args, **kwargs)

    def merge(self, other, axis=-1, dim1=[0, 1, 2], dim2=[0, 1, 2], **kwargs):
        '''Merge horizontically with another iterator (of equal length).
           Specify axis 0 or 1/-1 to combine atoms or data, respectively
           (default: -1).
           Specify cartesian dimensions to be used from data by dim1/dim2
           (default: [0, 1, 2]).
           <Other> iterator must not be used anymore!
           To concatenate iterators along the frame axis, use "+".
           BETA'''

        def _add(obj1, obj2):
            '''combine two frames'''
            obj1.axis_pointer = axis
            obj2.axis_pointer = axis

            obj1.data = obj1.data[:, dim1]
            obj2.data = obj2.data[:, dim2]

            obj1 += obj2
            return obj1

        def _func(obj1, obj2, **kwargs):
            # --- next(obj1) is called before loading mask
            try:
                next(obj2)
                return _add(obj1, obj2)
            except StopIteration:
                with _warnings.catch_warnings():
                    _warnings.warn('Merged iterator exhausted!',
                                   RuntimeWarning,
                                   stacklevel=1)
                return obj1

        self._frame = _add(self._frame, other._frame)
        self.__dict__.update(self._frame.__dict__)
        other._chaste = False

        self._mask(self, _func, other, **kwargs)


class MOMENTS(_MOMENTS, _ITERATOR, _FRAME):
    '''A generator of MOMENT frames (BETA).'''
    def __init__(self, *args, **kwargs):
        self._kernel = MOMENTSFrame

        self._kwargs = {}
        # --- initialise list of masks
        self._kwargs['_masks'] = []

        if len(args) == 0:
            pass

        elif len(args) == 1:
            fn = args[0]
            self._fn = fn
            fmt = kwargs.get('fmt', fn.split('.')[-1])
            if fmt == 'bz2':
                kwargs.update({'bz2': True})
                fmt = fn.split('.')[-2]
            self._fmt = fmt

            # self._topology = XYZFrame(fn, **kwargs)
            if self._fmt == "cpmd" or 'MOMENTS' in fn:
                self._fmt = "cpmd"
                if 'symbols' not in kwargs:
                    with _warnings.catch_warnings():
                        _warnings.filterwarnings('ignore',
                                                 category=UserWarning)
                        kwargs.update({
                            'symbols': cpmd_kinds_from_file(fn)
                            })
                self._gen = _cpmdIterator(fn, filetype='MOMENTS', **kwargs)

            else:
                raise ValueError('Unknown format: %s.' % self._fmt)

            # --- keep kwargs for iterations
            self._kwargs.update(kwargs)

            self._kwargs['range'] = kwargs.get('range', (0, 1, float('inf')))
            self._fr, self._st, buf = self._kwargs['range']

            # --- Load first frame w/o consuming it
            self._fr -= self._st
            next(self)
            self._chaste = True

        else:
            raise TypeError("File reader of %s takes exactly 1 argument!"
                            % self.__class__.__name__)

    def expand(self):
        '''Perform iteration on remaining iterator and load
           entire trajectory
           (may take some time)
           '''
        self._chaste = True
        data = [self.data for _fr in self]
        out = {'data': _np.array(data)}

        self._kwargs.update(out)

        return _MOMENTSTrajectory(**self._kwargs)

    def write(self, fn, **kwargs):
        self._unwind(fn,
                     func='write',
                     events={0: {'append': True}},
                     **kwargs
                     )
        if kwargs.get('rewind', True):
            self.rewind()


class _XYZTrajectory(_XYZ, _TRAJECTORY):
    '''Load full XYZ trajectory into memory'''

    def _sync_class(self, **kwargs):
        _TRAJECTORY._sync_class(self)
        _XYZ._sync_class(self)

    def calculate_nuclear_velocities(self, ts=0.5):
        '''finite diff, linear (frame1-frame0, frame2-frame1, etc.)'''
        if _np.linalg.norm(self.vel_au) != 0:
            _warnings.warn('Overwriting existing velocities in object!',
                           stacklevel=2)
        self.vel_au[:-1] = _np.diff(self.pos_aa,
                                    axis=0) / (ts * constants.v_au2aaperfs)


class _MOMENTSTrajectory(_MOMENTS, _TRAJECTORY):
    '''Load full MOMENTS trajectory into memory'''

    def _sync_class(self, **kwargs):
        _TRAJECTORY._sync_class(self)
        _MOMENTS._sync_class(self)


class VibrationalModes(_XYZ, _MODES):
    def _sync_class(self, **kwargs):
        # --- keep order for APT/AAT calculation
        _XYZ._sync_class(self)
        _MODES._sync_class(self, **kwargs)

    def calculate_nuclear_velocities(self, occupation='single',
                                     temperature=300):
        '''Occupation can be single, average, or random.'''
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
                        comments=self.comments[mode],
                        **kwargs
                        )


class NormalModes(VibrationalModes):
    pass
