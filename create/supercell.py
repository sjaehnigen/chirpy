# !/usr/bin/env python
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
import os as _os
import copy as _copy
import numpy as _np
import warnings as _warnings

from ..topology.dissection import assign_molecule as _assign_molecule
# from ..topology.mapping import get_atom_spread as _get_atom_spread
from ..topology.mapping import get_cell_vec as _get_cell_vec
from ..topology.mapping import detect_lattice as _get_symmetry
from ..classes.core import _CORE
from ..classes.trajectory import XYZFrame as _XYZFrame
from ..classes.system import Supercell as _Supercell
from ..physics import constants
from ..mathematics.algebra import angle


class _BoxObject(_CORE):

    # --- DEV log
    # volume is determined by _cell_vec_aa() / cell_vec_aa()
    # methods that are allowed to manipulate _cell_vec_aa and/or volume_aa3:
    # __init__, __pow__ (via __mul__)
    # plan: check routine compares _method() to method and complains
    # Symmetry should be determined from cell_aa
    # empty-box init allowed (bare)
    # --- END

    def __init__(self, *args, **kwargs):
        '''
        members: list of (n, XYZFrame object) tuples with n being
                 the no. of molecules within the box'''
        self.members = kwargs.get("members", [])
        self.member_set = kwargs.get("members", [])
        self.origin_aa = kwargs.get('origin_aa', _np.zeros((3)).astype(float))
        self.cell_aa_deg = kwargs.get('cell_aa_deg', _np.zeros(6,))
        self.pbc = kwargs.get('pbc', True)
        if len(args) != 0:
            self.__dict__.update(self.read(*args, **kwargs).__dict__)
        self.symmetry = kwargs.get('symmetry', _get_symmetry(self.cell_aa_deg))
        _BoxObject._sync_class(self)
        self.cell_vec_aa = self._cell_vec_aa(**kwargs)
        self.volume_aa3 = self._volume_aa3()
        self._clean_members()

    @classmethod
    def read(cls, *args, **kwargs):
        if kwargs.get('install_mol_gauge') is not None:
            kwargs['wrap_mols'] = True
        else:
            kwargs['wrap'] = True
        _load = _Supercell(*args, **kwargs)

        nargs = {}
        nargs['cell_aa_deg'] = _load.cell_aa_deg
        if _np.any(_load.cell_aa_deg == 0.):
            _warnings.warn('Cannot detect cell dimensions!',  # Guessing.',
                           UserWarning)
#             nargs['cell_aa_deg'] = _np.concatenate((
#                                   _np.array(_get_atom_spread(_load.XYZ.data)),
#                                   _np.ones(3) * 90.
#                                   ))

        if _load.mol_map is not None:
            return cls(
                    members=[(1, _s)
                             for _s in _load.XYZ._frame._split(
                             _load.mol_map)],
                    **nargs)
        else:
            return cls(members=[(1, _load.XYZ._frame)],
                       **nargs)

    def _cell_vec_aa(self, **kwargs):
        return _get_cell_vec(self.cell_aa_deg)

    def _cell_aa_deg(self):
        if hasattr(self, 'cell_aa_deg'):
            if self.cell_aa_deg is not None:
                return self.cell_aa_deg
        if hasattr(self, 'cell_vec_aa'):
            return _np.concatenate(
                _np.linalg.norm(self.cell_vec_aa, axis=-1),
                _np.array([
                    angle(self.cell_vec_aa[1], self.cell_vec_aa[2]),
                    angle(self.cell_vec_aa[0], self.cell_vec_aa[2]),
                    angle(self.cell_vec_aa[0], self.cell_vec_aa[1])
                    ]) * 180./_np.pi
                )

    def _volume_aa3(self):
        return _np.dot(self.cell_vec_aa[0],
                       _np.cross(self.cell_vec_aa[1], self.cell_vec_aa[2]))

    def _sync_class(self):
        '''Calculates intensive properties only'''
        self.n_members = len(self.members)
        self.mass_amu = sum([_n * sum(_m.masses_amu)
                             for _n, _m in self.member_set])
        self.n_atoms = sum([_n * _m.n_atoms for _n, _m in self.members])
        try:
            self.cell_aa_deg = self._cell_aa_deg()
        except AttributeError:
            pass
        if not hasattr(self, 'symmetry'):
            self.symmetry = _get_symmetry(self.cell_aa_deg)
    # def routine: check all xx attributes against _xx() methods

    def _clean_members(self):
        if self.n_members == 0:
            return None
        _eq = _np.zeros((self.n_members,) * 2)
        for _ii, (_i, _m) in enumerate(self.members):
            _eq[_ii, _ii:] = _np.array([bool(_m._is_equal(_n, noh=True)[0])
                                       for _j, _n in self.members[_ii:]])
        _N = self.n_members
        _M = self.n_members
        _ass = _np.zeros((_N)).astype(int)
        _sp = 0
        for _im in range(_N):
            if _ass[_im] == 0:
                _sp += 1
                _ass, _M = _assign_molecule(_ass, _sp, _N, _eq, _im, _M)
            if _M == 0:
                break
        _n, _m = _np.array(self.members).T
        self.member_set = [(_n[_ass == _i].sum(), _m[_ass == _i][0])
                           for _i in _np.unique(_ass)]
        self._sync_class()

    def __add__(self, other):
        if not isinstance(other, _BoxObject):
            raise TypeError('unsupported operand type(s) for +: \
\'%s\' and \'%s\'' % (type(self).__name__, type(other).__name__))
        new = _copy.deepcopy(self)
        if _np.allclose(self.cell_vec_aa, other.cell_vec_aa):
            new.members += other.members
        else:
            raise AttributeError('The two objects have different \
cell attributes!')
        new._sync_class()
        new._clean_members()
        return new
        # Later: choose largest cell param and lowest symmetry

    def __mul__(self, other):
        new = _copy.deepcopy(self)
        if isinstance(other, int):
            for _i in range(other-1):
                new += self
        # elif isinstance(other, _BoxObject):
        # ToDo: connect it to pow
        else:
            raise TypeError('unsupported operand type(s) for *: \
\'%s\' and \'%s\'' % (type(self).__name__, type(other).__name__))
        return new

    def __pow__(self, other):
        print('\nWARNING: Power in beta state. Proceed with care!\n')
        if not isinstance(other, int):
            raise TypeError('unsupported operand type(s) for *: \
\'%s\' and \'%s\'' % (type(self).__name__, type(other).__name__))
        new = _copy.deepcopy(self)
        new.members *= other
        new._cell_vec_aa = lambda: other ** (1/3) * self._cell_vec_aa()
        new.cell_vec_aa = new._cell_vec_aa()
        new.volume_aa3 = new._volume_aa3()
        new._sync_class()
        new._clean_members()
        return new

    # def __radd__(self, other):
    #     return self.__add__(other)

    # def __iadd__(self, other):
    #     self = self.__add__(other)
    #     return self

    # def __rmul__(self, other):
    #     return self.__mul__(other)

    # def __imul__(self, other):
    #     self = self.__mul__(other)
    #     return self

    # def __ipow__(self, other):
    #     self = self.__pow__(other)
    #     return self

    def _mol_map(self):
        _imol = 0
        mol_map = []
        for _m in self.member_set:
            for _i in range(_m[0]):
                mol_map += _m[1].n_atoms * [_imol]
                _imol += 1
        return _np.array(mol_map)

    def print_info(self):
        print(77 * '–')
        print('%-12s' % self.__class__.__name__)
        print(77 * '–')
        print('%-12s %s' % ('Periodic', self.pbc))
        print('%12d Members\n%12d Atoms\n%12.4f amu\n%12.4f aa3' %
              (self.n_members, self.n_atoms, self.mass_amu, self.volume_aa3))
        print(77 * '–')
        print('CELL ' + ' '.join(map('{:10.5f}'.format, self.cell_aa_deg)))
        print(77 * '-')
        print(' A   ' + ' '.join(map('{:10.5f}'.format, self.cell_vec_aa[0])))
        print(' B   ' + ' '.join(map('{:10.5f}'.format, self.cell_vec_aa[1])))
        print(' C   ' + ' '.join(map('{:10.5f}'.format, self.cell_vec_aa[2])))
        print(77 * '–')
        print('%45s %8s %12s' % ('File', 'No.', 'Molar Mass'))
        print(77 * '-')
        print('\n'.join(['%45s %8d %12.4f' %
                         (_m[1].fn, _m[0], sum(_m[1].masses_amu))
                         for _m in self.member_set]))
        print(77 * '–')

    def create(self, **kwargs):
        # most important class (must not be adapted within derived classes)
        # work in progress... # creates a system object (Supercell)
        pass

    def write(self, **kwargs):
        _SC = self.create(**kwargs)
        _SC.write('supercell.xyz')
#        _SC.write('supercell.pdb')


class MolecularCrystal(_BoxObject):
    def _sync_class(self):
        if _np.any(self.cell_aa_deg == 0.) or self.cell_aa_deg is None:
            raise ValueError('%s requires valid cell dimensions!'
                             % self.__class__.__name__)

        self.lattice = _get_symmetry(self.cell_aa_deg)
        _BoxObject._sync_class(self)

    # CPMD priority (monoclinic): (2, 0, 1)
    def propagate(self, frame, multiply=(1, 1, 1), priority=(0, 1, 2)):
        '''Convolute FRAME object with unitcell.'''
        multiply = _np.array(multiply)

        cart_vec_aa = _get_cell_vec(self.cell_aa_deg,
                                    n_fields=frame.n_fields,
                                    priority=priority)
        frame.axis_pointer = -2
        new = _copy.deepcopy(frame)
        for iz, z in enumerate(multiply):
            tmp = _copy.deepcopy(new)
            for iiz in range(z-1):
                tmp.data[:, :3] += cart_vec_aa[None, iz]
                new += tmp

        # ToDo: for modes (should be possible from FRAME object)
        # elif hasattr(data, 'eival_cgs'):
        #     cart_vec_aa = _get_cell_vec(self.cell_aa_deg,
        #                                 n_fields=3,
        #                                 priority=priority)
        #     new = _copy.deepcopy(data)
        #     for iz, z in enumerate(multiply):
        #         tmp = _copy.deepcopy(new)
        #         for iiz in range(z-1):
        #             tmp.pos_au[:, :] += cart_vec_aa[_np.newaxis, iz] *\
        #                                 constants.l_aa2au
        #             new += tmp
        #    new._sort()
        new.cell_aa_deg[:3] *= multiply

        return new

    def create(self, **kwargs):
        _SC = self.members[0][1]
        _SC.axis_pointer = -2
        _mol = 0
        mol_map = [_mol] * len(_SC.symbols)
        for _i in range(self.members[0][0]-1):
            _SC += self.members[0][1]
            _mol += 1
            mol_map += [_mol] * len(self.members[0][1].symbols)

        for _m in self.members[1:]:
            for _i in range(_m[0]):
                _SC += _m[1]
                _mol += 1
                mol_map += [_mol] * len(self.members[0][1].symbols)

        multiply = kwargs.get('multiply', (1, 1, 1))
        _uc_mol_map = mol_map
        for _repeat in range(1, _np.prod(multiply)):
            mol_map = _np.concatenate((mol_map, _uc_mol_map + _np.amax(mol_map)))

        return _Supercell(XYZ=self.propagate(_SC, **kwargs), mol_map=mol_map)


_solvents = {}


class Solution(_BoxObject):
    def __init__(self, *args, **kwargs):
        self.symmetry = 'orthorhombic'  # by definition
        self.solvent = kwargs.get("solvent")
        self.rho_g_cm3 = kwargs.get("rho_g_cm3", 1.0)
        self.solutes = kwargs.get("solutes", [])
        self.c_mol_L = kwargs.get("c_mol_L", [1.0])

        # ToDo: ask for max atoms, max volume or max counts of members etc.
        # to get simulation cell ready

        if self.solvent is None:
            print('\nERROR: You have to specify a solvent as coordinate file \
or select one from the library!')
            _sys.exit(1)
        if self.solvent in _solvents:
            self.solvent = _solvents[self.solvent]
        if 0. in self.c_mol_L:
            print('\nERROR: You have to specify non-zero values of \
concentration!')
            _sys.exit(1)

        # TMP vars
        _slt = [_XYZFrame(_s) for _s in self.solutes]
        _slv = _XYZFrame(self.solvent)

        # --- CALCULATE INTENSIVE PROPERTIES (per 1L)
        # solvent concentration (externalise?)
        _c_slv_mol_L = (1000 * self.rho_g_cm3 -
                        sum([_c * _o.masses_amu.sum()
                             for _c, _o in zip(self.c_mol_L, _slt)])
                        ) / _slv.masses_amu.sum()

        if _c_slv_mol_L <= _np.amax(self.c_mol_L):
            raise ValueError('The given solutes\' concentrations exceed \
density of %.2f!\n' % self.rho_g_cm3)

        _c_min_mol_L = _np.amin(self.c_mol_L)

        # --- CALCULATE EXTENSIVE PROPERTIES
        # Get counts of solvent relative to solutes
        # default: set lowest c to 1
        _n = (self.c_mol_L + [_c_slv_mol_L]) / _c_min_mol_L

        def _dev_warning(_d, _id):
            if _d > 0.01:
                print('\nWARNING: Member counts differ from input value by \
more than 1%%:\n  - %s\n' % _id)
        [_dev_warning(abs(round(_in) - _in) /
                      _in, ([self.solvent] + self.solutes)[_ii])
         for _ii, _in in enumerate(_n)]

        nargs = {_k: kwargs.get(_k) for _k in kwargs.keys()
                 if _k not in [
                        "solvent",
                        "rho_g_cm3",
                        "solutes",
                        "c_mol_L",
                        "member_set",
                        ]}

        _BoxObject.__init__(
                self,
                members_set=[(int(round(_in)), _is)
                             for _in, _is in zip(_n, _slt + [_slv])],
                **nargs)  # by definition solvent is the last member

        del _slt, _slv, _c_slv_mol_L, _n, _c_min_mol_L, _dev_warning

    @classmethod
    def read(cls, fn, **kwargs):
        _tmp = _BoxObject.read(fn, **kwargs)
        # this is unpythonic but used only temporary until __init__()
        # routine has been adapted for read
        _out = cls.__new__(cls)
        for _key in _tmp.__dict__:
            setattr(_out, _key, getattr(_tmp, _key))
        _out._sync_class()
        return _out

    def _cell_vec_aa(self, **kwargs):
        # what to do if cell_aa argument given here??
        # ==> check if total volume is the same;
        # if yes use the given cell values, otherwise raise Exception
        if self.symmetry == 'orthorhombic':  # should be cubic
            return _np.diag(3 * [(self.mass_amu /
                                 self.rho_g_cm3 /
                                 (constants.avog * 1E-24)) ** (1/3)])

    def _c_mol_L(self):
        return [_m[0] / (constants.avog * 1E-27) / self.volume_aa3
                for _m in self.members]

    def _rho_g_cm3(self):
        return self.mass_amu / (constants.avog * 1E-24) / self.volume_aa3

    def _sync_class(self):
        _BoxObject._sync_class(self)
        self.c_mol_L = self._c_mol_L()
        self.rho_g_cm3 = self._rho_g_cm3()

    def print_info(self):
        _BoxObject.print_info(self)
        print('%12.4f g / cm³' % self.rho_g_cm3)
        print('\n'.join(map('{:12.4f} mol / L'.format, self.c_mol_L)))

    def create(self, **kwargs):
        return self._fill_box(**kwargs)

    def _fill_box(self, **kwargs):
        '''requires packmol'''
        # calculate packmol box
        _box_aa = _np.concatenate((self.origin_aa, _np.dot(_np.ones((3)),
                                  self.cell_vec_aa)))
        # if periodic: add vacuum edges
        if self.pbc:
            _box_aa += _np.array([1., 1., 1., -1., -1., -1.])

        with open('packmol.inp', 'w') as f:

            f.write('tolerance 2.000' + 2*'\n')
            f.write('filetype xyz' + 2*'\n')
            f.write('output .simbox.xyz' + '\n')

            for _im, _m in enumerate(self.member_set):
                _fn = '.member-%03d.xyz' % _im
                _m[1].write(_fn)
                f.write('\n')
                f.write('structure %s' % _fn + '\n')
                f.write('  number %d' % _m[0] + '\n')
                f.write('  inside box ' + ' '.join(map('{:.3f}'.format,
                                                   _box_aa)) + '\n')
                f.write('end structure' + '\n')

        print("Calling packmol ... (see packmol.log)")
        _os.system("packmol < packmol.inp > packmol.log")
        print("Done.")
        # # this is a little awkward (quick and dirty)
        # read packmol output and centre residue
        if self.pbc:
            _load = _Supercell(".simbox.xyz",
                               cell_aa_deg=self._cell_aa_deg(),
                               mol_map=self._mol_map(),
                               center_residue=0)
        else:
            _load = _Supercell(".simbox.xyz",
                               cell_aa_deg=self._cell_aa_deg(),
                               mol_map=self._mol_map())

        self.mol_map = self._mol_map()
        if kwargs.get('sort') is not None:
            _load.sort_atoms()
        # if kwargs.get('write_PDB', True):
        #    _load.write("topology.pdb")

        # clean files
        # _os.remove(".tmp_packmol.inp")
        for _im, _m in enumerate(self.member_set):
            _os.remove(".member-%03d.xyz" % _im)
        _os.remove(".simbox.xyz")

        return _load
# class Mixture, MolecularCrystal, IonicCrystal, GasPhase(Mixture)
