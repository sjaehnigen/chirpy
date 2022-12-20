# ----------------------------------------------------------------------
#
#  ChirPy
#
#    A python package for chirality, dynamics, and molecular vibrations.
#
#    https://github.com/sjaehnigen/chirpy
#
#
#  Copyright (c) 2020-2023, The ChirPy Developers.
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
# ----------------------------------------------------------------------

import numpy as _np
import warnings as _warnings

from .core import AttrDict
from ..snippets import tracked_extract_keys as _tracked_extract_keys
from ..snippets import equal as _equal
from ..config import ChirPyWarning as _ChirPyWarning
from .core import CORE as _CORE
from .trajectory import XYZ, VibrationalModes
from ..topology.dissection import define_molecules as _define_molecules
from ..topology.dissection import read_topology_file as _read_topology_file
from .. import constants
from ..visualise import print_info


class _SYSTEM(_CORE):
    '''Parent class that parses and manages properties of a chemical system
       organised in attributed classes.'''

    def __init__(self, *args, **kwargs):
        '''Manually given arguments overwrite file attributes'''
        self._topo = kwargs.get('fn_topo')
        if self._topo is not None:
            self._topo = _read_topology_file(self._topo)
            self._topo = _tracked_extract_keys(kwargs, msg='of topology file!',
                                               **self._topo)
            kwargs.update(self._topo)

        self.mol_map = kwargs.get("mol_map")
        self.weights = kwargs.get("weights")

        if len(args) != 0:
            self.read_fn(*args, **kwargs)
        else:
            if (fn := kwargs.get('fn')) is not None:
                self.read_fn(fn, **kwargs)
            else:
                self.XYZ = kwargs.pop('XYZ')
        try:
            self.cell_aa_deg = self.XYZ.cell_aa_deg
            self.symbols = self.XYZ.symbols
            # ToDo: Dict of atom kinds (with names)
            self.kinds = AttrDict({_s: constants.elements[_s]
                                   if _s in constants.elements
                                   else 'UNKNOWN'
                                   for _s in self.symbols})
            self.molecular_formula = AttrDict({_s: self.symbols.count(_s)
                                               for _s in sorted(self.symbols)})

            if kwargs.get('sort', False):
                self.sort_atoms()

            if kwargs.get('define_molecules', False):
                # --- overwrites molmap
                self.define_molecules()

            if kwargs.get('wrap_molecules', False):
                if self.mol_map is None:
                    self.define_molecules()
                self.wrap_molecules()

            if (center_mol := kwargs.get('center_molecule')) is not None:
                self.center_molecule(center_mol),

            if kwargs.get('clean_residues',
                          False) and self.mol_map is not None:
                self.clean_residues()
            # --- ToDo: necessary? Symbols are checked in _XYZ
            self._check_consistency()

        except KeyError:
            with _warnings.catch_warnings():
                _warnings.warn('Initialised void %s!'
                               % self.__class__.__name__,
                               _ChirPyWarning,
                               stacklevel=2)

    def _check_consistency(self):
        if self._topo is not None:
            for _k in self._topo:
                if _k is not None and 'topo' not in _k:
                    _v = self.XYZ.__dict__.get(_k, self.__dict__.get(_k))
                    if _v is not None and not _equal(_v, self._topo[_k]):
                        _warnings.warn('Topology file '
                                       f'{self._topo["fn_topo"]}'
                                       ' does not represent molecule '
                                       f'{self.XYZ._fn} in {_k}!',
                                       _ChirPyWarning,
                                       stacklevel=3)
                        print(self._topo[_k])
                        print(_v)

    def read_fn(self, *args, **kwargs):
        self.XYZ = self._XYZ(*args, **kwargs)
        fmt = self.XYZ._fmt

        if fmt in ['molden', 'mol', 'xvibs', 'orca', 'g09', 'gaussian']:
            try:
                self.Modes = VibrationalModes(*args, **kwargs)
            except NameError:
                pass

        elif fmt == "pdb":
            if self.mol_map is None:  # re-reads file
                _topo = _read_topology_file(*self.XYZ._fn)
                self.mol_map = _topo['mol_map']

    def center_molecule(self, index, **kwargs):
        weights = kwargs.pop('weights', self.weights)

        if self.mol_map is None:
            self.define_molecules()
        self.wrap_molecules()
        self.XYZ.center_coordinates(
                selection=[_is for _is, _i in enumerate(self.mol_map)
                           if _i == index],
                weights=weights,
                )
        self.wrap_molecules()

    def wrap_molecules(self, **kwargs):
        weights = kwargs.pop('weights', self.weights)
        if self.mol_map is None:
            raise AttributeError('Wrap molecules requires a topology '
                                 '(mol_map)!')

        self.XYZ.wrap_molecules(self.mol_map, weights=weights, **kwargs)

    def wrap(self, **kwargs):
        self.XYZ.wrap(**kwargs)

    def extract_molecules(self, mols):
        '''Split XYZ through topology and select molecule(s) according to given
           ids
        mols  ...  list of molecular indices
        '''
        if self.mol_map is None:
            raise AttributeError('Extract molecules requires a topology '
                                 '(mol_map)!')

        self.XYZ.split(self.mol_map, select=mols)
        self.mol_map = _np.array([_i for _i in self.mol_map if _i in mols])

        self.symbols = self.XYZ.symbols
        # self.names = self.XYZ.names

    def extract_atoms(self, atoms):
        '''Split XYZ through topology and select atoms according to given
           ids
        atoms  ...  list of atomic indices
        '''
        self.XYZ.split(_np.arange(len(self.symbols)), select=atoms)
        if self.mol_map is not None:
            self.mol_map = _np.array([_i
                                      for _i in self.mol_map if _i in atoms])

        self.symbols = self.XYZ.symbols
        # self.names = self.XYZ.names

    def define_molecules(self, silent=False):
        '''Create molecular map (mol_map) based on distance
           criteria'''
        if self.mol_map is not None and not silent:
            _warnings.warn('Overwriting existing mol_map!',
                           _ChirPyWarning, stacklevel=2)

        n_map = tuple(_define_molecules(self.XYZ.pos_aa,
                                        self.XYZ.symbols,
                                        cell_aa_deg=self.cell_aa_deg))
        self.mol_map = n_map
        self.clean_residues()

    def clean_residues(self):
        '''Update residue numbers in XYZ (but not names!).
           Modes not supported.'''

        if hasattr(self.XYZ, 'residues'):
            self.XYZ.residues = tuple([[_im+1, _resn]
                                       for _im, (_resid, _resn) in
                                       zip(self.mol_map, self.XYZ.residues)])

        else:
            self.XYZ.residues = tuple([[_im+1, 'MOL'] for _im in self.mol_map])

    def sort_atoms(self, *args):
        '''Sort atoms alphabetically (default)'''
        ind = self.XYZ.sort(*args)
        self.symbols = self.XYZ.symbols

        if hasattr(self, 'Modes'):
            self.Modes.sort(ind, *args)

        if self.mol_map is not None:
            self.mol_map = _np.array(self.mol_map)[ind].flatten().tolist()
            self.clean_residues()

        if self._topo is not None:
            self._topo['mol_map'] = _np.array(
                                 self._topo['mol_map'])[ind].flatten().tolist()
            self._topo['symbols'] = tuple(_np.array(
                                 self._topo['symbols'])[ind])  # .flatten())
            # --- symbols analogues (update residues via mol_map [see above])
            for key in ['names']:
                try:
                    self._topo[key] = tuple(
                            _np.array(self._topo[key])[ind].tolist()
                            )
                except AttributeError:
                    pass

    def print_info(self):
        # Todo: use self._print_info = [print_info.print_header]
        print_info.print_header(self)
        print('%12d Atoms\n%12s' %
              (self.XYZ.n_atoms, self.XYZ.symbols))
        if self.mol_map is not None:
            print(f'Molecular Map:\n{self.mol_map}')
        print_info.print_cell(self)
        print(77 * '–')

    def _parse_write_args(self, fn, **kwargs):
        '''Work in progress...'''
        nargs = {}
        fmt = kwargs.get('fmt', fn.split('.')[-1])
        nargs['fmt'] = fmt

        if fmt == 'pdb':
            self.clean_residues()
            nargs = {_s: getattr(self, _s)
                     for _s in ('mol_map', 'cell_aa_deg')}
        if fmt == 'xyz':
            nargs.update(kwargs)
        else:
            nargs.update(kwargs)

        return nargs

    def write(self, fn, **kwargs):
        '''Write entire XYZ/Modes content to file (frame or trajectory).'''

        nargs = self._parse_write_args(fn, **kwargs)
        if hasattr(self, 'Modes'):
            self.Modes.write(fn, **nargs)
        else:
            self.XYZ.write(fn, **nargs)

    def write_frame(self, fn, **kwargs):
        '''Write current XYZ frame to file (frame or trajectory).'''

        nargs = self._parse_write_args(fn, **kwargs)
        self.XYZ._frame.write(fn, **nargs)


class Molecule(_SYSTEM):
    def _XYZ(self, *args, **kwargs):
        return XYZ(*args, range=(0, 1, 1), **kwargs)


class Supercell(_SYSTEM):
    def _XYZ(self, *args, **kwargs):
        return XYZ(*args, **kwargs)
