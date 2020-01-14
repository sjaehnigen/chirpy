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

import numpy as _np
import warnings as _warnings

from .core import _CORE
from .trajectory import XYZ, XYZFrame, VibrationalModes
from ..snippets import tracked_update as _tracked_update
from ..snippets import equal as _equal
from ..topology.dissection import define_molecules as _define_molecules
from ..topology.dissection import read_topology_file as _read_topology_file


class _SYSTEM(_CORE):
    '''Parent class that parses and manages properties of a chemical system
       organised in attributed classes.'''

    def __init__(self, *args, **kwargs):
        '''Manually given arguments overwrite file attributes'''
        # python3.8: use walrus
        self._topo = kwargs.get('fn_topo')
        if self._topo is not None:
            # python 3.8: use walrus
            self._topo = _read_topology_file(self._topo)
            _tracked_update(kwargs, self._topo)

        self.mol_map = kwargs.get("mol_map")

        try:
            if len(args) != 0:
                self.read_fn(*args, **kwargs)
            else:
                # python3.8: use walrus
                fn = kwargs.get('fn')
                if fn is not None:
                    self.read_fn(fn, **kwargs)
                else:
                    self.XYZ = kwargs.pop('XYZ')
            self.cell_aa_deg = self.XYZ.cell_aa_deg

            if kwargs.get('sort', False):
                self.sort_atoms()

            if kwargs.get('wrap_mols', False):
                if self.mol_map is None:
                    self.install_molecular_origin_gauge()
                self.wrap_molecules()

            center_mol = kwargs.get('center_molecule')
            if center_mol is not None:
                # if python 3.8: use walrus
                self.wrap_molecules()
                self.XYZ.center_coordinates(
                        [_is for _is, _i in enumerate(self.mol_map)
                         if _i == center_mol],
                        use_com=True
                        )  # **kwargs)
                self.wrap_molecules()

            # Consistency check
            if self._topo is not None:
                for _k in self._topo:
                    # python3.8: use walrus
                    _v = self.XYZ.__dict__.get(_k, self.__dict__.get(_k))
                    if _k is not None:
                        if not _equal(_v, self._topo[_k]):
                            print(_v, self._topo[_k])
                            raise ValueError('Topology file does not represent'
                                             ' molecule in {}!'.format(_k)
                                             )

            # extract_mols = kwargs.get('extract_molecules')
            # if extract_mols is not None:
            #     # if python 3.8: use walrus
            #     self.extract_molecules(extract_mols)

        except KeyError:
            with _warnings.catch_warnings():
                _warnings.warn('Initialised void %s!'
                               % self.__class__.__name__,
                               stacklevel=2)

    def read_fn(self, *args, **kwargs):
        self.XYZ = self._XYZ(*args, **kwargs)
        fmt = self.XYZ._fmt

        if fmt in ['xvibs', 'orca']:  # re-reads file
            self.Modes = VibrationalModes(*args, **kwargs)

        elif fmt == "pdb":
            if self.mol_map is None:  # re-reads file
                self._topo = _read_topology_file(self.XYZ._fn)
                self.mol_map = self._topo['mol_map']

    def wrap_molecules(self):
        if self.mol_map is None:
            raise AttributeError('Wrap molecules requires a topology '
                                 '(mol_map)!')

        self.mol_c_aa = self.XYZ.wrap_molecules(self.mol_map)

    def wrap_atoms(self):
        self.XYZ.wrap_atoms()

    def extract_molecules(self, mols):
        '''mols: list of molecular indices
           BETA; not for iterators (yet)'''
        if self.mol_map is None:
            raise AttributeError('Extract molecules requires a topology '
                                 '(mol_map)!')

        self.XYZ.split(self.mol_map, select=mols)

    def install_molecular_origin_gauge(self, **kwargs):
        # iterator: keyword: do it anew or keep existing mol_map
        if self.mol_map is not None:
            _warnings.warn('Overwriting existing mol_map!', stacklevel=2)

        n_map = tuple(_define_molecules(self.XYZ.pos_aa,
                                        self.XYZ.symbols,
                                        cell_aa_deg=self.cell_aa_deg)-1)
        self.mol_map = n_map

    def sort_atoms(self, **kwargs):
        '''Sort atoms alphabetically (default)'''
        ind = self.XYZ.sort(**kwargs)
        if hasattr(self, 'Modes'):
            self.Modes.sort(ind, **kwargs)
        if self.mol_map is not None:
            self.mol_map = list(*map(list, _np.array(self.mol_map)[ind]))
        if self._topo is not None:
            self._topo['mol_map'] = list(*map(list,
                                         _np.array(self._topo['mol_map'])[ind]
                                              ))
            self._topo['symbols'] = tuple(*map(tuple,
                                          _np.array(self._topo['symbols'])[ind]
                                               ))

    def print_info(self):
        print(77 * '–')
        print('%-12s' % self.__class__.__name__)
        print(77 * '–')
        # print('%-12s %s' % ('Periodic', self.pbc))
        # print('%12d Members\n%12d Atoms\n%12.4f amu\n%12.4f aa3' %
        #       (self.n_members, self.n_atoms, self.mass_amu, self.volume_aa3))
        # print(77 * '–')
        # print('CELL ' + ' '.join(map('{:10.5f}'.format, self.cell_aa_deg)))
        # print(77 * '-')
        # print(' A   '+ ' '.join(map('{:10.5f}'.format, self.cell_vec_aa[0])))
        # print(' B   '+ ' '.join(map('{:10.5f}'.format, self.cell_vec_aa[1])))
        # print(' C   '+ ' '.join(map('{:10.5f}'.format, self.cell_vec_aa[2])))
        # print(77 * '–')
        # print('%45s %8s %12s' % ('File', 'No.', 'Molar Mass'))
        # print(77 * '-')
        # print('\n'.join(['%45s %8d %12.4f' %
        #                  (_m[1].fn, _m[0], _m[1].masses_amu.sum())
        #                  for _m in self.members]))
        print(77 * '–')

    def _parse_write_args(self, fn, **kwargs):
        '''Work in progress...'''
        nargs = {}
        fmt = kwargs.get('fmt', fn.split('.')[-1])
        nargs['fmt'] = fmt

        if fmt == 'pdb':
            if self.mol_map is None:
                _warnings.warn('Could not find mol_map.', stacklevel=2)
                self.mol_map = _np.zeros(self.XYZ.n_atoms).astype(int)
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
        return XYZFrame(*args, **kwargs)


class Supercell(_SYSTEM):
    def _XYZ(self, *args, **kwargs):
        return XYZ(*args, **kwargs)
