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

from ..classes.trajectory import XYZIterator, VibrationalModes
from ..topology.dissection import define_molecules as _define_molecules
from ..topology.dissection import read_topology_file as _read_topology_file

# switch function instead of elif
# def f(x):
#    return {
#        'a': 1,
#        'b': 2
#    }.get(x, 9)    # 9 is default if x not found


class _SYSTEM():
    '''Parent class that parses and manages properties of a chemical system
       organised in attributed classes.'''

    def __init__(self, fn, **kwargs):
        '''Manually given arguments overwrite file attributes'''

        fmt = kwargs.get('fmt', fn.split('.')[-1])
        self.mol_map = kwargs.get("mol_map")
        self.XYZ = self._XYZ(fn, **kwargs)
        self.cell_aa_deg = self.XYZ.cell_aa_deg

        # ToDo: After repairing / cleaning up Modes implement transfer
        # (mol info not written in XYZ or Modes)
        # kwargs.get('extract_mols'
        if fmt == 'xvibs':  # re-reads file
            self.Modes = VibrationalModes(fn, **kwargs)

        if kwargs.get('fn_topo') is not None:
            self.install_molecular_origin_gauge(**kwargs)

        elif fmt == "pdb":
            if self.mol_map is None:  # re-reads file
                kwargs.update({'fn_topo': fn})
                self.install_molecular_origin_gauge(**kwargs)

        # --- ITERATOR DEVELOP info: methods called here cannot access ITER
        #       --> shift: extract_mol, center_res etc. directly to ITER
        # --- NEEDS WORKUP
        if not any([_a <= 0.0 for _a in self.cell_aa_deg[:3]]):
            if kwargs.get('wrap_mols', False):
                if self.mol_map is None:
                    self.install_molecular_origin_gauge()
                self.wrap_molecules()

            if kwargs.get('sort', False):
                # Oh no! Modes will not be sorted. --> Sort that out!
                self.sort_atoms()

            center_res = kwargs.get('center_residue')
            if center_res is not None:
                # if python 3.8: use walrus
                self.wrap_molecules()
                self.XYZ._center_position(
                        self.mol_c_aa[center_res],
                        self.cell_aa_deg
                        )  # **kwargs)
                self.wrap_molecules()

        extract_mols = kwargs.get('extract_mols')
        if extract_mols is not None:
            # if python 3.8: use walrus
            self.extract_molecules(extract_mols)
        # ---

    def wrap_molecules(self):
        if self.mol_map is None:
            raise AttributeError('Wrap molecules requires a topology '
                                 '(mol_map)!')

        self.mol_c_aa = self.XYZ._wrap_molecules(self.mol_map,
                                                 self.cell_aa_deg)

    def extract_molecules(self, mols):
        '''mols: list of molecular indices
           BETA; not for iterators (yet)'''
        if self.mol_map is None:
            raise AttributeError('Extract molecules requires a topology '
                                 '(mol_map)!')

        _split = self.XYZ._split(self.mol_map)
        _out = _split[mols[0]]

        for i in mols[1:]:
            _out += _split[i]

        self.XYZ = _out

    def install_molecular_origin_gauge(self, **kwargs):
        # iterator: keyword: do it anew or keep existing mol_map
        fn = kwargs.get('fn_topo')
        if self.mol_map is not None:
            _warnings.warn('Found topology file, overwriting given mol_map!',
                           RuntimeWarning)

        if fn is not None:
            n_map, symbols = _read_topology_file(fn)
            if symbols != self.XYZ.symbols:
                raise ValueError('Topology file does not represent molecule!')
            # Check agreement topology with self (unit cell, ...)?

        else:
            n_map = tuple(_define_molecules(self)-1)

        n_mols = max(n_map)+1
        self.mol_map = n_map
        self.n_mols = n_mols

        # (mol info not written in XYZ or Modes)
        # --- NEEDS WORKUP
        if hasattr(self, 'Modes'):
            pass
            # should not depend on Modes
            # idea: get xyz data beforehand and send it to _define_molecules
            # which also offers wrap and returns the molecular centres of mass
        # ---

    def sort_atoms(self, **kwargs):
        '''Sort atoms alphabetically (default)'''
        ind = self.XYZ._sort()
        if self.mol_map is not None:
            self.mol_map = self.mol_map[ind]

    def print_info(self):
        print(77 * '–')
        print('%-12s' % self.__class__.__name__.upper())
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

    def write(self, fn, **kwargs):
        '''Work in progress...'''
        fmt = kwargs.pop('fmt', fn.split('.')[-1])
        nargs = {}

        if fmt == 'pdb':
            if self.mol_map is None:
                _warnings.warn('Could not find mol_map.', UserWarning)
                self.mol_map = _np.zeros(self.XYZ.n_atoms).astype(int)
            nargs = {_s: getattr(self, _s)
                     for _s in ('mol_map', 'cell_aa_deg')}
        if fmt == 'xyz':
            nargs.update(kwargs)
        else:
            _warnings.warn("Direct output disabled for format %s" % fmt,
                           UserWarning)
            nargs.update(kwargs)
        self.XYZ.write(fn, fmt=fmt, **nargs)


class Molecule(_SYSTEM):
    def _XYZ(self, *args, **kwargs):
        return XYZIterator(*args, **kwargs)


class Supercell(_SYSTEM):
    def _XYZ(self, *args, **kwargs):
        return XYZIterator(*args, **kwargs)
