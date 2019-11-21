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

from ..classes.crystal import UnitCell as _UnitCell
from ..classes.trajectory import XYZFrame, XYZTrajectory, XYZIterator, \
        VibrationalModes
from ..physics import constants
from ..physics.constants import masses_amu, valence_charges
from ..topology.dissection import define_molecules as _define_molecules
from ..topology.dissection import read_topology_file as _read_topology_file
from ..topology.mapping import dec as _dec

# switch function instead of elif
# def f(x):
#    return {
#        'a': 1,
#        'b': 2
#    }.get(x, 9)    # 9 is default if x not found

# ToDo: Repair UnitCell, get rid of XYZData_UnitCell attribute
#       Outsource Topology for mol_map


class _SYSTEM():
    def __init__(self, fn, **kwargs):
        '''Manually given arguments overwrite file attributes'''

        if int(_np.version.version.split('.')[1]) < 14:
            raise EnvironmentError('Numpy version has to be >= 1.14.0! '
                                   'You are using %s.' % _np.version.version)

        fmt = kwargs.get('fmt', fn.split('.')[-1])
        self.mol_map = kwargs.get("mol_map")
        self.XYZData = self._XYZ(fn, **kwargs)
        self.cell_aa_deg = self.XYZData.cell_aa_deg

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
            # UnitCell is DEPRECATED
            # NEW: _UnitCell as generator
            try:
                self._UnitCell = _UnitCell(self.cell_aa_deg)
                if kwargs.get('cell_multiply') is not None:
                    # if python 3.8: use walrus
                    cell_multiply = kwargs.get('cell_multiply')
                    # priority from CPMD (monoclinic): (2, 0, 1)
                    cell_priority = kwargs.get('cell_priority', (2, 0, 1))
                    self.XYZData_UnitCell = _copy.deepcopy(self.XYZData)
                    self.XYZData = self._UnitCell.propagate(
                                        self.XYZData,
                                        cell_multiply,
                                        priority=cell_priority
                                        )
                    # ---
                    # TMP cell and mol_map is not replicated
                    self.mol_map *= int(_np.prod(cell_multiply))
                    self.cell_aa_deg[:3] *= _np.array(cell_multiply)
                    # TMP reorder: fix it nicely
                    # _cp = list(cell_priority)
                    # self.cell_aa_deg[:3] = self.cell_aa_deg[:3][_cp]
                    # self.cell_aa_deg[3:] = self.cell_aa_deg[3:][_cp]
                    # ---

                    if hasattr(self, 'Modes'):
                        self.Modes_UnitCell = _copy.deepcopy(self.Modes)
                        self.Modes = self._UnitCell.propagate(
                                          self.Modes,
                                          cell_multiply,
                                          priority=cell_priority
                                          )
            except TypeError:
                pass

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
                self.XYZData._center_position(
                        self.mol_c_aa[center_res],
                        self.cell_aa_deg
                        )  # **kwargs)
                self.wrap_molecules()
            # ---

        extract_mols = kwargs.get('extract_mols')
        if extract_mols is not None:
            # if python 3.8: use walrus
            self.extract_molecules(extract_mols)

    def wrap_molecules(self):
        if self.mol_map is None:
            raise AttributeError('Wrap molecules requires a topology '
                                 '(mol_map)!')

        self.mol_c_aa = self.XYZData._wrap_molecules(self.mol_map,
                                                     self.cell_aa_deg)

    def extract_molecules(self, mols):
        '''mols: list of molecular indices
           BETA; not for iterators (yet)'''
        if self.mol_map is None:
            raise AttributeError('Extract molecules requires a topology '
                                 '(mol_map)!')

        _split = self.XYZData._split(self.mol_map)
        _out = _split[mols[0]]

        for i in mols[1:]:
            _out += _split[i]

        self.XYZData = _out
        # Modes, UnitCell?

    def install_molecular_origin_gauge(self, **kwargs):
        fn = kwargs.get('fn_topo')
        if self.mol_map is not None:
            _warnings.warn('Found topology file, overwriting given mol_map!',
                           RuntimeWarning)

        if fn is not None:
            n_map, symbols = _read_topology_file(fn)
            if symbols != self.XYZData.symbols:
                raise ValueError('Topology file does not represent molecule!')
            # Check agreement topology with self (unit cell, ...)?

        else:
            n_map = tuple(_define_molecules(self)-1)

        n_mols = max(n_map)+1
        self.mol_map = n_map
        self.n_mols = n_mols

        # --- NEEDS WORKUP
        if hasattr(self, 'Modes'):
            ZV, M = zip(*[(valence_charges[s], masses_amu[s])
                          for s in self.XYZData.symbols])
            ZV, M = _dec(ZV, n_map), _dec(M, n_map)
            # Modes
            pos_au = _dec(self.Modes.pos_au, n_map)
            # wrap molecules (in order to get "correct" com),
            # reference: first atom 0 of mol
            if hasattr(self, '_UnitCell'):
                abc = self._UnitCell.abc
                pos_au = [p-_np.around((p-p[0, None, :]) /
                                       (abc*constants.l_aa2au)
                                       ) * abc*constants.l_aa2au
                          for p in pos_au]
                # !!! writing into another object (evil) !!!
                self.Modes.cell_au = abc*constants.l_aa2au
            # calculate molecular centers of mass
            self.Modes.mol_com_au = _np.array([_np.sum(pos_au[i_mol] *
                                               M[i_mol][:, None], axis=0) /
                                               M[i_mol].sum()
                                               for i_mol in range(n_mols)])
            self.Modes.mol_com_au = _np.remainder(self.Modes.mol_com_au,
                                                  self.Modes.cell_au)
            self.Modes.mol_map = n_map
            self.Modes.n_mols = n_mols
        # ---

    def sort_atoms(self, **kwargs):
        '''Sort atoms alphabetically (default)'''
        ind = self.XYZData._sort()
        if self.mol_map is not None:
            self.mol_map = self.mol_map[ind]

    def write(self, fn, **kwargs):
        '''Work in progress...'''
        fmt = kwargs.get('fmt', fn.split('.')[-1])
        nargs = {}
        if fmt == 'pdb':
            if self.mol_map is None:
                _warnings.warn('Could not find mol_map.', UserWarning)
                self.mol_map = _np.zeros(self.XYZData.n_atoms).astype(int)
            nargs = {_s: getattr(self, _s)
                     for _s in ('mol_map', 'cell_aa_deg')}
        else:
            raise NotImplementedError("System object supports only PDB output"
                                      "for now (use _XYZ attribute instead)")
        self.XYZData.write(fn, fmt=fmt, **nargs)


class Supercell(_SYSTEM):
    def _XYZ(self, *args, **kwargs):
        return XYZTrajectory(*args, **kwargs)


class Molecule(_SYSTEM):
    def _XYZ(self, *args, **kwargs):
        return XYZTrajectory(*args, **kwargs)


class Supercell_ITER(_SYSTEM):
    def _XYZ(self, *args, **kwargs):
        return XYZIterator(*args, **kwargs)


class Snapshot(_SYSTEM):
    def _XYZ(self, *args, **kwargs):
        return XYZFrame(*args, **kwargs)
