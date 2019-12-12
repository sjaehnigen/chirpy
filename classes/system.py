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

from ..snippets import tracked_update as _tracked_update
from ..snippets import equal as _equal
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

    def __init__(self, *args, **kwargs):
        '''Manually given arguments overwrite file attributes'''
        # python3.8: use walrus
        fn_topo = kwargs.get('fn_topo')
        if fn_topo is not None:
            self._topo = _read_topology_file(fn_topo)
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

            # --- ITERATOR DEVELOP info: methods called here cannot access ITER
            #       --> shift: extract_mol, center_res etc. directly to ITER
            #       --> allow empty / bare system init
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
                    self.XYZ.center_position(
                            self.mol_c_aa[center_res],
                            self.cell_aa_deg
                            )  # **kwargs)
                    self.wrap_molecules()

            extract_mols = kwargs.get('extract_mols')
            if extract_mols is not None:
                # if python 3.8: use walrus
                self.extract_molecules(extract_mols)
            # ---
            # Consistency check
            if hasattr(self, '_topo'):
                for _k in self._topo:
                    # python3.8: use walrus
                    _v = self.XYZ.__dict__.get(_k, self.__dict__.get(_k))
                    if _k is not None:
                        if not _equal(_v, self._topo[_k]):
                            print(_v, self._topo[_k])
                            raise ValueError('Topology file does not represent'
                                             ' molecule in {}!'.format(_k)
                                             )

        except KeyError:
            _warnings.warn('Initialised empty SYSTEM object!')

    def read_fn(self, *args, **kwargs):
        self.XYZ = self._XYZ(*args, **kwargs)
        fmt = self.XYZ._fmt
        # ToDo: After repairing / cleaning up Modes implement transfer
        # (mol info not written in XYZ or Modes)
        # kwargs.get('extract_mols'
        if fmt == 'xvibs':  # re-reads file
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

        _split = self.XYZ._split(self.mol_map)
        _out = _split[mols[0]]

        for i in mols[1:]:
            _out += _split[i]

        self.XYZ = _out

    def install_molecular_origin_gauge(self, **kwargs):
        # iterator: keyword: do it anew or keep existing mol_map
        if self.mol_map is not None:
            _warnings.warn('Overwriting existing mol_map!')

        n_map = tuple(_define_molecules(self)-1)
        self.mol_map = n_map

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
        ind = self.XYZ.sort(**kwargs)
        if self.mol_map is not None:
            self.mol_map = list(_np.array(self.mol_map)[ind])

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

        return nargs

    def write(self, fn, **kwargs):
        '''Write entire XYZ content to file (frame or trajectory).'''

        nargs = self._parse_write_args(fn, **kwargs)
        self.XYZ.write(fn, **nargs)

    def write_frame(self, fn, **kwargs):
        '''Write current XYZ frame to file (frame or trajectory).'''

        nargs = self._parse_write_args(fn, **kwargs)
        self.XYZ._frame.write(fn, **nargs)


class Molecule(_SYSTEM):
    def _XYZ(self, *args, **kwargs):
        return XYZIterator(*args, **kwargs)


class Supercell(_SYSTEM):
    def _XYZ(self, *args, **kwargs):
        return XYZIterator(*args, **kwargs)
