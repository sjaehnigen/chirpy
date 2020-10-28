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


import numpy as np
import MDAnalysis as mda
from CifFile import ReadCif as _ReadCif
import warnings
from concurrent_iterator.process import Producer
import fortranformat as ff

from .generators import _reader, _open
from ..topology.mapping import detect_lattice, get_cell_vec

from ..physics import constants
# close_neighbours


# --- kernels

def _xyz(frame, **kwargs):
    '''Kernel for processing xyz frame.'''

    convert = kwargs.get('convert', 1.)
    n_lines = kwargs.get('n_lines')
    _atomnumber = int(next(frame).strip())

    if n_lines != _atomnumber + 2:
        raise ValueError('inconsistent XYZ file')

    comment = next(frame).rstrip('\n')
    _split = (_l.strip().split() for _l in frame)
    symbols, data = zip(*[(_l[0], _l[1:]) for _l in _split])

    if len(data) != n_lines - 2:
        raise ValueError('broken or incomplete file')

    return np.array(data).astype(float)*convert, symbols, comment


def _cpmd(frame, **kwargs):
    '''Kernel for processing cpmd frame.'''

    convert = kwargs.get('convert', 1.)
    n_lines = kwargs.get('n_lines')
    filetype = kwargs.get('filetype')
    # --- is this a python bug?
    # --- generator needs at least one call of next() to work properly
    data = []
    data.append(next(frame).strip().split())

    for _l in frame:
        _l = _l.strip().split()
        data.append(_l)

    if len(data) != n_lines:
        raise ValueError('Tried to read broken or incomplete file!')

    if 'GEOMETRY' in filetype:
        _data = np.array(data).astype(float)

    elif filetype in ['TRAJSAVED', 'TRAJECTORY', 'MOMENTS']:
        _data = np.array(data).astype(float)[:, 1:]

    else:
        raise ValueError('Unknown CPMD filetype %s' % filetype)

    # --- convert positions
    # _data[:, :3] *= constants.l_au2aa
    return _data*convert


def _arc(frame, **kwargs):
    '''Kernel for processing xyz frame.'''

    convert = kwargs.get('convert', 1.)
    n_lines = kwargs.get('n_lines')
    _head = next(frame).strip().split()
    _atomnumber = int(_head[0])
    comment = ' '.join(_head[1:])

    if n_lines != _atomnumber + 1:
        raise ValueError('inconsistent XYZ file')

    # --- FORTRAN conversion of numbers; we read single items, choosing broad
    #     range hence.
    _ff = ff.FortranRecordReader('(F160.16)')
    _split = (_l.strip().split() for _l in frame)

    numbers, symbols, data, types, connectivity =\
        zip(*[(int(_l[0]),
              _l[1],
              [_ff.read(_i)[0] for _i in _l[2:5]],
              [int(_i) for _i in _l[5:6]],
              list(map(int, _l[6:])))
              for _l in _split])
    # --- flatten
    types = tuple([_it for _t in types for _it in _t])

    if len(data) != n_lines - 1:
        raise ValueError('broken or incomplete file')

    return np.array(data).astype(float)*convert, symbols, numbers, types,\
        connectivity, comment


# --- unit parser for iterators

def _convert(units):
    if isinstance(units, list):
        convert = np.array([constants.get_conversion_factor(_i, _j)
                            for _i, _j in units])
    elif isinstance(units, tuple):
        convert = constants.get_conversion_factor(*units)
    else:
        raise ValueError('invalid units')
    return convert


# --- iterators

def xyzIterator(FN, **kwargs):
    '''Iterator for xyzReader
       Usage: next() returns data, symbols, comments of
       current frame'''
    _kernel = _xyz

    with _open(FN, 'r', **kwargs) as _f:
        _nlines = int(_f.readline().strip()) + 2
        _comment = _f.readline().strip()
    if 'CPMD' in _comment or 'GEOMETRY' in FN:
        warnings.warn('It seems as if you are reading an XYZ file generated '
                      'by CPMD. Check velocity units!', stacklevel=2)
        if 'units' not in kwargs:
            kwargs['units'] = 3*[('length', 'aa')]+3*[('velocity', 'aa')]

    if (units := kwargs.pop('units', 'default')) != 'default':
        kwargs['convert'] = _convert(units)

    return Producer(_reader(FN, _nlines, _kernel, **kwargs),
                    maxsize=20, chunksize=4)


def cpmdIterator(FN, **kwargs):
    '''Iterator for  cpmdReader
       Known types: GEOMETRY, TRAJECTORY, MOMENTS
       Usually expects additional metadata of the system
       through kwargs.'''
    _kernel = _cpmd
    symbols = kwargs.pop('symbols', None)
    if 'filetype' not in kwargs:
        # --- try guess
        if (_F := FN.split('/')[-1]) == 'MOL':
            kwargs['filetype'] = 'MOMENTS'
        else:
            kwargs['filetype'] = _F

    if symbols is None:
        _nlines = 1
    else:
        _nlines = len([_k for _k in symbols])  # type-independent

    # --- units
    if (units := kwargs.pop('units', 'default')) != 'default':
        kwargs['convert'] = _convert(units)
    elif kwargs['filetype'] in ['TRAJECTORY', 'GEOMETRY']:
        kwargs['convert'] = _convert(3*[('length', 'au')] +
                                     3*[('velocity', 'au')])
    elif kwargs['filetype'] == 'MOMENTS':
        kwargs['convert'] = _convert(3*[('length', 'au')] +
                                     6*[('velocity', 'au')])

    return Producer(_reader(FN, _nlines, _kernel, **kwargs),
                    maxsize=20, chunksize=4)


def arcIterator(FN, **kwargs):
    '''Iterator for arcReader
       Usage: next() returns data, symbols, numbers, types, and connectivity
       of current frame'''
    _kernel = _arc

    with _open(FN, 'r', **kwargs) as _f:
        _nlines = int(_f.readline().strip().split()[0]) + 1

    if (units := kwargs.pop('units', 'default')) != 'default':
        kwargs['convert'] = _convert(units)

    return Producer(_reader(FN, _nlines, _kernel, **kwargs),
                    maxsize=20, chunksize=4)


# --- complete readers

def xyzReader(FN, **kwargs):
    '''Read complete XYZ file at once.
       Returns data, symbols, comments of current frame'''
    data, symbols, comments = zip(*xyzIterator(FN, **kwargs))
    return np.array(data), symbols[0], list(comments)


def arcReader(FN, **kwargs):
    '''Read complete ARC file at once.
       Returns data, symbols, numbers, types, and connectivity
       of current frame'''
    data, symbols, numbers, types, connectivity, comments =\
        zip(*arcIterator(FN, **kwargs))

    # --- FUTURE: support of changes in type or connectivity (if Tinker
    # supports it)
    return np.array(data), symbols[0], numbers[0], types[0], connectivity[0],\
        list(comments)


# ------ external readers

def pdbIterator(FN):
    '''Iterator for pdbReader relying on MDAnalysis
       Usage: next() returns data, names, symbols, res,
       cell_aa_deg, title of current frame.

       https://www.mdanalysis.org/docs/documentation_pages/coordinates/PDB.html
       '''
    with warnings.catch_warnings():  # ignroe MDAnalysis warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        u = mda.Universe(FN)

    for ts in u.trajectory:
        # only take what is needed in ChirPy
        data = u.coord.positions
        resns = u.atoms.resnames
        resids = u.atoms.resids
        names = tuple(u.atoms.names)
        symbols = tuple(u.atoms.types)
        cell_aa_deg = u.dimensions
        title = u.trajectory.title
        if np.prod(cell_aa_deg) == 0.0:
            warnings.warn('no or invalid cell specified in pdb file',
                          RuntimeWarning)
            cell_aa_deg = None
        if len(title) == 0:
            title = None
        else:
            title = title[0]

        yield data, names, symbols, tuple([list(_n) for _n in zip(
                                             resids,
                                             resns)]), cell_aa_deg, title


def pdbReader(FN, **kwargs):
    '''Read complete PDB file at once using MDAnalysis.
       Returns data, names, symbols, res, cell_aa_deg, title
       of current frame.
       Does not support variable cell size, use iterator for this.

       https://www.mdanalysis.org/docs/documentation_pages/coordinates/PDB.html
       '''

    data, names, symbols, res, cell_aa_deg, title = \
        tuple([_b for _b in zip(*pdbIterator(FN, **kwargs))])

    return np.array(data), tuple(names[0]), tuple(symbols[0]), tuple(res[0]), \
        cell_aa_deg[0], list(title)


def cifReader(fn):
    '''Read CIF file and return a filled unit cell.
       '''
    def _measurement2float(number):
        def _convert(st):
            return float(st.replace('(', '').replace(')', ''))
        if isinstance(number, str):
            return _convert(number)
        elif isinstance(number, list):
            return [_convert(_st) for _st in number]

    def get_label(_list):
        _label = None
        for _l in _list:
            _label = _load.get(_l, _label)
        return _label

    _read = _ReadCif(fn)
    title = _read.keys()[0]
    _load = _read[title]
    cell_aa_deg = np.array([_measurement2float(_load[_k]) for _k in [
                                                          '_cell_length_a',
                                                          '_cell_length_b',
                                                          '_cell_length_c',
                                                          '_cell_angle_alpha',
                                                          '_cell_angle_beta',
                                                          '_cell_angle_gamma'
                                                          ]])
#     data = np.array([_measurement2float(_load[_k]) for _k in [
#                                                           '_atom_site_fract_x',
#                                                           '_atom_site_fract_y',
#                                                           '_atom_site_fract_z'
#                                                           ]]).T
    x = np.array(_measurement2float(_load['_atom_site_fract_x']))
    y = np.array(_measurement2float(_load['_atom_site_fract_y']))
    z = np.array(_measurement2float(_load['_atom_site_fract_z']))

    if False:
        print(f'Asymmetric unit: {len(list(zip(x, y, z)))} atoms')

    symbols = tuple(_load['_atom_site_type_symbol'])
    names = tuple(_load['_atom_site_label'])

    _space_group_label = get_label([
            '_space_group_crystal_system',
            '_symmetry_cell_setting'
            ])

    if _space_group_label is None:
        warnings.warn('No space group label found in file!', stacklevel=2)

    elif detect_lattice(cell_aa_deg) != _space_group_label:
        warnings.warn('The given space group and cell parametres do not match!'
                      ' %s != %s' % (_space_group_label,
                                     detect_lattice(cell_aa_deg)),
                      RuntimeWarning,
                      stacklevel=2)

    _space_group_symop = get_label([
            '_space_group_symop_operation_xyz',
            '_symmetry_equiv_pos_as_xyz',
            ])
    if _space_group_symop is None:
        warnings.warn('No symmetry operations found in file!', stacklevel=2)
    else:
        _x = _y = _z = []
        _symbols = _names = ()
        # -- apply given symmetry operations on asymmetric unit
        #    (assumes first operation to be the identity: 'x, y, z')
        for op in _space_group_symop:
            _op = [__op.strip() for __op in op.split(',')]
            _x = np.append(_x, eval(_op[0]), axis=0)
            _y = np.append(_y, eval(_op[1]), axis=0)
            _z = np.append(_z, eval(_op[2]), axis=0)
            _symbols += symbols
            _names += names

    data = np.array([_x, _y, _z]).T

    # -- change to cell vector base
    cell_vec_aa = get_cell_vec(cell_aa_deg)
    data = np.tensordot(data, cell_vec_aa, axes=1)

    # --- clean data (atom duplicates), a little clumsy
    #     NOT REQUIRED FOR INTEGER FILES!
    # ind = np.array(sorted([
    #        _j[0]
    #        for _i in close_neighbours(data, cell=cell_aa_deg, crit=0.0)
    #        for _j in _i[1]
    #        ]))
    # data = np.delete(data, ind, axis=0)

    # --- add frames dimension (no support of cif trajectories)
    return np.array([data]), _names, _symbols, cell_aa_deg, [title]