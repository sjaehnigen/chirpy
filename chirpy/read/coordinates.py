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
#  Copyright (c) 2010-2022, The ChirPy Developers.
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
# -------------------------------------------------------------------


import numpy as np
import warnings
import copy

from CifFile import ReadCif as _ReadCif
import fortranformat as ff

from .generators import _reader, _open, _container
from ..topology.mapping import detect_lattice, get_cell_vec
from ..constants import convert as _convert
from .. import config

if config.__os__ == 'Linux':
    from ..external.concurrent_iterator.process import Producer


# --- kernels

def _xyz(frame, convert=1, n_lines=1):
    '''Kernel for processing xyz frame.'''

    # --- frame never starts with blank line --> EOF
    #     (+treatment of blank lines at EOF)
    if (_first_line := next(frame).strip()) == '':
        warnings.warn('blank line invoked end of file',
                      config.ChirPyWarning,
                      stacklevel=2)
        raise StopIteration
    _atomnumber = int(_first_line)

    if n_lines != _atomnumber + 2:
        raise ValueError('XYZ file inconsistent')

    comment = next(frame).rstrip('\n').strip()
    _split = (_l.strip().split() for _l in frame)
    symbols, data = zip(*[(_l[0], _l[1:]) for _l in _split])

    if len(data) != n_lines - 2:
        raise ValueError('XYZ file broken or incomplete')

    return np.array(data).astype(float)*convert, symbols, comment


def _cpmd(frame, convert=1, n_lines=1, filetype='TRAJECTORY'):
    '''Kernel for processing cpmd frame.'''

    # --- generator needs one+ call next() to allow for StopIteration
    data = []
    # --- frame never starts with blank line --> EOF
    #     (+treatment of blank lines at EOF)
    if (_first_line := next(frame).strip()) == '':
        warnings.warn('blank line invoked end of file',
                      config.ChirPyWarning,
                      stacklevel=2)
        raise StopIteration
    data.append(_first_line.split())

    for _l in frame:
        _l = _l.strip().split()
        data.append(_l)

    if len(data) != n_lines:
        raise ValueError('CPMD file broken or incomplete')

    if 'GEOMETRY' in filetype:
        _data = np.array(data).astype(float)

    elif filetype in ['TRAJECTORY', 'MOMENTS']:
        _data = np.array(data).astype(float)[:, 1:]

    else:
        raise ValueError('Unknown CPMD filetype %s' % filetype)

    return _data * convert


def _free(frame, columns='iddd', convert=1, n_lines=1):
    '''Kernel for processing free format frame.
       Column support: i s m d
       i ... iterations/frames
       s ... symbols/numbers
       d ... data'''

    def _parse_columns(line):
        content = {}
        # --- ToDo: is this slowing down the generator?
        for _c, _l in zip(columns, line.strip().split()):
            if _c == 'd':
                if 'd' not in content:  # important for keeping column order
                    content['d'] = []
                content[_c].append(float(_l))  # FortranF?
            else:
                content[_c] = _l
        content['d'] = np.array(content['d']) * convert
        return tuple(content.values())

    # --- generator needs at least one call of next() to work properly
    data = []
    # --- frame never starts with blank line --> EOF
    #     (+treatment of blank lines at EOF)
    if (_first_line := next(frame).strip()) == '':
        warnings.warn('blank line invoked end of file',
                      config.ChirPyWarning,
                      stacklevel=2)
        raise StopIteration
    data.append(_parse_columns(_first_line))

    for _l in frame:
        data.append(_parse_columns(_l))

    if len(data) != n_lines:
        raise ValueError('File broken or incomplete')

    _return = tuple(zip(*data))

    return _return


def _arc(frame, convert=1, n_lines=1, cell_line=False):
    '''Kernel for processing arc frame.'''

    CELL = cell_line
    # --- frame never starts with blank line --> EOF
    #     (+treatment of blank lines at EOF)
    if (_first_line := next(frame).strip()) == '':
        warnings.warn('blank line invoked end of file',
                      config.ChirPyWarning,
                      stacklevel=2)
        raise StopIteration
    _head = _first_line.split()
    _atomnumber = int(_head[0])
    comment = ' '.join(_head[1:])
    if CELL:
        cell_aa_deg = list(map(float, next(frame).strip().split()))

    if n_lines != _atomnumber + 1 + CELL:
        raise ValueError('ARC file inconsistent')

    # --- FORTRAN conversion of numbers; we read single items, choosing broad
    #     range hence.
    _ff = ff.FortranRecordReader('(F160.16)')
    _split = (_l.strip().split() for _l in frame)

    # --- ToDo: these are not the atomic numbers but ids
    numbers, symbols, data, types, connectivity =\
        zip(*[(int(_l[0]),
              _l[1],
              [_ff.read(_i)[0] for _i in _l[2:5]],
              [int(_i) for _i in _l[5:6]],
              list(map(int, _l[6:])))
              for _l in _split])
    # --- flatten
    types = tuple([_it for _t in types for _it in _t])

    if len(data) != n_lines - 1 - CELL:
        raise ValueError('ARC file broken or incomplete')

    _return = np.array(data).astype(float)*convert, symbols, numbers, types,\
        connectivity, comment

    if CELL:
        _return += (cell_aa_deg,)
    return _return


def _pdb(frame, convert=1., n_lines=1):
    '''Kernel for processing PDB frame'''
    names, resns, resids, data, symbols, cell_aa_deg, title = \
        [], [], [], [], [], None, None

    def mk_int(s):
        return int(s) if s.strip() else 0

    # --- explict for loop for adpated handling StopIteration
    while True:
        try:
            line = next(frame)
        except StopIteration:
            if symbols == []:
                raise StopIteration
            break

        record = line[:6].strip()
        match record:
            case 'TITLE':
                title = line[10:80].rstrip('\n')
            case 'CRYST1':
                cell_aa_deg = np.array([
                    line[6:15],
                    line[15:24],
                    line[24:33],
                    line[33:40],
                    line[40:47],
                    line[47:54]
                    ]).astype(float)
                # data['space_group'] = line[55:66]
                # data['Z_value'    ] = int(line[66:70])
            case ('ATOM' | 'HETATM'):
                # atom_ser_nr.append(int(line[6:11]))
                names.append(line[12:16].strip())
                # alt_loc_ind.append(line[16])
                resns.append(line[17:21].strip())
                # NB: adding [20] to resn
                # Note: line[20] seems to be blank
                # chain_ind.append(line[21]) ??
                resids.append(mk_int(line[22:26]))  # residue sequence number
                # code_in_res.append(line[26])
                data.append(list(map(float, [
                    line[30:38], line[38:46], line[46:54]
                    ])))
                # occupancy.append(float(line[54:60]))
                # temp_fact.append(float(line[60:66]))
                # seg_id.append(line[72:76])
                _s = line[76:78].strip()
                if (_s := line[76:78].strip()) == '':
                    warnings.warn('invalid or missing element symbol column'
                                  'in PDB',
                                  config.ChirPyWarning, stacklevel=2)
                    symbols.append(names[-1][:2])
                else:
                    symbols.append(_s)

    # --- data length not unamibigously connected to n_lines in PDB
    # if len(data) != n_lines:
    #     raise ValueError('File broken or incomplete')

    return np.array(data), \
        tuple(names), \
        tuple(symbols), \
        tuple([list(_n) for _n in zip(resids, resns)]), \
        cell_aa_deg, \
        title


# --- iterators


def xyzIterator(FN, **kwargs):
    '''Iterator for xyzReader
       Usage: next() returns data, symbols, comments of
       current frame'''
    _kernel = _xyz

    with _open(FN, 'r', **kwargs) as _f:
        _nlines = int(_f.readline().strip()) + 2
        _comment = _f.readline().strip()

    if (units := kwargs.pop('units', 'default')) != 'default':
        kwargs['convert'] = _convert(units)

    elif 'CPMD' in _comment or 'GEOMETRY' in FN:
        warnings.warn('Assuming angstrom per a.u. as velocity unit for '
                      'CPMD-generated XYZ file',
                      config.ChirPyWarning,
                      stacklevel=2)
        kwargs['convert'] = _convert(3*[('length', 'aa')]
                                     + 3*[('velocity', 'aa')])

    if config.__os__ == 'Linux':
        return Producer(_reader(FN, _nlines, _kernel, **kwargs),
                        maxsize=20, chunksize=4)
    else:
        return _reader(FN, _nlines, _kernel, **kwargs)


def cpmdIterator(FN, **kwargs):
    '''Iterator for  cpmdReader
       Known types: GEOMETRY, TRAJECTORY, MOMENTS
       Usually expects additional metadata of the system
       through kwargs.'''
    _kernel = _cpmd
    symbols = kwargs.pop('symbols', None)
    if (filetype := kwargs.get('filetype')) is None:
        filetype = FN.split('/')[-1]
        # --- try guess
        if any([_f in filetype for _f in [
                            'TRAJSAVED',
                            'CENTERS'
                            ]]):
            filetype = 'TRAJECTORY'
        if any([_f in filetype for _f in [
                            'MOL',
                            ]]):
            filetype = 'MOMENTS'

        kwargs.update(dict(filetype=filetype))

    if symbols is None:
        _nlines = 1
    else:
        _nlines = len([_k for _k in symbols])  # type-independent

    # --- units
    if (units := kwargs.pop('units', 'default')) != 'default':
        kwargs['convert'] = _convert(units)
    elif kwargs['filetype'] == 'MOMENTS':
        kwargs['convert'] = _convert(3*[('length', 'au')] +
                                     6*[('velocity', 'au')])
    # elif kwargs['filetype'] in ['TRAJECTORY', 'GEOMETRY']:
    else:
        kwargs['convert'] = _convert(3*[('length', 'au')] +
                                     3*[('velocity', 'au')])

    if config.__os__ == 'Linux':
        return Producer(_reader(FN, _nlines, _kernel, **kwargs),
                        maxsize=20, chunksize=4)
    else:
        return _reader(FN, _nlines, _kernel, **kwargs)


def arcIterator(FN, **kwargs):
    '''Iterator for arcReader
       Usage: next() returns data, symbols, numbers, types, and connectivity
       of current frame'''
    _kernel = _arc

    with _open(FN, 'r', **kwargs) as _f:
        _nlines = int(_f.readline().strip().split()[0]) + 1
        if len(_f.readline().strip().split()) == 6:
            # --- primitive check if there is a cell line
            _nlines += 1
            kwargs.update({'cell_line': True})

    if (units := kwargs.pop('units', 'default')) != 'default':
        kwargs['convert'] = _convert(units)
    elif FN.split('.')[-1] == 'vel':
        kwargs['convert'] = _convert(3*[('velocity', 'aa_ps')])

    if config.__os__ == 'Linux':
        return Producer(_reader(FN, _nlines, _kernel, **kwargs),
                        maxsize=20, chunksize=4)
    else:
        return _reader(FN, _nlines, _kernel, **kwargs)


def freeIterator(FN, columns='iddd', nlines=None, units=1, **kwargs):
    '''Iterator for free data of the format:
         i(frame) [m s ... (additional columns)] x0 x1 x2 ... (coordinate)

       columns: custom format (one letter per column)
         i ... frame
         d ... data
         s ... symbol
         m ... molecule id
         z ... atomic number
         n ... name (NOT IMPLEMENTED)
         r ... residue index (NOT IMPLEMENTED)
         t ... type (NOT IMPLEMENTED)
       nlines: number of lines per frame
         if None auto-guessed if frame (\'i\') is in the column set
       units: conversion set with one item per data coloumn (\'d\')
       '''
    _kernel = _free
    kwargs['columns'] = columns
    if 'd' not in columns:
        raise ValueError('freeIterator excepts at least one data column')

    try:
        kwargs['convert'] = _convert(units)
    except KeyError:
        raise KeyError('freeIterator expects units argument to be set '
                       'with one item per data column')

    # --- auto guess nlines
    if (_nlines := nlines) is None:
        _nlines = 1
        try:
            _icol = columns.index('i')
            with _open(FN, 'r') as _f:
                _fr = _f.readline().strip().split()[_icol]
                try:
                    while _f.readline().strip().split()[_icol] == _fr:
                        _nlines += 1
                except IndexError:
                    pass
        except ValueError:
            # -- ToDo: add more options from other columns (m surtout)
            pass

    if config.__os__ == 'Linux':
        return Producer(_reader(FN, _nlines, _kernel, **kwargs),
                        maxsize=20, chunksize=4)
    else:
        return _reader(FN, _nlines, _kernel, **kwargs)


def pdbIterator(FN, **kwargs):
    '''Iterator for PDB files
       Usage: next() returns data, names, symbols, residues, cell_aa_deg, title
       '''
    _kernel = _pdb

    with open(FN) as _f:
        for _nlines, _line in enumerate(_f, 1):
            if 'END' in _line:
                break

    if config.__os__ == 'Linux':
        return Producer(_reader(FN, _nlines, _kernel, **kwargs),
                        maxsize=20, chunksize=4)
    else:
        return _reader(FN, _nlines, _kernel, **kwargs)


# --- containers


def _coordContainer(*args, iterator=xyzIterator, **kwargs):
    '''Iterate over multiple trajectory files. The files have to
       agree in the number of atoms and frames.
       The output has the form of the first given file.
       iterator ... file reader or list of readers (mixed format case)

       '''
    # --- works for all iterators with data at return position 0
    if isinstance(iterator, list):
        _iterators = iterator
    else:
        _iterators = [iterator] * len(args)

    # --- pop kwargs items before passing
    convert = 1.
    if (units := kwargs.pop('units', 'default')) != 'default':
        # --- do not send convert to readers as usual, convert here
        convert = _convert(units)
        kwargs['units'] = 1

    # --- deepcopy: create independent (!) versions of kwargs
    reader_a, fn_a, args_a, kwargs_a = zip(
         *[(_iter, _fn, (), copy.deepcopy(kwargs))
             for _iter, _fn in zip(_iterators, args)]
         )
    _dim = None
    for _frame in _container(reader_a, fn_a, args_a, kwargs_a):
        if _dim is None:
            _dim = np.array(_frame[0]).shape

        yield np.transpose(
                    np.array(_frame[0]), axes=(1, 0, 2)
                    ).reshape(_dim[1], _dim[0]*_dim[2]) * convert, \
            *[_fr[0] for _fr in _frame[1:]]


def xyzContainer(*args, **kwargs):
    '''Iterate over multiple XYZ trajectory files. The files have to
       agree in the number of atoms and frames'''
    return _coordContainer(*args, iterator=xyzIterator, **kwargs)


def arcContainer(*args, **kwargs):
    '''Iterate over multiple ARC trajectory files. The files have to
       agree in the number of atoms and frames'''
    return _coordContainer(*args, iterator=arcIterator, **kwargs)


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
    buf = list(zip(*arcIterator(FN, **kwargs)))

    data, symbols, numbers, types, connectivity, comments = buf[:6]

    _return = (np.array(data), symbols[0], numbers[0], types[0],
               connectivity[0], list(comments))

    try:  # --- cell
        _return += (np.array(buf[6][0]),)
    except IndexError:
        pass

    # --- ToDo: FUTURE: support of changes in type or connectivity (if Tinker
    #                   supports it); and cell
    return _return


def pdbReader(FN, **kwargs):
    '''Read complete PDB file at once using MDAnalysis.
       Returns data, names, symbols, res, cell_aa_deg, title
       of current frame.
       Does not support variable cell size, use iterator for this.

       https://www.mdanalysis.org/docs/documentation_pages/coordinates/PDB.html
       '''
    buf = list(zip(*pdbIterator(FN, **kwargs)))

    data, names, symbols, res, cell_aa_deg, title = buf[:6]

    return np.array(data), tuple(names[0]), tuple(symbols[0]), tuple(res[0]), \
        cell_aa_deg[0], list(title)


# ------ external readers


def cifReader(FN, fill_unit_cell=True):
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
        if _label is None:
            warnings.warn('could not find at least one item from '
                          f'attribute list: {_list}. Broken file?',
                          RuntimeWarning,
                          stacklevel=2)
        return _label

    _read = _ReadCif(FN)
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

    symbols = tuple(get_label(['_atom_site_type_symbol']))
    names = tuple(get_label(['_atom_site_label']))

    _space_group_label = get_label([
            '_space_group_crystal_system',
            '_symmetry_cell_setting'
            ]).lower()

    if _space_group_label is None:
        warnings.warn('No space group label found in file!',
                      config.ChirPyWarning,
                      stacklevel=2)

    elif detect_lattice(cell_aa_deg) != _space_group_label:
        warnings.warn('The given space group and cell parametres do not match!'
                      ' %s != %s' % (_space_group_label,
                                     detect_lattice(cell_aa_deg)),
                      config.ChirPyWarning,
                      stacklevel=2)

    _space_group_symop = get_label([
            '_space_group_symop_operation_xyz',
            '_symmetry_equiv_pos_as_xyz',
            ])
    if _space_group_symop is None:
        warnings.warn('No symmetry operations found in file!',
                      config.ChirPyWarning,
                      stacklevel=2)
    else:
        _x = _y = _z = []
        _symbols = _names = ()
        # -- apply given symmetry operations on asymmetric unit
        #    (assumes first operation to be the identity: 'x, y, z')
        for _io, op in enumerate(_space_group_symop):
            _op = [__op.strip() for __op in op.split(',')]
            _x = np.append(_x, eval(_op[0]), axis=0)
            _y = np.append(_y, eval(_op[1]), axis=0)
            _z = np.append(_z, eval(_op[2]), axis=0)
            _symbols += symbols
            _names += names
            if _io == 0 and not fill_unit_cell:
                break

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
