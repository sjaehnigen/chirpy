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


from itertools import zip_longest
import warnings

from ..constants import convert as _convert
from .. import config


def _write_xyz_frame(filename, data, symbols, comment,
                     selection=None, append=False):
    """WriteFrame(filename, data, symbols, comment, append=False)
    Input:
        1. filename: File to read
        2. data: np.array of shape (#atoms, #fields/atom)
        3. symbols: tuple of atom symbols (contains strings)
        4. comment line (string)
        5. selection: list of integers or None
        6. append: Append to file (optional, default = False)
    Output: None"""

    format = '  %s'
    # print(data.shape, data.shape[1])
    for field in range(data.shape[1]):
        # format += '    %16.12f'
        format += '      %14.10f'  # cp2k format ...
    format += '\n'
    if selection is None:
        n_atoms = len(symbols)
        _range = range(n_atoms)
    else:
        n_atoms = len(selection)
        _range = selection
    obuffer = '% 8d\n' % n_atoms
    # if comment[-1] != '\n':
    #     comment += '\n'
    obuffer += comment.rstrip('\n') + '\n'

    for i in _range:
        tmp = ['{0: <2}'.format(symbols[i])] + [c for c in data[i]]
        obuffer += format % tuple(tmp)
    fmt = 'w'
    if append:
        fmt = 'a'
    with open(filename, fmt) as f:
        f.write(obuffer)


def xyzWriter(fn, data, symbols,
              comments=None, units='default', selection=None, append=False):
    """WriteXYZFile(filename, data, symbols, comments, append=False)
       Input:
        1. fn: File to write
        2. data: np.array of shape ([#frames,] #atoms, #fields/atom)
        3. symbols: tuple of atom symbols (contains strings)
        4. list of comment lines (contains strings)
        5. append: Append to file (optional, default = False)

        Output: None"""

    convert = _convert(units)
    if len(data.shape) == 2:
        # ---frame
        if comments is None:
            comments = ''
        _write_xyz_frame(fn, data/convert, symbols,
                         comments, selection=selection, append=append)

    elif len(data.shape) == 3:
        # --- trajectory
        n_frames = len(data)
        for fr in range(n_frames):
            if comments is None:
                comment = ''
            else:
                comment = comments[fr]
            _write_xyz_frame(fn, data[fr]/convert, symbols,
                             comment, selection=selection,
                             append=append or fr != 0)

    else:
        raise AttributeError('Wrong data shape!', data.shape)


def _write_arc_frame(fn, data, symbols, numbers,
                     types=[], connectivity=[],
                     comment=None, selection=None, append=False):
    """
    Input:
        1. fn: File to write to
        2. data: np.array of shape (#atoms, #fields/atom)
        3. symbols: tuple of atom symbols (contains strings)
        4. numbers
        5. types
        6. connectivity
        7. append: Append to file (optional, default = False)

    Return: None"""
    # --- kwargs for additional header lines (crystal etc.)

    # --- header
    if selection is None:
        n_atoms = len(symbols)
        _range = list(range(n_atoms))
    else:
        n_atoms = len(selection)
        _range = selection

    obuffer = '{:>6d}'.format(n_atoms)
    if comment is not None:
        obuffer += f'  {comment}'
    obuffer += '\n'

    # --- corpus
    # --- important to keep order
    _iterator = list(zip_longest(numbers, symbols, data,
                                 types, connectivity,
                                 fillvalue=None))
    for _i in _range:
        _n, _s, _d, _t, _c = _iterator[_i]
        _line = '{:>6d}  {:3s}' + 3 * '{:>12.6f}'
        obuffer += _line.format(_n, _s, *_d)
        if _t is not None:
            obuffer += '{:>6d}'.format(_t)
        if _c is not None:
            obuffer += (len(_c)*'{:>6d}').format(*_c)
        obuffer += '\n'

    # --- type
    fmt = 'w'
    if append:
        fmt = 'a'
    with open(fn, fmt) as f:
        f.write(obuffer)


def arcWriter(fn, data, symbols, types=[], connectivity=[],
              comments=None, units='default', selection=None,
              append=False):
    """
       Input:
        1. fn: File to write to
        2. data: np.array of shape (#atoms, #fields/atom)
        3. symbols: list of atom symbols (contains strings)
        4. numbers
        5. types
        6. connectivity
        7. append: Append to file (optional, default = False)

        Return: None"""
    convert = _convert(units)

    numbers = list(range(1, len(symbols)+1))
    if len(data.shape) == 2:
        # ---frame
        _write_arc_frame(fn, data/convert, symbols, numbers, types,
                         connectivity,
                         comment=comments, selection=selection,
                         append=append)

    elif len(data.shape) == 3:
        # --- trajectory
        n_frames = len(data)
        for fr in range(n_frames):
            _write_arc_frame(fn, data[fr]/convert, symbols, numbers, types,
                             connectivity, comment=comments[fr],
                             selection=selection,
                             append=append or fr != 0)

    else:
        raise AttributeError('Wrong data shape!', data.shape)


def _write_pdb_frame(fn, data, names, symbols, residues, box, title,
                     selection=None, append=False):
    """WritePDB(fn, data, types, symbols, residues, box, comment, append=False)
       Input:
        1. filename: File to write
        2. data: np.array of shape (#atoms, #fields/atom)
        3. names: tuple of atom names
        4. symbols: tuple of atom symbols (contains strings)
        5. residues: tuple of fragment no. consistent to symbols with
                     residue names
        6. box: list of box parameters (xyz, 3angles)
        7. comment line (string)
        8. append: Append to file (optional, default = False)

Output: None"""

    if 0. in box:
        warnings.warn('expected non-void cell for output',
                      config.ChirPyWarning, stacklevel=2)
    format = '%s%7d %-5s%-4s%5d    '
    for field in range(data.shape[1]):
        format += '%8.3f'
    format += '%6.2f%6.2f %8s%2s\n'
    if selection is None:
        n_atoms = len(symbols)
        _range = range(n_atoms)
    else:
        n_atoms = len(selection)
        _range = selection
    obuffer = 'TITLE     %s\n' % title
    obuffer += 'CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1%12d\n' % (
                                                               box[0],
                                                               box[1],
                                                               box[2],
                                                               box[3],
                                                               box[4],
                                                               box[5],
                                                               1
                                                               )
    for i in _range:
        tmp = ['ATOM']
        tmp += [i+1]
        tmp += [names[i]]
        tmp += [residues[i][1]]
        tmp += [int(residues[i][0])]
        tmp += [c for c in data[i]]
        tmp += [1]
        tmp += [0]
        tmp += ['']
        tmp += [symbols[i]]
        obuffer += format % tuple(tmp)

    obuffer += 'MASTER        1    0    0    0    0    0    0    0 '
    obuffer += '%4d    0 %4d    0\nEND' % (n_atoms, n_atoms)

    fmt = 'w'
    if append:
        fmt = 'a'
    with open(fn, fmt) as f:
        f.write(obuffer)


def pdbWriter(fn, data, names, symbols, residues, box, title, selection=None,
              append=False):
    """WritePDBFile(filename, data, symbols, comments, append=False)
       Input:
        1. fn: File to write
        2. data: np.array of shape ([#frames,] #atoms, #fields/atom)
        3. symbols: list of atom symbols (contains strings)
        4. list of comment lines (contains strings)
        5. append: Append to file (optional, default = False)

        Output: None"""
    if len(data.shape) == 2:
        # ---frame
        _write_pdb_frame(fn, data, names, symbols, residues,
                         box, title, selection=selection, append=append)

    elif len(data.shape) == 3:
        # --- trajectory (NOT ENCOURAGED TO USE CHIRPY FOR PDB TRAJ FILES!)
        n_frames = len(data)
        for fr in range(n_frames):
            _write_pdb_frame(fn, data[fr], names, symbols, residues,
                             box, title, selection=selection,
                             append=append or fr != 0)
    else:
        raise AttributeError('Wrong data shape!', data.shape)


# def pdbWriter(fn, data, types, symbols, residues, box, title, append=False):
#   """WritePDB(fn, data, types, symbols, residues, box, comment, append=False)
#        Input:
#         1. filename: File to write
#         2. data: np.array of shape (#atoms, #fields/atom)
#         3. types: list of atom types
#         4. symbols: list of atom symbols (contains strings)
#         5. residues: np.array of fragment no. consistent to symbols with
#                      residue names
#         6. box: list of box parameters (xyz, 3angles)
#         7. comment line (string)
#         8. append: Append to file (optional, default = False)
#
# Output: None"""
#
#     format = '%s%7d %-5s%-4s%5d    '
#     for field in range(data.shape[1]):
#         format += '%8.3f'
#     format += '%6.2f%6.2f % 12s\n'
#     n_atoms = len(symbols)
#     obuffer = 'TITLE     %s\n' % title
#     obuffer += 'CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1%12d\n' % (
#                                                                box[0],
#                                                                box[1],
#                                                                box[2],
#                                                                box[3],
#                                                                box[4],
#                                                                box[5],
#                                                                1
#                                                                )
#     for i in range(n_atoms):
#         tmp = ['ATOM'] + [i+1] + [types[i]] + [residues[i][1]] + \
#                 [int(residues[i][0])] + [c for c in data[i]] + [1] \
#                 + [0] + [symbols[i]]
#         obuffer += format % tuple(tmp)
#     obuffer += 'MASTER        1    0    0    0    0    0    0    0 '
#     obuffer += '%4d    0 %4d    0\nEND' % (n_atoms, n_atoms)
#
#     fmt = 'w'
#     if append:
#         fmt = 'a'
#     with open(fn, fmt) as f:
#         f.write(obuffer)
