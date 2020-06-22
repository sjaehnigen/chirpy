#!/usr/bin/env python
# ------------------------------------------------------
#
#  ChirPy
#
#    A buoyant python package for analysing supramolecular
#    and electronic structure, chirality and dynamics.
#
#
#  Developers:
#    2010-2016  Arne Scherrer
#    since 2014 Sascha Jähnigen
#
#  https://hartree.chimie.ens.fr/sjaehnigen/chirpy.git
#
# ------------------------------------------------------


def _write_xyz_frame(filename, data, symbols, comment, append=False):
    """WriteFrame(filename, data, symbols, comment, append=False)
    Input:
        1. filename: File to read
        2. data: np.array of shape (#atoms, #fields/atom)
        3. symbols: list of atom symbols (contains strings)
        4. comment line (string)
        5. append: Append to file (optional, default = False)
    Output: None"""

    format = '  %s'
    # print(data.shape, data.shape[1])
    for field in range(data.shape[1]):
        # format += '    %16.12f'
        format += '      %14.10f'  # cp2k format ...
    format += '\n'
    n_atoms = len(symbols)
    obuffer = '% 8d\n' % n_atoms
    # if comment[-1] != '\n':
    #     comment += '\n'
    obuffer += comment.rstrip('\n') + '\n'
    # print(format)
    for i in range(n_atoms):
        tmp = ['{0: <2}'.format(symbols[i])] + [c for c in data[i]]
        obuffer += format % tuple(tmp)
    fmt = 'w'
    if append:
        fmt = 'a'
    with open(filename, fmt) as f:
        f.write(obuffer)


def xyzWriter(fn, data, symbols, comments, append=False):
    """WriteXYZFile(filename, data, symbols, comments, append=False)
       Input:
        1. fn: File to write
        2. data: np.array of shape ([#frames,] #atoms, #fields/atom)
        3. symbols: list of atom symbols (contains strings)
        4. list of comment lines (contains strings)
        5. append: Append to file (optional, default = False)

        Output: None"""
    if len(data.shape) == 2:
        # ---frame
        _write_xyz_frame(fn, data, symbols,
                         comments, append=append)

    elif len(data.shape) == 3:
        # --- trajectory
        n_frames = len(comments)
        for fr in range(n_frames):
            _write_xyz_frame(fn, data[fr], symbols,
                             comments[fr], append=append or fr != 0)

    else:
        raise AttributeError('Wrong data shape!', data.shape)


def _write_arc_frame(fn, data, symbols, numbers, types, connectivity,
                     comment=None, append=False):
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
    # --- kwargs for additional header lines (crystal etc.)

    # --- header
    n_atoms = len(symbols)
    obuffer = '{:>6d}'.format(n_atoms)
    if comment is not None:
        obuffer += f'  {comment}'
    obuffer += '\n'

    # --- corpus

    for _n, _s, _d, _t, _c in zip(numbers, symbols, data, types, connectivity):
        _line = '{:>6d}  {:3s}' + 3 * '{:>12.6f}' + '{:>6d}' +\
            len(_c)*'{:>6d}' + '\n'
        obuffer += _line.format(_n, _s, *_d, _t, *_c)

    # --- type
    fmt = 'w'
    if append:
        fmt = 'a'
    with open(fn, fmt) as f:
        f.write(obuffer)


def arcWriter(fn, data, symbols, numbers, types, connectivity,
              comments=None, append=False):
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

    if len(data.shape) == 2:
        # ---frame
        _write_arc_frame(fn, data, symbols, numbers, types, connectivity,
                         comment=comments, append=append)

    elif len(data.shape) == 3:
        # --- trajectory
        n_frames = len(data)
        for fr in range(n_frames):
            _write_arc_frame(fn, data[fr], symbols, numbers, types,
                             connectivity, comment=comments[fr],
                             append=append or fr != 0)

    else:
        raise AttributeError('Wrong data shape!', data.shape)


def _write_pdb_frame(fn, data, types, symbols, residues, box, title,
                     append=False):
    """WritePDB(fn, data, types, symbols, residues, box, comment, append=False)
       Input:
        1. filename: File to write
        2. data: np.array of shape (#atoms, #fields/atom)
        3. types: list of atom types
        4. symbols: list of atom symbols (contains strings)
        5. residues: np.array of fragment no. consistent to symbols with
                     residue names
        6. box: list of box parameters (xyz, 3angles)
        7. comment line (string)
        8. append: Append to file (optional, default = False)

Output: None"""

    format = '%s%7d %-5s%-4s%5d    '
    for field in range(data.shape[1]):
        format += '%8.3f'
    format += '%6.2f%6.2f % 12s\n'
    n_atoms = len(symbols)
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
    for i in range(n_atoms):
        tmp = ['ATOM'] + [i+1] + [types[i]] + [residues[i][1]] + \
                [int(residues[i][0])] + [c for c in data[i]] + [1] \
                + [0] + [symbols[i]]
        obuffer += format % tuple(tmp)
    obuffer += 'MASTER        1    0    0    0    0    0    0    0 '
    obuffer += '%4d    0 %4d    0\nEND' % (n_atoms, n_atoms)

    fmt = 'w'
    if append:
        fmt = 'a'
    with open(fn, fmt) as f:
        f.write(obuffer)


def pdbWriter(fn, data, types, symbols, residues, box, title, append=False):
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
        _write_pdb_frame(fn, data, types, symbols, residues,
                         box, title, append=append)

    elif len(data.shape) == 3:
        # --- trajectory (NOT ENCOURAGED TO USE CHIRPY FOR PDB TRAJ FILES!)
        n_frames = len(data)
        for fr in range(n_frames):
            _write_pdb_frame(fn, data[fr], types, symbols, residues,
                             box, title, append=append or fr != 0)
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
