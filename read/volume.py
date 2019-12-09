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


import numpy as np

from .generators import _reader

def _cube(frame, **kwargs):
    '''Kernel for processing cube frame.'''

    if kwargs.get('n_lines') != int(frame[0].strip()) + 2:
        raise ValueError('Inconsistent XYZ file!')

    comment = frame[0] + frame[1].rstrip('\n')
    _split = (_l.strip().split() for _l in frame[2:])
    symbols, data = zip(*[(_l[0], _l[1:]) for _l in _split])

    return np.array(data).astype(float), symbols, comment


#         origin_au, n_grid, cell_vec_au = (0,0,0), [0,0,0], [(0,0,0), (0,0,0), (0,0,0)]
#         n_atoms, *origin_au = map(float, next(_it).split())
#         for i in range(3):
#             n_grid[i], *cell_vec_au[i] = map(float, next(_it).split())
#         n_atoms, *n_grid = map(int, [n_atoms] + n_grid )
#         numbers, coords_au = [0]*n_atoms, [(0,0,0)]*n_atoms
#         for i_atom in range(n_atoms):
#             numbers[i_atom], dummy, *coords_au[i_atom] = map(float, next(_it).split())
#         numbers = list(map(int, numbers))
#         volume_data = list()
#         for line in _it:
#             volume_data.extend(line.strip().split())

def cubeIterator(FN, **kwargs):
    '''Iterator for xyzReader
       Usage: next() returns data, symbols, comments of
       current frame'''
    _kernel = _cpmd

    with open(FN, 'r') as _f:
        _f.readline()
        _f.readline()
        _natoms = int(_f.readline().strip())
        _nx = int(_f.readline().strip())
        _ny = int(_f.readline().strip())
        _nz = int(_f.readline().strip())
        _nlines = 6 + abs(_natoms) + (int(_nz) / 6 + 1) * _ny * _nx

        if _natoms < 0:
            _nlines += 1

    return _reader(FN, _nlines, _kernel, **kwargs)



def _gen(fn):
    '''Global generator for all formats'''
    return (_line for _line in fn)

def cubeReader(FN):
    '''Iteratively read the content of a cube file and
       return a dictionary'''
    with open(FN, 'r') as _f:
        _it = _gen(_f)
        comments = next(_it) + next(_it)
        origin_au, n_grid, cell_vec_au = (0,0,0), [0,0,0], [(0,0,0), (0,0,0), (0,0,0)]
        n_atoms, *origin_au = map(float, next(_it).split())
        for i in range(3):
            n_grid[i], *cell_vec_au[i] = map(float, next(_it).split())
        n_atoms, *n_grid = map(int, [n_atoms] + n_grid )
        numbers, coords_au = [0]*n_atoms, [(0,0,0)]*n_atoms
        for i_atom in range(n_atoms):
            numbers[i_atom], dummy, *coords_au[i_atom] = map(float, next(_it).split())
        numbers = list(map(int, numbers))
        volume_data = list()
        for line in _it:
            volume_data.extend(line.strip().split())
        volume_data = np.array(volume_data).astype(np.float).reshape(n_grid)
        data = {'comments':comments,'origin_au':origin_au,'cell_vec_au':cell_vec_au,\
                'coords_au':coords_au,'numbers':numbers,'volume_data':volume_data}
        return data

# Something that may be included
# def ReadGZIPCubeFile(filename):
#     f = gzip.open(filename, 'rb')
#     inbuffer = f.read().decode('UTF-8')
#     f.close()
#     data, numbers, coords, cell, comment1, comment2, origin = ParseCubeFile(inbuffer)
#     return data, numbers, coords, cell, comment1, comment2, origin
# 
# def WriteGZIPCubeFile(filename, comment1, comment2, numbers, coords, cell, data, origin=np.zeros(3)):
#     outbuffer = AssembleCubeFile(comment1, comment2, numbers, coords, cell, data, origin=origin)
#     f = gzip.open(filename, 'wb')
#     f.write(bytes(outbuffer, 'UTF-8'))
#     f.close()

