#!/usr/bin/env python
#------------------------------------------------------
#
#  ChirPy 0.1
#
#  https://hartree.chimie.ens.fr/sjaehnigen/ChirPy.git
#
#  2010-2016 Arne Scherrer
#  2014-2019 Sascha Jähnigen
#
#
#------------------------------------------------------


import numpy as np

def _gen(fn):
    '''Global generator for all formats'''
    return (_line for _line in fn)

def cubeReader(FN):
    '''Module by Arne Scherrer'''
    with open(FN, 'r') as _f:
        _it = _gen(_f)
        comments = next(_it) + next(_it)
        origin_au, n_grid, cell_au = (0,0,0), [0,0,0], [(0,0,0), (0,0,0), (0,0,0)]
        n_atoms, *origin_au = map(float, next(_it).split())
        for i in range(3):
            n_grid[i], *cell_au[i] = map(float, next(_it).split())
        n_atoms, *n_grid = map(int, [n_atoms] + n_grid )
        numbers, coords_au = [0]*n_atoms, [(0,0,0)]*n_atoms
        for i_atom in range(n_atoms):
            numbers[i_atom], dummy, *coords_au[i_atom] = map(float, next(_it).split())
        numbers = list(map(int, numbers))
        volume_data = list()
        for line in _it:
            volume_data.extend(line.strip().split())
        volume_data = np.array(volume_data).astype(np.float).reshape(n_grid)
        data = {'comments':comments,'origin_au':origin_au,'cell_au':cell_au,\
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

