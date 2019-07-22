#!/usr/bin/env python

import numpy as np

def cubeReader(fn):
    '''Module by Arne Scherrer'''
    with open(fn, 'r') as f:
        comments = f.readline() + f.readline()
        origin_au, n_grid, cell_au = (0,0,0), [0,0,0], [(0,0,0), (0,0,0), (0,0,0)]
        n_atoms, *origin_au = map(float, f.readline().split())
        for i in range(3):
            n_grid[i], *cell_au[i] = map(float, f.readline().split())
        n_atoms, *n_grid = map(int, [n_atoms] + n_grid )
        numbers, coords_au = [0]*n_atoms, [(0,0,0)]*n_atoms
        for i_atom in range(n_atoms):
            numbers[i_atom], dummy, *coords_au[i_atom] = map(float, f.readline().split())
        numbers = list(map(int, numbers))
        volume_data = list()
        for line in f.readlines():
            volume_data.extend(line.strip().split())
        volume_data = np.array(volume_data).astype(np.float).reshape(n_grid)
        data = {'comments':comments,'origin_au':origin_au,'cell_au':cell_au,\
                'coords_au':coords_au,'numbers':numbers,'volume_data':volume_data}
        return data

