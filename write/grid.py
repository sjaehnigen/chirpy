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

# outbuffer method may result in memory outage ==> Replace it
def cubeWriter(filename, comment1, comment2, numbers, coords, cell, data, origin=np.zeros(3)):
    outbuffer = AssembleCubeFile(comment1, comment2, numbers, coords, cell, data, origin=origin)
    f = open(filename, 'w')
    f.write(outbuffer)
    f.close()

def AssembleCubeFile(comment1, comment2, numbers, coords, cell, data, origin=np.zeros(3)):
    obuffer = ''
    obuffer += comment1.rstrip('\n').replace('\n','')+'\n'
    obuffer += comment2.rstrip('\n').replace('\n','')+'\n'
    dim = list(data.shape)
    n_atoms = coords.shape[0]
    obuffer += '   %2d  %10.6f  %10.6f  %10.6f\n'%(n_atoms, origin[0], origin[1], origin[2])
    for i in range(3):
        obuffer += '   %2d  %10.6F  %10.6f  %10.6f\n'%(dim[i], cell[i][0], cell[i][1], cell[i][2])
    for atom in range(n_atoms):
        obuffer += '   %2d  %10.6f  %10.6f  %10.6f  %10.6f\n'%(numbers[atom], numbers[atom], coords[atom][0], coords[atom][1], coords[atom][2])
    for i_x in range(dim[0]):
        for i_y in range(dim[1]):
            for i_z in range(dim[2]):
                obuffer += '%13.5E'%data[i_x][i_y][i_z]
                if i_z % 5 == 0 and i_z != 0:
                    obuffer += '\n'
            obuffer += '\n'
    return obuffer

