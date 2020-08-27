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


import numpy as np
from ..physics.constants import eijk


def divrot(data, cell_vec):
    '''Gridded calculation of divergence and rotation of a vector field
       using gradient along the dimensions of the grid (x, y, z only in the
       case of tetragonal cells).

           data of shape 3, x, y, z
           cell_vec ... grid unit vectors (spacing)
       '''
    gradients = np.array(np.gradient(data, 1,
                                     np.linalg.norm(cell_vec[0]),
                                     np.linalg.norm(cell_vec[1]),
                                     np.linalg.norm(cell_vec[2]))[1:])
    div = gradients.trace(axis1=0, axis2=1)
    rot = np.einsum('ijk, jklmn->ilmn', eijk, gradients)

    return div, rot
