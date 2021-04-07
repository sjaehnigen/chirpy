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
#  Copyright (c) 2010-2021, The ChirPy Developers.
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

# outbuffer method may result in memory outage ==> Replace it

from ..physics import constants


def cubeWriter(fn, comments, numbers, pos_aa, cell_vec_aa, data,
               append=False,
               origin_aa=[0.0, 0.0, 0.0]):
    '''Write grid/volume data into a Gaussian Cube file.
       cell_vec_aa specifies the three cell vectors A, B, C in atomic units
       numbers is a lits or tuple of atomic numbers (or any numbers).
       Expects a single cube frame and does not support direct
       output of cube trajectories (use an iterator and append=True for this).
       '''
    pos_au = pos_aa * constants.l_aa2au
    cell_vec_au = cell_vec_aa * constants.l_aa2au
    origin_au = origin_aa * constants.l_aa2au
    fmt = 'w'
    if append:
        fmt = 'a'
    with open(fn, fmt) as f:
        outbuffer = _assemble_cube_file(
                                comments[0],
                                comments[1],
                                numbers,
                                pos_au,
                                cell_vec_au,
                                data,
                                origin=origin_au
                                )
        f.write(outbuffer)


def _assemble_cube_file(comment1,
                        comment2,
                        numbers,
                        coords,
                        cell,
                        data,
                        origin
                        ):
    '''Old code but stil in use. Revised and corrected in Dec 2019.'''
    obuffer = ''
    obuffer += comment1.rstrip('\n').replace('\n', '')+'\n'
    obuffer += comment2.rstrip('\n').replace('\n', '')+'\n'
    dim = list(data.shape)
    n_atoms = coords.shape[0]

    obuffer += '   %2d  %10.6f  %10.6f  %10.6f\n' % (
                                                  n_atoms,
                                                  origin[0],
                                                  origin[1],
                                                  origin[2]
                                                  )

    for i in range(3):
        obuffer += '   %2d  %10.6F  %10.6f  %10.6f\n' % (
                                                       dim[i],
                                                       cell[i][0],
                                                       cell[i][1],
                                                       cell[i][2]
                                                       )

    for atom in range(n_atoms):
        obuffer += '   %2d  %10.6f  %10.6f  %10.6f  %10.6f\n' % (
                                                            numbers[atom],
                                                            numbers[atom],
                                                            coords[atom][0],
                                                            coords[atom][1],
                                                            coords[atom][2]
                                                            )

    for i_x in range(dim[0]):
        for i_y in range(dim[1]):
            for i_z in range(dim[2]):
                if i_z % 6 == 0 and i_z != 0:
                    obuffer += '\n'
                obuffer += '%13.5E' % data[i_x][i_y][i_z]
            obuffer += '\n'

    return obuffer
