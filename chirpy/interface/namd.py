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


import subprocess
import numpy as np

# ToDo: old code


def read_output_file(fn_out, clean=False):
    fn = " ".join(fn_out)
    p = subprocess.Popen("egrep 'ENERGY:' %s | grep -v 'IMPRECISION'" % fn,
                         stdout=subprocess.PIPE,
                         shell=True)
    tmp = p.communicate()[0]

    data = list()
    for line in tmp.split('\n')[:-1]:
        data.append(line.split()[1:])
    data = np.array(data)

    print(data.shape)
    tss = np.array(data[1:, 0]).astype(int)
    energies_tot_si = np.array(data[1:, 10]).astype(float) * 4184.00
    energies_kin_si = np.array(data[1:, 9]).astype(float) * 4184.00
    energies_pot_si = np.array(data[1:, 12]).astype(float) * 4184.00
    temperatures = np.array(data[1:, 11]).astype(float)
    pressures_si = np.array(data[1:, 18]).astype(float) * 100000
    volumes_aa = np.array(data[1:, 17]).astype(float)

    # remove redundant steps
    if clean:
        tss, index = np.unique(tss, return_index=True)
        diffs = np.diff(index)
        not_one = diffs[diffs != 1] - 1
        redundant = index[diffs != 1] + not_one

        energies_tot_si = np.delete(energies_tot_si, redundant)
        energies_kin_si = np.delete(energies_kin_si, redundant)
        energies_pot_si = np.delete(energies_pot_si, redundant)
        temperatures = np.delete(temperatures, redundant)
        pressures_si = np.delete(pressures_si, redundant)
        volumes_aa = np.delete(volumes_aa, redundant)

    return tss, energies_tot_si, energies_kin_si, energies_pot_si, \
        temperatures, pressures_si, volumes_aa
