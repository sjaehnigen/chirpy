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
#  Copyright (c) 2010-2020, The ChirPy Developers.
#
#
#  Released under the GNU General Public Licence, v3
#
#   ChirPy is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published
#   by the Free Software Foundation, either version 3 of the License.
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

import sys
import numpy as np

from ..physics import constants
from ..topology import mapping
import copy


# --- ToDo: OLD CODE
def g09Reader(filename, run=1):
    # --- extract run
    with open(filename, 'r') as f:
        inbuffer = f.read()
    archives = inbuffer.count('Normal termination of Gaussian')
    if archives == 0:
        raise ValueError('No normal termination found!')
    if archives != run:
        raise ValueError('Found %d runs instead of %d!' % (archives, run))
    positions = [0]
    for i in range(run):
        tmp = inbuffer.index('Normal termination of Gaussian', positions[-1]+1)
        positions.append(tmp)
    inbuffer = inbuffer[positions[-2]:positions[-1]]
    # print(inbuffer, inbuffer.count('@'))
    # extract archive block
    if inbuffer.count('@') != 1:
        raise ValueError('No unique archive found!')
    # print(inbuffer)
    tmp_inbuffer = inbuffer.replace('\n ', '')
    start = tmp_inbuffer.rfind('\\\\#')
    end = inbuffer.rfind('@')

    archive_block = inbuffer[start:end].replace('\n ', '')
    archive_block = archive_block.rstrip('\\\\').split('\\\\')[3:]

    # for f in archive_block:
    #     print(f)

    # --- extract properties
    properties = dict()
    # --- extract title
    start = inbuffer.index(
            '-----------------------------------------------------\n') + 53
    end = inbuffer.index(
            '-----------------------------------------------------\n',
            start) - 12
    title = inbuffer[start:end].replace('\n', '').strip()
    properties['title'] = title

    def ParseAtomBlock(atom_block, properties):
        # print(atom_block)
        atom_block = atom_block.split('\\')[1:]
        symbols = list()
        coords = list()
        # print(atom_block)
        for atom in atom_block:
            tmp = atom.split(',')
            # print(len(tmp))
            symbols.append(tmp[0])
            if len(tmp) == 5:
                tmp = tmp[2:]
            else:
                tmp = tmp[1:]
            coords.append([float(e) for e in tmp])
        coords = np.array(coords)
        # --- ToDo: change key to pos_aa / pos_au
        properties['coords'] = coords
        properties['symbols'] = symbols
        properties['n_atoms'] = len(symbols)
        return properties

    def ParsePropertyBlock(property_block, properties):
        property_block = property_block.split('\\')
        for line in property_block:
            tmp = line.split('=')
            string_properties = ['Version', 'State', 'PG']
            if tmp[0] not in string_properties:
                properties[tmp[0]] = np.array(eval(tmp[1]))
            else:
                properties[tmp[0]] = tmp[1]
        return properties

    def ParseHessianBlock(hessian_block, n_atoms, properties):
        tmp = np.array(eval(hessian_block))
        # n_atoms = int((np.sqrt(2*tmp.shape[0]+0.5)-0.5)/3)
        hessian = np.zeros((3*n_atoms, 3*n_atoms))
        for col in range(3 * n_atoms):
            for row in range(col + 1):
                ind = col * (col + 1) // 2 + row
                hessian[col][row] = tmp[ind]
                hessian[row][col] = tmp[ind]
        properties['hessian'] = hessian
        return properties

    def WhatIsThis(block, n_atoms, properties):
        tmp = np.array(eval(block.rstrip('\\')))
        properties['tmp'] = tmp.reshape((n_atoms, 3))
        return properties

    properties = ParseAtomBlock(archive_block[0], properties)
    properties = ParsePropertyBlock(archive_block[1], properties)
    n_atoms = properties['n_atoms']

    if len(archive_block) > 2:
        properties = ParseHessianBlock(archive_block[2], n_atoms, properties)

    if len(archive_block) > 3:
        properties = WhatIsThis(archive_block[3], n_atoms, properties)

    # --- extract input coordinates
    if run == 2:
        start = inbuffer.find(
                'Redundant internal coordinates taken from checkpoint file')
        end = inbuffer.find('Recover connectivity data from disk.')
        in_coords_aa = list()
        if start != -1 and end != -1:
            for line in inbuffer[start:end].split('\n')[3:-1]:
                in_coords_aa.append([
                    float(e) for e in line.strip().split(',')[2:]])
            in_coords_aa = np.array(in_coords_aa)
            properties['in_coords_aa'] = in_coords_aa
    else:
        properties['in_coords_aa'] = properties['coords']

    pos1 = inbuffer.rfind('- Thermochemistry -')
    pos2 = inbuffer.rfind('Molecular mass')
    properties['numbers'] = list()
    properties['masses'] = list()
    for line in inbuffer[pos1:pos2].split('\n')[3:-1]:
        properties['numbers'].append(int(line.split()[-4]))
        properties['masses'].append(float(line.split()[-1]))

    pos1 = inbuffer.rfind('Standard orientation:')
    properties['std_coords'] = np.zeros((n_atoms, 3))
    for i, line in enumerate(inbuffer[pos1:].split('\n')[5:n_atoms+5]):
        properties['std_coords'][i] = np.array(
                [float(e) for e in line.strip().split()[-3:]])

    # --- post-processing
    if 'hessian' in properties:
        # ToDo: OLD CODE (not TESTED!)
        # --- diagonalise Hessian
        p_tra = True  # 't' in projections
        p_rot = True  # 'r' in projections

        # data['coords'] = data['std_coords']
        res = calculate_normal_modes(
                properties['n_atoms'],
                properties['masses'],
                properties['coords'],
                properties['hessian'],
                p_tra,
                p_rot
                )
        e_vec, e_val, mwe_vec, cmc = res
        n_modes = e_val.shape[0]
        e_vec = e_vec.reshape((n_modes, properties['n_atoms'], 3))

        properties['n_modes'] = n_modes
        properties['modes'] = mwe_vec
        properties['eivec'] = e_vec
        properties['omega_cgs'] = e_val
        properties['cmc'] = cmc

    return properties


def read_g09_md(filename, masses):
    def ExtractFrame(f, offset, flen=30000):
        f.seek(offset)
        inbuffer = f.read(flen)
        p1 = inbuffer.index('Step Number')
        # print(p1)
        p2 = inbuffer[p1:].index(
                'TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ') + p1
        # print(p2)
        return inbuffer[p1:p2], p2 + offset

    def ParseFrame(frame, n_atoms):
        frame = frame.replace('D+', 'E+')
        frame = frame.replace('D-', 'E-')
        p0 = frame.index('ETot = ')
        etot = float(frame[p0:p0+30].split()[2])
        p1 = frame.index('Cartesian coordinates: (bohr)')
        dat = frame[p1:].split('\n')
        pos = dat[1:n_atoms+1]
        vel = dat[n_atoms+2:2*n_atoms+2]
        data = list()
        for i, p, v in zip(range(n_atoms), pos, vel):
            data.append([float(e) for e in [p.split()[3],
                                            p.split()[5],
                                            p.split()[7],
                                            v.split()[3],
                                            v.split()[5],
                                            v.split()[7]]])
        return np.array(data), etot

    with open(filename, 'r') as f:
        # inbuffer = f.read(20000)
        # p1 = inbuffer.index('Input orientation')
        # p2 = inbuffer[p1:].index('Distance matrix') + p1
        # n_atoms = len(inbuffer[p1:p2].split('\n')[5:-2])
        n_atoms = len(masses)
        offset = 0  # p2
        frames = list()
        energies = list()
        try:
            while True:  # len(frames) < 2: #
                frame, offset = ExtractFrame(f, offset, flen=30000)
                # print(offset)
                frame, energy = ParseFrame(frame, n_atoms)
                if len(frames) == 0:
                    print(offset, frame)
                frames.append(frame)
                energies.append(energy)
        except ValueError as e:
            print(e)
            print('Trajectory finished after %d' % len(frames))
        frames = np.array(frames)
        energies = np.array(energies)
        pos = frames[:, :, :3]*constants.l_au2aa
        vel = frames[:, :, 3:]*constants.t_au  # /np.sqrt(constants.m_amu_au)
        # atomic_mass_unit = 1822.88848367 #*constants.m_p_si)
        for i_at in range(n_atoms):
            vel[:, i_at, :] /= np.sqrt(masses[i_at])
            # np.sqrt(atomic_mass_unit)*constants.t_fs2au
            # *1E-12 # sqrt(amu)*bohr/sec to a.u.
    return pos, vel, energies


def g09_extract_ir_data(filename):
    with open(filename, 'r') as f:
        data = np.array([line.split('--')[-1].split()
                         for line in f.readlines()
                         if 'Frequencies' in line or 'IR Inten' in line])
    freqs, inten = tuple([data[i::2].flatten().astype(float) for i in [0, 1]])
    return freqs, inten


def g09_extract_vcd_data(filename):
    with open(filename, 'r') as f:
        data = np.array([line.split('--')[-1].split()
                         for line in f.readlines()
                         if 'Frequencies' in line or 'Rot. str.' in line])
    freqs, rotst = tuple([data[i::2].flatten().astype(float) for i in [0, 1]])
    return freqs, rotst


def g09_extract_anharmonic_ir_data(filename):
    with open(filename, 'r') as f:
        istart = 1e99
        read1 = 0
        read2 = 0
        read3 = 0
        data1 = list()
        data2 = list()
        data3 = list()
        for i, line in enumerate(f.readlines()):
            if "Anharmonic Infrared Spectroscopy" in line:
                istart = i
            if 'Fundamental Bands' in line and i > istart:
                read1 = 1
            if 'Overtones' in line and i > istart:
                read2 = 1
            if 'Combination Bands' in line and i > istart:
                read3 = 1
            if line.strip() == '':
                read1 = 0
                read2 = 0
                read3 = 0
            read1 == 1 and data1.append(line)
            read2 == 1 and data2.append(line)
            read3 == 1 and data3.append(line)
    data = [
            [line.strip().split()[2], line.strip().split()[4]]
            for line in data1[3:]
            ] + [
                 [line.strip().split()[2], line.strip().split()[3]]
                 for line in data2[3:]
                 ] + [
                      [line.strip().split()[3], line.strip().split()[4]]
                      for line in data3[3:]
                      ]
    data = np.array(data).astype(float)

    return data[:, 0], data[:, 1]


def g09_get_ir_spectrum(filename, x0, x1, w,
                        n=1024, shape='gaussian', anharmonic=False, vcd=False):
    if anharmonic:
        freqs, inten = g09_extract_anharmonic_ir_data(filename)
    else:
        if vcd:
            freqs, inten = g09_extract_vcd_data(filename)
        else:
            freqs, inten = g09_extract_ir_data(filename)
    spectrum = np.zeros((n, 2))
    spectrum[:, 0] = np.linspace(x0, x1, n, endpoint=True)
    if shape == 'gaussian':
        spectrum[:, 1] = np.array([GaussianConvolution(point, freqs, inten, w)
                                   for point in spectrum[:, 0]])
    elif shape == 'lorentzian':
        spectrum[:, 1] = np.array([LorentzianConvolution(
                                    point, freqs, inten, w)
                                  for point in spectrum[:, 0]])
    else:
        print('Invalid line shape parametre')
        sys.exit(1)
    if not vcd:
        spectrum[:, 1] = 1000-spectrum[:, 1]
    return spectrum, freqs


# ToDo: Should not be here
def GaussianConvolution(x, freqs, inten, w):
    # subs = (m-x)/(0.5*w)
    return sum([inten[z] * np.exp(-np.log(2) * ((m-x) / (0.5*w))**2)
                for z, m in enumerate(freqs)])  # DEBUG unit convention!


def LorentzianConvolution(x, freqs, inten, w):
    # subs = (m-x)/(0.5*w)
    return sum([inten[z] / (1 + ((m-x) / (0.5*w))**2)
                for z, m in enumerate(freqs)])  # DEBUG unit convention!


def calculate_normal_modes(n_atoms, masses, coords, hessian,
                         p_tra=True, p_rot=True):
    sder, cmc = Projection(n_atoms, masses, coords, hessian, p_tra, p_rot)
    e_vec, e_val, mwe_vec = diagonalize_dynamical_matrix(n_atoms, masses, sder)
    omit = (int(p_tra) + int(p_rot)) * 3
    e_vec = e_vec[omit:]
    mwe_vec = mwe_vec[omit:]
    e_val = e_val[omit:]
    return e_vec, e_val, mwe_vec, cmc


def Projection(n_atoms, masses, coords, hessian, p_tra=True, p_rot=True):
    dim = n_atoms*3
    # no PBC support
    # --- ToDo: [0] necessary?
    com = mapping.cowt(coords[None, :, :], masses)[0]
    cmc = coords - com

    tra = np.zeros((3, dim))
    rot = np.zeros((3, dim))

    if p_tra:
        tmp = 1.0/np.sqrt(n_atoms)
        for at in range(n_atoms):
            k = at*3
            for i in range(3):
                tra[i][k+i] = tmp
            rot[0][k+1] = cmc[at][2]
            rot[0][k+2] = -cmc[at][1]
            rot[1][k+0] = -cmc[at][2]
            rot[1][k+2] = cmc[at][0]
            rot[2][k+0] = cmc[at][1]
            rot[2][k+1] = -cmc[at][0]
    if p_rot:
        for at in range(n_atoms):
            k = at*3
            for i in range(3):
                for j in range(3):
                    tmp = np.dot(rot[i], tra[j])
                    rot[i] = rot[i]-tmp*tra[j]
                for j in range(i):
                    tmp = np.dot(rot[i], rot[j])
                    rot[i] = rot[i]-tmp*rot[j]
                tmp = np.dot(rot[i], rot[i])  # indent??
                if tmp > 1E-10:
                    tmp = 1.0/np.sqrt(tmp)
                else:
                    tmp = 0.0
                rot[i] = tmp*rot[i]
    proj = np.diag(np.ones(dim))
#    print(tra)
    if p_tra:
        for i in range(3):
            proj = proj-np.outer(tra[i], tra[i])
#    print(proj)
    if p_rot:
        for i in range(3):
            proj = proj-np.outer(rot[i], rot[i])
    sder = copy.deepcopy(hessian)
    sder = np.dot(np.dot(proj, sder), proj)
    return sder, cmc


def diagonalize_dynamical_matrix(n_atoms, masses, hessian):
    """DiagonalizeDynamicalMatrix(n_atoms, masses, hessian)"""
    sder = copy.deepcopy(hessian)
    dim = n_atoms*3
    invmass = list()
    for at in range(n_atoms):
        for i in range(3):
            invmass.append(1.0/np.sqrt(masses[at]))
    for i in range(dim):
        for j in range(dim):
            sder[i][j] = sder[i][j]*invmass[i]*invmass[j]
    eigval_usort, eigvec_usort = np.linalg.eig(sder)
    eigval_usort = [5140.487 * np.sign(np.real(e))*np.sqrt(abs(np.real(e)))
                    for e in eigval_usort]
    eigval = np.sort(eigval_usort)
    eigvec = np.zeros((dim, dim))
    mwevec = np.zeros((dim, dim))
    for i in range(dim):
        j = eigval_usort.index(eigval[i])
        eigvec[i] = np.real(eigvec_usort[:, j])
    # mwevec = copy.copy(eigvec)
    for i in range(dim):
        mwevec[:, i] = eigvec[:, i]*invmass[i]

    return eigvec, eigval, mwevec
