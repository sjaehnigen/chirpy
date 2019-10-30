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


import sys
import numpy as np

from lib import constants

# ToDo: Needs some cleanup


def read_g09_file(filename, run=1):
    # extract run
    with open(filename, 'r') as f:
        inbuffer = f.read()
    archives = inbuffer.count('Normal termination of Gaussian')
    if archives == 0:
        raise Exception('No normal termination found!')
    if archives < run:
        raise Exception('Only %d runs found!'%archives)
    positions = [0]
    for i in range(run):
        tmp = inbuffer.index('Normal termination of Gaussian', positions[-1]+1)
        positions.append(tmp)
    # print(positions)
    inbuffer = inbuffer[positions[-2]:positions[-1]]
    # print(inbuffer, inbuffer.count('@'))
    # extract archive block
    if inbuffer.count('@') != 1:
        raise Exception('No unique archive found!')
    # print(inbuffer)
    tmp_inbuffer = inbuffer.replace('\n ','')
    start = tmp_inbuffer.rfind('\\\\#')
    end = inbuffer.rfind('@')
    archive_block = inbuffer[start:end].replace('\n ','')
    archive_block = archive_block.rstrip('\\\\').split('\\\\')[3:]

    # for f in archive_block:
    #     print(f)
    #     print()
    # print(archive_block)

    # extract properties
    properties = dict()
    # extract title
    start = inbuffer.index('-----------------------------------------------------\n') + 53
    end = inbuffer.index('-----------------------------------------------------\n', start) - 12
    title = inbuffer[start:end].replace('\n','').strip()
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
        properties['coords' ] = coords
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
        for col in range(3*n_atoms):
            for row in range(col+1):
                ind = col*(col+1)/2+row
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

    # extract input coordinates
    if run == 2:
        start = inbuffer.find('Redundant internal coordinates taken from checkpoint file')
        end = inbuffer.find('Recover connectivity data from disk.')
        in_coords_aa = list()
        if start != -1 and end != -1:
            for line in inbuffer[start:end].split('\n')[3:-1]:
                in_coords_aa.append([float(e) for e in line.strip().split(',')[2:]])
            in_coords_aa = np.array(in_coords_aa)
            properties['in_coords_aa'] = in_coords_aa
    else:
        properties['in_coords_aa'] = properties['coords']

    pos1 = inbuffer.rfind('- Thermochemistry -')
    pos2 = inbuffer.rfind('Molecular mass')
    properties['numbers']=list()
    properties['masses' ]=list()
    for line in inbuffer[pos1:pos2].split('\n')[3:-1]:
        properties['numbers'].append(int(line.split()[-4]))
        properties['masses'].append(float(line.split()[-1]))

    pos1 = inbuffer.rfind('Standard orientation:')
    properties['std_coords'] = np.zeros((n_atoms, 3))
    for i,line in enumerate(inbuffer[pos1:].split('\n')[5:n_atoms+5]):
        properties['std_coords'][i]=np.array([float(e) for e in line.strip().split()[-3:]])

    return properties


def read_g09_md(filename, masses):

    def ExtractFrame(f, offset, flen=30000):
        f.seek(offset)
        inbuffer = f.read(flen)
        p1 = inbuffer.index('Step Number')
        # print(p1)
        p2 = inbuffer[p1:].index('TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ') + p1
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
        for i,p,v in zip(range(n_atoms), pos,vel):
            data.append([float(e) for e in [p.split()[3], p.split()[5], p.split()[7], v.split()[3], v.split()[5], v.split()[7]]])
        return np.array(data), etot

    with open(filename, 'r') as f:
        # inbuffer = f.read(20000)
        # p1 = inbuffer.index('Input orientation')
        # p2 = inbuffer[p1:].index('Distance matrix') + p1
        # n_atoms = len(inbuffer[p1:p2].split('\n')[5:-2])
        n_atoms = len(masses)
        offset= 0#p2
        frames = list()
        energies = list()
        try:
            while True:# len(frames) < 2: #
                frame, offset = ExtractFrame(f, offset, flen=30000)
                # print(offset)
                frame, energy = ParseFrame(frame, n_atoms)
                if len(frames) == 0:
                    print(offset, frame)
                frames.append(frame)
                energies.append(energy)
        except ValueError as e:
            print(e)
            print('Trajectory finished after %d'%len(frames))
        frames = np.array(frames)
        energies = np.array(energies)
        pos = frames[:,:,:3]*constants.l_au2aa
        vel = frames[:,:,3:]*constants.t_au#/np.sqrt(constants.m_amu_au)
        # atomic_mass_unit = 1822.88848367 #*constants.m_p_si)
        for i_at in range(n_atoms):
            vel[:,i_at,:] /= np.sqrt(masses[i_at])
             #np.sqrt(atomic_mass_unit)*constants.t_fs2au# *1E-12 # sqrt(amu)*bohr/sec to a.u.
    return pos, vel, energies


def g09_extract_ir_data(filename):
    with open(filename,'r') as f:
        data = np.array([line.split('--')[-1].split() for line in f.readlines() if 'Frequencies' in line or 'IR Inten' in line ])
    freqs,inten = tuple([data[i::2].flatten().astype(float) for i in [0,1] ])
    return freqs, inten

def g09_extract_vcd_data(filename):
    with open(filename,'r') as f:
        data = np.array([line.split('--')[-1].split() for line in f.readlines() if 'Frequencies' in line or 'Rot. str.' in line ])
    freqs,rotst = tuple([data[i::2].flatten().astype(float) for i in [0,1] ])
    return freqs, rotst

def g09_extract_anharmonic_ir_data(filename):
    with open(filename,'r') as f:
        istart = 1e99
        read1=0
        read2=0
        read3=0
        data1=list()
        data2=list()
        data3=list()
        for i,line in enumerate(f.readlines()):
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
            read1 ==1 and data1.append(line)
            read2 ==1 and data2.append(line)
            read3 ==1 and data3.append(line)
    data = [[line.strip().split()[2],line.strip().split()[4]] for line in data1[3:]] + [[line.strip().split()[2],line.strip().split()[3]] for line in data2[3:]] + [[line.strip().split()[3],line.strip().split()[4]] for line in data3[3:]]
    data = np.array(data).astype(float)
    return data[:,0], data[:,1]

def g09_get_ir_spectrum(filename,x0,x1,w,n=1024,shape='gaussian',anharmonic=False,vcd=False):
    if anharmonic:
        freqs, inten = g09_extract_anharmonic_ir_data(filename)
    else:
        if vcd:
            freqs, inten = g09_extract_vcd_data(filename)
        else:
            freqs, inten = g09_extract_ir_data(filename)
    spectrum = np.zeros((n,2))
    spectrum[:,0] = np.linspace(x0,x1,n,endpoint=True)
    if shape=='gaussian':
        spectrum[:,1] = np.array([GaussianConvolution(point,freqs,inten,w) for point in spectrum[:,0]])
    elif shape=='lorentzian':
        spectrum[:,1] = np.array([LorentzianConvolution(point,freqs,inten,w) for point in spectrum[:,0]])
    else:
        print('Invalid line shape parametre')
        sys.exit(1)
    if not vcd:
        spectrum[:,1] = 1000-spectrum[:,1]
    return spectrum, freqs

# Should not be here
def GaussianConvolution(x,freqs,inten,w):
    #subs = (m-x)/(0.5*w)
    return sum([inten[z]*np.exp(-np.log(2)*((m-x)/(0.5*w))**2) for z,m in enumerate(freqs)]) #DEBUG unit convention!

def LorentzianConvolution(x,freqs,inten,w):
    #subs = (m-x)/(0.5*w)
    return sum([inten[z]/(1 + ((m-x)/(0.5*w))**2) for z,m in enumerate(freqs)]) #DEBUG unit convention!
