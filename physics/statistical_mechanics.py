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
from ..physics import constants

def CalculateKineticEnergies(vel_au, masses):
    """CalculateKineticEnergies(velocities, data): velocities in a.u., masses in amu"""
    n_frames = vel_au.shape[0]
    n_atoms  = vel_au.shape[1]
    e_kin_au = np.zeros((n_frames, n_atoms))
    for i_at in range(n_atoms):
        e_kin_au[:,i_at] = masses[i_at]*np.square(vel_au[:,i_at]).sum(axis=1)
    return 0.5*e_kin_au*constants.m_amu_au # in au

#REVISE, SPLIT/MERGE, AND DEBUG Boltzmann stuff
def CalculateBoltzmannVelocities(vel_au, n_bins, species):
    """CalculateBoltzmannVelocities(velocities, n_bins, species): velocities in a.u."""
    if len(vel_au.shape) == 3:
        velnorm_au = np.linalg.norm(vel_au,axis=2)
    elif len(vel_au.shape) == 2:
        velnorm_au = vel_au
    histogram  = np.empty((species.shape[0],2,n_bins))

    for sp, atoms in enumerate(species):
        print(' -- species %s / %s'%(sp+1,species.shape[0]))
        print('        no. of atoms: %s'%len(atoms))
        histogram[sp,1,:], edges = np.histogram(velnorm_au[:,atoms], bins=n_bins, range=None, normed=False, weights=None, density=None)
        histogram[sp,0,:] = edges[0:-1]
    del velnorm_au
    return histogram

def BoltzmannAnalysis(velnorm_au,numBins,species):
    print('Starting Boltzmann analysis.')
    if species.shape[0] > 10:
        raise Exception('Are you sure? You are about to scan for a huge amount of species: %s!'%species.shape[0])
    #histogram  = CalculateBoltzmannVelocities(vels_au, numBins, species)
    histogram  = CalculateBoltzmannVelocities(velnorm_au, numBins, species) #if using scaled vels
    print('Histogram analysis finished.')
    return histogram

def BoltzmannExpected(T,species,numBins,masses_amu,histogram):
    def prob(m,v,T):
        N =  pow((m*constants.m_amu_au/(2.0*np.pi*constants.k_B_au*T)),(3.0/2.0))*4*np.pi
        p1 = np.exp(-(m*constants.m_amu_au*v**2)/(2*constants.k_B_au*T))
        p = N*p1*v**2
        return p

    simul = np.zeros((species.shape[0],numBins))
    theor = np.zeros((species.shape[0],numBins))

    for sp, atoms in enumerate(species):
        #print(histogram[sp,0])
        for atom in atoms:
            theor[sp] += prob(masses_amu[atom],histogram[sp,0],T)
        simul[sp] += histogram[sp,1]
        simul[sp] /= np.amax(simul[sp])    
        theor[sp] /= np.amax(theor[sp])    
    
    return simul, theor
