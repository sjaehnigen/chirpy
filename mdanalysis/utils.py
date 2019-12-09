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


import sys 
import os

#import MDAnalysis as mda
#import MDAnalysis.analysis.rms
#import MDAnalysis.analysis.align

import numpy as np
from ..physics import constants

#migrate these tools to geometry or so
def dihedral(p1,p2,p3,p4):
    '''Definition along given points p1--p2--p3--p4; all p as 3d-np.vectors or np.arrays (last axis will be used)'''
    v1 = p1-p2
    v2 = p3-p2
    v3 = p4-p3
    n1 = np.cross(v1,v2)
    n2 = np.cross(v3,v2)
    dr = np.inner(np.cross(n1,n2),v2)
    dr /= np.linalg.norm(dr)
    #dih = np.arccos(np.inner(n1,n2)/(np.linalg.norm(n1,axis=-1)*np.linalg.norm(n2,axis=-1)))[0]
    dih = np.arccos(np.inner(n1,n2)/(np.linalg.norm(n1,axis=-1)*np.linalg.norm(n2,axis=-1)))
    dih*=dr #Direction of rotation
    return dih


############################################################################
#migrated from proteintools, completely untested with new MDA version

def AlignFrame(trj,ref,selection,mass_weighted=False):
    '''Use first frame of trj'''
    MDAnalysis.analysis.align.alignto(trj,ref, select=selection, mass_weighted=mass_weighted)
    return trj

def AlignTrajectory(trj,ref,selection,mass_weighted=False,scratch='/scratch/ssd/tmp/'):
    fn_tmp = os.path.join(scratch,'tmp.dcd')
    MDAnalysis.analysis.align.rms_fit_trj(trj,ref, select=selection, 
                                          filename=fn_tmp, 
                                          rmsdfile=None, 
                                          prefix='rmsfit_', 
                                          tol_mass=0.1, 
                                          strict=False, 
                                          force=True, 
                                          quiet=False,
                                          mass_weighted=mass_weighted
                                         )
    trj.load_new(fn_tmp)
    os.remove(fn_tmp)
    return trj

def CalculateRMSD(u,alignto,selections,mass_weighted=False):
    R = MDAnalysis.analysis.rms.RMSD(u, reference=u, select=alignto, groupselections=selections,
        #filename='rmsd.dat', 
        mass_weighted=mass_weighted, 
        tol_mass=0.1, 
        ref_frame=0,
        )
    R.run()
    b=R.rmsd    
    return b 

def CalculateRMSF(selection):
    R = MDAnalysis.analysis.rms.RMSF(selection)
    R.run()
    b=R.rmsf
    return b

##SCRIPTS BELOW HAVE TO BE REVISED AND DEBUGGED
def GuessVelocities(u,dt_fs,avg_energy_kin_si,n_atoms,masses_amu):
#This script does too much
    import MDAnalysis.analysis.density as density
    import tempfile
    fn_coords      = tempfile.TemporaryFile()#dir='/scratch/ssd/tmp/')
    fn_diff        = tempfile.TemporaryFile()#dir='/scratch/ssd/tmp/')
    fn_vels        = tempfile.TemporaryFile()#dir='/scratch/ssd/tmp/')
    fn_veln        = tempfile.TemporaryFile()#dir='/scratch/ssd/tmp/')

#loop traj (COSTS MEMORY AND TIME!)
    print('Please wait! This may take some time ...')
    a = u.select_atoms("all")
    dims     = list()
    tmp_coords   = list()
    for ts in u.trajectory:
        u.atoms.wrap()      
        dims.append(ts.dimensions)
        tmp_coords.append(a.coordinates().astype('float')) 
    print('Read coordinates: Done.')
    sys.stdout.flush()

    dims   = np.array(dims)

    tmp_coords = np.array(tmp_coords)
    n_frames, n_atoms, three = tmp_coords.shape
    coords = np.memmap(fn_coords,dtype='float64',mode='w+',shape=(n_frames,n_atoms,3))
    coords[:,:,:] = np.array(tmp_coords)
    del tmp_coords

    dt = dt_fs*constants.t_fs2au #2fs --> in au?
    #A quite inaccurate method, especially for large timesteps 
    diff        = np.memmap(fn_diff,dtype='float64',mode='w+',shape=(n_frames-1,n_atoms,3))
    diff[:,:,:] = np.abs(np.diff(coords,axis=0))%(dims[1:,np.newaxis,0:3]) #dims are discretised --> leads to (small) artefacts in modulo calcs (i.e. too high vels)

    vels_aa         = np.memmap(fn_vels,dtype='float64',mode='w+',shape=(n_frames-1,n_atoms,3))
    vels_aa[:,:,:]  = diff-2*(diff%(dims[1:,np.newaxis,0:3]/2)) #*constants.l_aa2au/dt not necessary if scaling with Ekin, generally too small vels

    velnorm_au       = np.memmap(fn_veln,dtype='float64',mode='w+',shape=(n_frames-1,n_atoms))
    velnorm_au[:,:]  = np.linalg.norm(vels_aa,axis=2)*constants.l_aa2au/dt

#Rescale velocities for each particle (since vels are averaged)
    #mean1  = np.mean(unscaled_vels_au,axis=0)
    mean1  = np.median(velnorm_au,axis=0)
    mean2  = (np.sqrt(avg_energy_kin_si/n_atoms*2)*1e10/1e15*constants.l_aa2au/constants.t_fs2au)/np.sqrt(masses_amu/1000)
    factor = np.mean(mean2/mean1)
    print(factor)
    #factor = 50
    #part_kin_si = (0.5*(unscaled_vels_au*1e15/1e10/constants.l_aa2au*constants.t_fs2au)**2*masses_amu[np.newaxis,:]/1000).sum(axis=1)
    #factors = energies_kin_si[1:]/part_kin_si
    #velnorm_au = unscaled_vels_au*factors[:,np.newaxis]
    
    print('Array vels uses %s MB memory each'%(vels_aa.nbytes/1024./1024.))
    print('Array velnorm uses %s MB memory each'%(velnorm_au.nbytes/1024./1024.))
    print('Done.')

    #del mean1, mean2, factor
    #del vels_au,velnorm_au
    return velnorm_au*factor#,vels_rescaled_au

