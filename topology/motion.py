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
# import sys
# import copy

# from ..physics import constants

# from geometry import transformations as trans
# from trajectory import trajectory_xyz_clean
# from trajectory import trajectory_xyz_purge_com
# from trajectory import analyze_trajectory


def AnalyzeMotion(traj_data, info): #first vels then pos??
    center_of_masses = np.zeros((info['n_frames'], 3))
    linear_momenta   = np.zeros((info['n_frames'], 3))
    angular_momenta  = np.zeros((info['n_frames'], 3))    
    angular_momenta_atomwise  = np.zeros((info['n_frames'], info['n_atoms'], 3))

  #  for frame in range(info['n_frames']):
       # print('Analyzing frame %s / %s...' % (frame,info['n_frames']),end='\r')
    center_of_masses = trans.CentersOfMass(traj_data[:,:,:3], info['masses'])
    linear_momenta   = trans.LinearMomenta(traj_data[:,:,3:], info['masses'])
    angular_momenta  = trans.AngularMomenta(traj_data[:,:,:3]-center_of_masses[:,np.newaxis,:], traj_data[:,:,3:], info['masses'])

    return center_of_masses, linear_momenta, angular_momenta

def AngularVelocities(coords,velocities,info):
    angvels  = np.zeros((info['n_frames'],info['n_atoms'],3))
    angmoms  = np.zeros((info['n_frames'],info['n_atoms'],3))
    pulse    = np.cross(coords-info['center_of_masses'][:,np.newaxis,:],info['angular_momenta'],axisa=2,axisb=1)
    tang_dir = pulse/np.linalg.norm(pulse, axis=2)[:,:,np.newaxis]
    mom_dir  = info['angular_momenta']/np.linalg.norm(info['angular_momenta'], axis=1)[:,np.newaxis]
    tang_vel = (tang_dir*velocities).sum(axis=2)[:,:,np.newaxis]*tang_dir
    angvels  = (np.cross(coords-info['center_of_masses'][:,np.newaxis,:], tang_vel,axisa=2,axisb=2)/np.square(coords-info['center_of_masses'][:,np.newaxis,:]).sum(axis=2)[:,:,np.newaxis])
    angmoms  = (np.cross(coords-info['center_of_masses'][:,np.newaxis,:], tang_vel,axisa=2,axisb=2)*info['masses'][np.newaxis,:,np.newaxis])
    angvels  = (angvels*mom_dir[:,np.newaxis,:]).sum(axis=2)[:,:,np.newaxis]*mom_dir[:,np.newaxis,:]
    angmoms  = (angmoms*mom_dir[:,np.newaxis,:]).sum(axis=2)[:,:,np.newaxis]*mom_dir[:,np.newaxis,:]
    return angvels, angmoms


def EulerAngles(x, y, z, convention='zyz', verbose=False):
    x /= np.linalg.norm(x,axis=-1)
    y /= np.linalg.norm(y,axis=-1)
    z /= np.linalg.norm(z,axis=-1)
    # http://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
    # http://en.wikipedia.org/wiki/Gimbal_lock
    if convention == 'zyz':
        c2 = z[2]
        beta = np.arccos(c2)
        if c2 > +(1-1E-10):
            if verbose:
                print('Gimbal Lock: z -> +Z')
            alpha = 0.0
            gamma = np.arctan2(x[1],x[0])-alpha
        elif c2 < -(1-1E-10):
            if verbose:
                print('Gimbal Lock: z -> -Z')
            alpha = 0.0
            gamma = alpha-np.arctan2(-x[1],-x[0])
        else:
            s2 = np.sqrt(1-c2**2)
            alpha = np.arctan2(z[1],+z[0])
            gamma = np.arctan2(y[2],-x[2])
    else:
        raise Exception('Convention %s not supported!'%convention)
    return np.array([alpha, beta, gamma])


# Also:
# def CalculateTranslationalTemperature(linmoms, info)
# def CalculateRotationalTemperature(angmoms, angvels, info)
# def CalculateTemperature(vels, info)
