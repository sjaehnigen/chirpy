#!/usr/bin/python3

import os, sys, copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from vcdtools import vcd_utils
from fileio.cpmd import WriteTrajectoryFile,WriteMomentsFile

#class SPECTRUM():
#
#this should be a trajectory object
class VCDSampling(): #later: merge it with itertools (do not load any traj data before the actual processing)
    def __init__(self,path,**kwargs): #**kwargs for named (dict), *args for unnamed
        fn_m = os.path.join(path,'MOMENTS')
        fn_t = os.path.join(path,'TRAJECTORY')
        self.path = path
        self.n_moms  = vcd_utils.NumberOfMoleculesFromMomentFile(fn_m) #later replace this by own routine     
        buf = np.loadtxt(fn_m, usecols=(1,2,3,4,5,6,7,8,9))
        self.n_frames = buf.shape[0]//self.n_moms        
        self.moments = buf.reshape((self.n_frames, self.n_moms, 3, 3))
        del buf
        
        self.n_atoms = vcd_utils.NumberOfMoleculesFromMomentFile(fn_t) #later replace this by own routine        
        buf = np.loadtxt(fn_t, usecols=(1,2,3,4,5,6))
        if self.n_frames != buf.shape[0]//self.n_atoms:
            raise Exception('ERROR: MOMENTY and TRAJECTORY file differ in their number of frames:%s, %s'%(self.n_frames,buf.shape[0]//self.n_atoms))
        self.trajectory = buf.reshape((self.n_frames, self.n_atoms, 6))
        del buf        
        
    def __add__(self,other):
        new = copy.deepcopy(self)
        new.moments = np.concatenate((self.moments,other.moments),axis=0)
        new.trajectory = np.concatenate((self.trajectory,other.trajectory),axis=0)     
        new.n_frames = self.n_frames + other.n_frames
        return new
    
    def __iadd__(self,other):
        self.moments = np.concatenate((self.moments,other.moments),axis=0)
        self.trajectory = np.concatenate((self.trajectory,other.trajectory),axis=0)     
        self.n_frames += other.n_frames
        return self
    
    def __prod__(self,other):
        new = copy.deepcopy(self)
        new.moments = np.concatenate((self.moments,other.moments),axis=1)
        new.trajectory = np.concatenate((self.trajectory,other.trajectory),axis=1)     
        new.n_moms = self.n_moms + other.n_moms        
        new.n_atoms = self.atoms + other.atoms
        return new
    
    
    def tail(self,n):
        new = copy.deepcopy(self)        
        new.moments = self.moments[-n:]
        new.trajectory = self.trajectory[-n:]
        new.n_frames = n
        return new
    
    def write(self,path):
        if path == self.path:
            raise Exception('ERROR: write path equals source path. TRAJECTORY and MOMENTS files would be overwritten!')
        if not os.path.exists(path):
            os.makedirs(path)            
        fn_m = os.path.join(path,'MOMENTS')
        fn_t = os.path.join(path,'TRAJECTORY')
        #WriteTrajectoryFile(fn_t, self.trajectory[:,:,:3], self.trajectory[:,:,3:], offset=0)

        fmt = '   %22.12E'
        with open(fn_t, 'w') as f:
            for i_frame in range(self.n_frames):
                for i_atom in range(self.n_atoms):
                    #f.write('% 7d'%(i_frame+1)+(fmt*9)%tuple(self.moments[i_frame,i_mom]))
                    f.write('% 7d'%(i_frame+1)
                            +(fmt*3)%tuple(self.trajectory[i_frame,i_atom,:3])
                            +(fmt*3)%tuple(self.trajectory[i_frame,i_atom,3:])
                            +'\n')        
        fmt = '   %22.12E'
        with open(fn_m, 'w') as f:
            for i_frame in range(self.n_frames):
                for i_mom in range(self.n_moms):
                    #f.write('% 7d'%(i_frame+1)+(fmt*9)%tuple(self.moments[i_frame,i_mom]))
                    f.write('% 7d'%(i_frame+1)
                            +(fmt*3)%tuple(self.moments[i_frame,i_mom,0])
                            +(fmt*3)%tuple(self.moments[i_frame,i_mom,1])
                            +(fmt*3)%tuple(self.moments[i_frame,i_mom,2])
                            +'\n')


    def unravel(self): #better use ther iterlist routines of Arne
        self.r_aa, self.c_aa, self.m_aa = tuple(np.rollaxis(self.moments,axis=2)) 
    #    return tuple(np.rollaxis(self.moments,axis=2))
        
