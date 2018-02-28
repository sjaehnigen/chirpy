#!/usr/bin/env python3
import numpy as np
import sys
import copy
from classes.volume import ScalarField,VectorField
from classes.domain import Domain3D,Domain2D
from scipy.integrate import simps
from scipy.signal import medfilt
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion, binary_dilation

#old
from fileio import cube,xyz #Replace
from lib import constants

class ElectronDensity(ScalarField):        
    def integral(self):
        self.n_electrons = self.voxel*simps(simps(simps(self.data)))
        self.threshold = 1.E-3

    def aim(self): #export this method to grid class
        '''Min Yu and Dallas R. Trinkle, Accurate and efficient algorithm for Bader charge integration, J. Chem. Phys. 134, 064111 (2011)'''
        def pbc(a,dim):
            return np.remainder(a,self.data.shape[dim])
        def env_basin(f,x,y,z):
            return np.array([f[x,y,z],f[pbc(x+1,0),y,z],f[x-1,y,z],f[x,pbc(y+1,1),z],f[x,y-1,z],f[x,y,pbc(z+1,2)],f[x,y,z-1]])
        
       # self.n_x       = self.data.shape[0]
       # self.n_y       = self.data.shape[1]
       # self.n_z       = self.data.shape[2]
        self.aim_threshold = self.threshold
        boundary_max = 0 
        boundary_max = max(boundary_max,np.amax(self.data[ 0,:,:]))
        boundary_max = max(boundary_max,np.amax(self.data[-1,:,:]))
        boundary_max = max(boundary_max,np.amax(self.data[:, 0,:]))
        boundary_max = max(boundary_max,np.amax(self.data[:,-1,:]))
        boundary_max = max(boundary_max,np.amax(self.data[:,:, 0]))
        boundary_max = max(boundary_max,np.amax(self.data[:,:,-1]))
        if boundary_max >= self.aim_threshold:
            #raise Exception('ERROR: Your density at the boundary exceeds given density threshold of %f! %f'%(self.aim_threshold,boundary_max))
            print('WARNING: Your density at the boundary exceeds given density threshold of %f! %f'%(self.aim_threshold,boundary_max))

        #neighborhood = generate_binary_structure(3,1)
        test = np.unravel_index(np.argsort(self.data.ravel())[::-1], self.data.shape)
        mask0 = np.zeros(self.data.shape,dtype=bool)
        
        #atoms = range(0,rho.n_atoms*2)
        atoms = range(self.n_atoms)

        basin = np.zeros([j for i in (self.data.shape,len(atoms)) for j in (i if isinstance(i,tuple) else (i,))])
        atoms = iter(atoms)
        n_points = (self.data > self.aim_threshold).sum()
        g0 = self.data
        g1 = np.roll(self.data,-1,axis=0)
        g2 = np.roll(self.data,+1,axis=0)
        g3 = np.roll(self.data,-1,axis=1)
        g4 = np.roll(self.data,+1,axis=1)
        g5 = np.roll(self.data,-1,axis=2)
        g6 = np.roll(self.data,+1,axis=2)
        R = np.array([g0, g1, g2, g3, g4, g5, g6]) - self.data[np.newaxis]
        R[R<0] = 0
        #R /= R.sum(axis=0)

        gain = R.sum(axis=0)
        for i in range(n_points): 
            if (100*i/n_points)%1 == 0: print('Scanning point %d/%d'%(i,n_points),end="\r")
            ix = test[0][i]
            iy = test[1][i]
            iz = test[2][i]
            if self.data[ix,iy,iz] > self.aim_threshold:
                #gain = R[:,ix,iy,iz].sum(axis=0) #calculate beforehand?
                if gain[ix,iy,iz] != 0: 
                    basin[ix,iy,iz] = (env_basin(basin,ix,iy,iz)*R[:,ix,iy,iz,np.newaxis]).sum(axis=0)/gain[ix,iy,iz]
                else: 
                    iatom = next(atoms)
                    basin[ix,iy,iz,iatom] = 1.0
        print('AIM analysis done.                                                                                                                          ')

        aim_atoms = list()
        pos_grid = self.pos_grid()
        for iatom in range(self.n_atoms):
            ind = np.unravel_index(np.argmin(np.linalg.norm(pos_grid[:,:,:,:]-self.pos_au[iatom,:,np.newaxis,np.newaxis,np.newaxis],axis=0)),self.data.shape)
            jatom = np.argmax(basin[ind])
            transfer = (self.comments,self.numbers[iatom].reshape((1)),self.pos_au[iatom].reshape((1,3)),self.cell_au,self.origin_au) # outer class won't be accessible even with inheritance
            aim_atoms.append(AIMAtom(basin[:,:,:,jatom],transfer)) 
        self.aim_atoms = np.array(aim_atoms)

        #NEXT: AIM consistency checks
        #   No. of atoms
        #   Sum Rules
        #   Gradient zero

class AIMAtom(Domain3D):
    def __init__(self,basin,transfer,**kwargs):
        self.comments,self.numbers,self.pos_au,self.cell_au,self.origin_au = transfer
        self.grid_shape = basin.shape
        self.indices = np.where(basin!=0)
        self.weights = basin[self.indices]

class ElectronCurrent(VectorField):        
#    def __init__(self,fn,**kwargs):
#        self = VectorField.__init__(self,fn)  
    pass


class ElectronicSystem():
    def __init__(self,fn,fn1,fn2,fn3,**kwargs):
        self.rho = ElectronDensity(fn=fn)
        self.j   = ElectronCurrent(fn1,fn2,fn3)
        self.rho.integral()
        if np.allclose(self.rho.origin_au,self.j.origin_au) and np.allclose(self.rho.cell_au,self.j.cell_au) and np.allclose(self.rho.pos_au,self.j.pos_au) and np.allclose(self.rho.numbers,self.j.numbers) and np.allclose(self.rho.voxel,self.j.voxel):
            pass
        else:
            raise Exception('ERROR: Density and Current Data is not consistent!')

    def grid(self):
        self.rho.grid()

    def pos_grid(self):
        self.rho.pos_grid()

    def crop(self,r,**kwargs):
        self.rho.crop(r)
        self.rho.integral()
        self.j.crop(r)

    def auto_crop(self,**kwargs):
        '''crop after threshold (default: ...)'''
        thresh=kwargs.get('thresh',self.rho.threshold)
#        isstate=kwargs.get('state',False)
#        if isstate: scale = 2*self.rho.data**2
        scale = self.rho.data
        a=np.amin(np.array(self.rho.data.shape)-np.argwhere(scale > thresh))
        b=np.amin(np.argwhere(scale > thresh))
        self.crop(min(a,b))


    def calculate_velocity_field(self,**kwargs): #check convetnion for states, use j so far
        lower_thresh=kwargs.get('lower_thresh',self.rho.threshold)
        upper_thresh=kwargs.get('upper_thresh',100000) #is this high enough?
#        isstate=kwargs.get('state',False)
        self.v = copy.deepcopy(self.j)
#        if isstate: scale = 2*self.rho.data**2 #restricted bot debugged
        scale = self.rho.data
        self.v.data = self.v.data/scale[None]*(scale>lower_thresh)[None]*(scale<upper_thresh)[None]


    def propagate_density(self,dt=8.0):
        '''dt in atomic units'''
        incr = np.array([self.rho.cell_au[0,0],self.rho.cell_au[1,1],self.rho.cell_au[2,2]])
        n_x = self.rho.data.shape[-3]
        n_y = self.rho.data.shape[-2]
        n_z = self.rho.data.shape[-1]
        xaxis = np.arange(0,n_x) 
        yaxis = np.arange(0,n_y) 
        zaxis = np.arange(0,n_z) 
        ind_grid = np.array(np.meshgrid(xaxis,yaxis,zaxis,indexing='ij'))

        rho2 = copy.deepcopy(self.rho)
        lower_thresh = self.rho.threshold/2
        upper_thresh = 100000

        #OPTION A #get nearest target point (no distribution of rho on multiple points, hope that is accurate enough
#        self.calculate_velocity_field(lower_thresh=lower_thresh)
#        v_field = self.v.data

        #OPTION B: use irrotational field to reduce noise
#        self.j.helmholtz_decomposition() 
#        scale = self.rho.data
#        v_field = self.j.irrotational_field/scale[None]*(scale>lower_thresh)[None]*(scale<upper_thresh)[None]
#        
#        target = ((np.round(v_field*dt/incr[:,None,None,None])+ind_grid)%np.array([n_x,n_y,n_z])[:,None,None,None]).astype(int)
#        ind = np.ravel_multi_index(target,self.rho.data.shape)
#        rho2.data = (self.rho.data.ravel()[ind]).reshape(self.rho.data.shape)

        #OPTIONS A and B are unstable!
        #OPTION C: Use div
        self.j.divergence_and_rotation()
        rho2.data -= self.j.div*dt
#        test = np.unravel_index(np.argsort(self.data.ravel())[::-1], self.data.shape)
#        rho2.data.fill(0.0)
#        rho2.data[target]=self.rho.data[point
#        rho2.data += self.j.div #needs some selection
        return rho2

    def read_nuclear_velocities(self,fn):
        '''Has to be in shape n_frames,n_atoms, 3'''
        self.nuc_vel_au,self.nuc_symbols,self.nuc_vel_comments = xyz.ReadTrajectory_BruteForce(fn)
#        self.nuc_vel_au = self.nuc_vel_au[0] #only first frame?
        if self.nuc_symbols != [constants.symbols[z-1] for z in self.rho.numbers]:
            raise Exception('ERROR: Nuclear velocity file does not match Electronic System!')

    def calculate_aim_differential_current(self):
        '''Map vector of nuclear velocity on atom domain'''
        self.v_diff = copy.deepcopy(self.v)
        for i in range(self.rho.n_atoms):
            field = self.rho.aim_atoms[i].map_vector(self.nuc_vel_au[0,i,:])
            self.v_diff.data-= field #self.rho.aim_atoms[i].map_vector(self.nuc_vel_au[0,i,:])
        self.v_diff.data = signal.medfilt(self.v_diff.data, kernel_size=3) #clean numerical noise 
        self.j_diff = copy.deepcopy(self.j)
        self.j_diff.data = self.v_diff.data*self.rho.data[np.newaxis]
    
#    def integrate_dt(self):
#        dt  = kwargs.get('dt_au',8)
#        iaf = kwargs.get('ia_flux',False)
#        aim = kwargs.get('aim',iaf)
#        self.rho_dt = self.propagate_density(dt=dt)
#        if aim: self.rho_dt.aim()
        

#class dividing surface
#class time evolution of surface
#class finite differences
#class SDF():

