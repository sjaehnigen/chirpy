#!/usr/bin/env python3
import numpy as np
import copy
import sys
import tempfile
from classes.volume import ScalarField
from scipy.interpolate import griddata
from scipy.integrate import simps
from reader.volume import cubeReader
from fileio.cube import WriteCubeFile #Replace it ?

eijk = np.zeros((3, 3, 3))
eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1

class Domain3D():
    '''This is not a derived scalar field class since it would be a waste of memory.'''
    def __init__(self,shape,indices,weights,**kwargs):
        self.grid_shape = shape
        self.indices = indices
        self.weights = weights

    #def map_scalar(self,s_field,s):
    #    if not np.allclose(s_field.shape,self.grid_shape):
    #        raise Exception('ERROR: Given scalar field is not compatible with AIM topology')

    def map_vector(self,v3):
        n_x,n_y,n_z = self.grid_shape
        v3_field = np.zeros((3,n_x,n_y,n_z))
        ind = self.indices
        v3_field[:,ind[0],ind[1],ind[2]] = self.weights[np.newaxis,:]*v3[:,np.newaxis]
        return v3_field

    #def integrate_vector_field():
        #if not np.allclose(v_field.shape[0],self.grid_shape):
        #    raise Exception('ERROR: Given vector field is not compatible with AIM topology')
    def integrate_volume(self,f):
        #return simps(f(self.indices)*self.weights)
        return np.sum(f(self.indices)*self.weights,axis=0)

#    def integrate_boundary():

    def expand(self):
        data = np.zeros(self.grid_shape)
        data[self.indices] = self.weights
        return data

    def write(self,fn,**kwargs):
        ScalarField.from_domain(self,**vars(self)).write(fn,**kwargs) 
        

class Domain2D():
    def __init__(self):
        pass


class DomainSet():
    def __init__(self,domains,**kwargs):
        '''domains is a list of domains that must all be consistent'''
#        self.origin_au = kwargs.get('origin_au',np.array([0.0,0.0,0.0]))
#        self.cell_au   = kwargs.get('cell_au',np.empty((0)))
#        if any([self.cell_au.size==0]): raise Exception('ERROR: Please give cell_au!')
#        self.voxel     = np.dot(self.cell_au[0],np.cross(self.cell_au[1],self.cell_au[2]))
        self.n_domains = len(domains)
        self.shape     = domains[0].grid_shape
        self.domains   = domains

    def boundaries(self):
        pass #Calculate domain boundaries/ias

    @staticmethod
    def _finite_differences(grid1,grid2,shape):
        if grid1.shape!=grid2.shape: raise Exception('ERROR: Different grid shapes!')
        FD = np.zeros((6,)+grid1.shape)

        FD[0] = np.roll(grid2,-1,axis=0)-grid1
        FD[1] = np.roll(grid2,+1,axis=0)-grid1
        FD[2] = np.roll(grid2,-1,axis=1)-grid1
        FD[3] = np.roll(grid2,+1,axis=1)-grid1
        FD[4] = np.roll(grid2,-1,axis=2)-grid1
        FD[5] = np.roll(grid2,+1,axis=2)-grid1
        
        self.fd = np.zeros((self.n_domains,6,)+self.shape)

        for i_d,(d1,d2) in enumerate(zip(domains1,domains2)):
            #if not all([isinstance(d1,(Domain3d,Domain2D)),isinstance(d2,(Domain3d,Domain2D))]): raise Exception('ERROR: Lists contain unknown domain types!')
            #print((type(d1).__bases__[0]))
            #print(issubclass(type(d1).__bases__[0],classes.domain.Domain3D))
            #if not all([issubclass(type(d1).__bases__[0],(Domain3D)),issubclass(type(d2).__bases__[0],(Domain3D))]): raise Exception('ERROR: Lists contain unknown domain types!',type(d1),type(d2))
            if d1.grid_shape != d2.grid_shape: raise Exception('ERROR: Domain differ in their respective grid shapes!')
            tmp1,tmp2 = d1.expand(),d2.expand()
            #this is a gradient
            self.fd[i_d,0] = np.roll(tmp2,-1,axis=0)-tmp1
            self.fd[i_d,1] = np.roll(tmp2,+1,axis=0)-tmp1
            self.fd[i_d,2] = np.roll(tmp2,-1,axis=1)-tmp1
            self.fd[i_d,3] = np.roll(tmp2,+1,axis=1)-tmp1
            self.fd[i_d,4] = np.roll(tmp2,-1,axis=2)-tmp1
            self.fd[i_d,5] = np.roll(tmp2,+1,axis=2)-tmp1


class FD_Domain():
    def __init__(self,domains1,domains2,**kwargs):
        '''domains1/2 are a list of domains that must all be consistent'''
        self.origin_au = kwargs.get('origin_au',np.array([0.0,0.0,0.0]))
        self.cell_au   = kwargs.get('cell_au',np.empty((0)))
        if any([self.cell_au.size==0]): raise Exception('ERROR: Please give cell_au!')

        self.voxel     = np.dot(self.cell_au[0],np.cross(self.cell_au[1],self.cell_au[2]))
        self.n_domains = len(domains1)
        self.shape = domains1[0].grid_shape
        self._tmp = kwargs.get('use_tempfile',False)
        if self.n_domains != len(domains2): raise Exception('ERROR: The two lists contain different numbers of domains!')
        self.fd = np.zeros((self.n_domains,6,)+self.shape)

        for i_d,(d1,d2) in enumerate(zip(domains1,domains2)):
            #if not all([isinstance(d1,(Domain3d,Domain2D)),isinstance(d2,(Domain3d,Domain2D))]): raise Exception('ERROR: Lists contain unknown domain types!')
            #print((type(d1).__bases__[0]))
            #print(issubclass(type(d1).__bases__[0],classes.domain.Domain3D))
            #if not all([issubclass(type(d1).__bases__[0],(Domain3D)),issubclass(type(d2).__bases__[0],(Domain3D))]): raise Exception('ERROR: Lists contain unknown domain types!',type(d1),type(d2))
            if d1.grid_shape != d2.grid_shape: raise Exception('ERROR: Domain differ in their respective grid shapes!')
            tmp1,tmp2 = d1.expand(),d2.expand()
            #this is a gradient
            self.fd[i_d,0] = np.roll(tmp2,-1,axis=0)-tmp1
            self.fd[i_d,1] = np.roll(tmp2,+1,axis=0)-tmp1
            self.fd[i_d,2] = np.roll(tmp2,-1,axis=1)-tmp1
            self.fd[i_d,3] = np.roll(tmp2,+1,axis=1)-tmp1
            self.fd[i_d,4] = np.roll(tmp2,-1,axis=2)-tmp1
            self.fd[i_d,5] = np.roll(tmp2,+1,axis=2)-tmp1

            

#    def _init_tempfile(self,name,shape):
#        try: fn = tempfile.NamedTemporaryFile(dir='/scratch/ssd/')
#        except (NotADirectoryError,FileNotFoundError): fn = tempfile.NamedTemporaryFile(dir='/tmp/')
#        setattr(self,name,np.memmap(fn,dtype='float64',mode='w+',shape=shape))

    @staticmethod
    def normalise(v,**kwargs): #export it (no class method)
        axis = kwargs.get('axis',0)
        norm = kwargs.get('norm',np.linalg.norm(v,axis=axis))
        v_dir = v/norm[None] #only valid for axis==0
        v_dir[np.isnan(v_dir)]=0.0
        return v_dir,norm


    def map_flux_density(self,j):
        self.gain = np.zeros((self.n_domains,3,)+self.shape)
        self.loss = np.zeros((self.n_domains,3,)+self.shape)

        j_dir,j_norm = self.normalise(j)

        for i_d in range(self.n_domains):    
        ###if j_x/_y/j_z >= my_thresh
            #these are the scalar products
            self.gain[i_d,0] += j_dir[0]*(j_dir[0]>0)*self.fd[i_d,0]*(self.fd[i_d,0]>0)
            self.gain[i_d,0] += j_dir[0]*(j_dir[0]<0)*self.fd[i_d,1]*(self.fd[i_d,1]>0)
            self.gain[i_d,1] += j_dir[1]*(j_dir[1]>0)*self.fd[i_d,2]*(self.fd[i_d,2]>0)
            self.gain[i_d,1] += j_dir[1]*(j_dir[1]<0)*self.fd[i_d,3]*(self.fd[i_d,3]>0)
            self.gain[i_d,2] += j_dir[2]*(j_dir[2]>0)*self.fd[i_d,4]*(self.fd[i_d,4]>0)
            self.gain[i_d,2] += j_dir[2]*(j_dir[2]<0)*self.fd[i_d,5]*(self.fd[i_d,5]>0)

            self.loss[i_d,0] += j_dir[0]*(j_dir[0]>0)*self.fd[i_d,0]*(self.fd[i_d,0]<0)
            self.loss[i_d,0] += j_dir[0]*(j_dir[0]<0)*self.fd[i_d,1]*(self.fd[i_d,1]<0)
            self.loss[i_d,1] += j_dir[1]*(j_dir[1]>0)*self.fd[i_d,2]*(self.fd[i_d,2]<0)
            self.loss[i_d,1] += j_dir[1]*(j_dir[1]<0)*self.fd[i_d,3]*(self.fd[i_d,3]<0)
            self.loss[i_d,2] += j_dir[2]*(j_dir[2]>0)*self.fd[i_d,4]*(self.fd[i_d,4]<0)
            self.loss[i_d,2] += j_dir[2]*(j_dir[2]<0)*self.fd[i_d,5]*(self.fd[i_d,5]<0)    
        self.gain *=  j_norm[None,None]
        self.loss *=  j_norm[None,None]


    def integrate_domains(self,**kwargs):
        flux = kwargs.get('flux',False)

        if flux:
            self.map_flux_density(kwargs.get('j'))
            

            try: fn = tempfile.TemporaryFile(dir='/scratch/ssd/')
            except (NotADirectoryError,FileNotFoundError): fn = tempfile.TemporaryFile(dir='/tmp/')
            transfer = np.memmap(fn,dtype='float64',mode='w+',shape=(self.n_domains,self.n_domains,3,)+self.shape)

            j_gain = self.gain.sum(axis=0)
            j_loss = self.loss.sum(axis=0)
            if not np.allclose(j_gain,-j_loss,atol=1.E-3): print('WARNING: Domain gain/loss unbalanced!')

            w_gain,j_gain = self.normalise(self.gain,norm=j_gain)
            w_loss,j_loss = self.normalise(self.loss,norm=j_loss)

            transfer[:,:] = w_gain[:,None]*w_loss[None,:]*j_gain[None,None]
            transfer[:,:] -= w_loss[:,None]*w_gain[None,:]*j_loss[None,None]
            intg = np.zeros((self.n_domains,self.n_domains,))
            #for i_d in range(self.n_domains):
                #for j_d in range(self.n_domains):
                    #data = self.fd[i_d,j_d]
                    #intg[i_d,j_d] = ScalarField(fmt='manual',data=data,**vars(self)).integral()
            intg = transfer.sum(axis=-1).sum(axis=-1).sum(axis=-1).sum(axis=-1)
            print(intg)

        else: #not debugged
            for i_d in range(self.n_domains):
                data = self.fd[i_d]
                intg = ScalarField(fmt='manual',data=data.sum(axis=0),**vars(self)).integral()
                print(intg)
#            tmp = 
#        if self._tmp: 


#            self._init_tempfile('gain',(self.n_domains,3,)+self.shape)
#            self._init_tempfile('transfer_gain',(self.n_domains,self.n_domains,3,)+self.shape)
#            self._init_tempfile('loss',(self.n_domains,3,)+self.shape)
#            self._init_tempfile('transfer_loss',(self.n_domains,self.n_domains,3,)+self.shape)
#            self.gain.fill(0.0) #unnecessary?
#            self.loss.fill(0.0) #unnecessary?
#        else: 
#            self.gain = np.zeros((self.n_domains,3,)+self.shape)
#            self.transfer_gain = np.zeros((self.n_domains,self.n_domains,3,)+self.shape)
#            self.loss = np.zeros((self.n_domains,3,)+self.shape)
#            self.transfer_loss = np.zeros((self.n_domains,self.n_domains,3,)+self.shape)
#        j_gain = self.gain.sum(axis=0)
#        j_loss = self.loss.sum(axis=0)
#        if not np.allclose(j_gain,-j_loss,atol=1.E-3): print('WARNING: Domain gain/loss unbalanced!')
#
#        w_gain,j_gain = self.normalise(self.gain,norm=j_gain)
#        w_loss,j_loss = self.normalise(self.loss,norm=j_loss)
#    
#        self.transfer_gain = w_gain[:,None]*w_loss[None,:]*j_gain[None,None]
#        self.transfer_loss = w_loss[:,None]*w_gain[None,:]*j_loss[None,None]
#self.n_electrons = self.voxel*simps(simps(simps(self.data)))





####get target point    
####for all atoms: Calculate balance: Transfer(point1,point2,atom_i) = (w_aim2(atom_i,point2)-w_aim1(atom_i,point1))*j_x
#
##del buf1, buf2
