#!/usr/bin/python3
#import unittest
#import logging
#import filecmp
#import types
#import filecmp
#
#from lib import debug
#from collections import OrderedDic

import sys, os, copy
import numpy as np
from fileio import cpmd,xyz
from lib import constants


class UnitCell(): #maybe all-in-one class
    def __init__(self,cell_aa_deg,**kwargs): #**kwargs for named (dict), *args for unnamed
        '''Angle Convention: alpha=from c to b, beta=from a to c, gamma=from b to a'''
        self.abc,self.albega = cell_aa_deg[:3],cell_aa_deg[3:]*np.pi/180.
        self.abc_unitcell = copy.deepcopy(self.abc)
    #ABC AND ALBEGA COULD ALSO BE TIME_DEPENDENT!
        
    def propagate(self,data,multiply,priority=(0,1,2)):
        '''data has to be XYZData or VibrationalModes object'''
        self.multiply = np.array(multiply)
        def initialise_cart_vec_aa(n_fields,priority):
            cart_vec_aa = np.zeros((3,n_fields))
            v0,v1,v2 = priority
            cart_vec_aa[v0,v0] = self.abc_unitcell[v0] 
            cart_vec_aa[v1,v1] = self.abc_unitcell[v1]*np.sin(self.albega[(3-v0-v1)])
            cart_vec_aa[v1,v0] = self.abc_unitcell[v1]*np.cos(self.albega[(3-v0-v1)])
            cart_vec_aa[v2,v2] = self.abc_unitcell[v2]*np.sin(self.albega[(3-v0-v2)])*np.sin(self.albega[(3-v1-v2)])
            cart_vec_aa[v2,v0] = self.abc_unitcell[v2]*np.cos(self.albega[(3-v0-v2)])            
            cart_vec_aa[v2,v1] = self.abc_unitcell[v2]*np.cos(self.albega[(3-v1-v2)])                    
#            cart_vec_aa[v,v      ] -= (cart_vec_aa[v,v]-self.abc[v]*np.abs(np.sin(self.albega[(v-1)  ])))*(np.linalg.norm(cart_vec_aa[(v+1)%3])!=0)                        
#            cart_vec_aa[v,(v+1)%3] += self.abc[v]*np.cos(self.albega[(v-1)  ])*(cart_vec_aa[(v+1)%3]!=0)                        
#            cart_vec_aa[v,v      ] -= (cart_vec_aa[v]-cart_vec_aa[v]*np.abs(np.sin(self.albega[(v+1)%3])))*(np.linalg.norm(cart_vec_aa[(v-1)])!=0)           
#            cart_vec_aa[v,v-1    ] += cart_vec_aa[v,v]*np.cos(self.albega[(v+1)%3])*(cart_vec_aa[(v-1)  ]!=0)
            return cart_vec_aa
        
        #Distinguish between XYZData and VibrationalModes
        if hasattr(data,'pos_aa'):
            cart_vec_aa = initialise_cart_vec_aa(data.n_fields,priority)
            data.axis_pointer=1
            new = copy.deepcopy(data)            
            for iz,z in enumerate(multiply):
                tmp = copy.deepcopy(new)            
                for iiz in range(z-1):
                    tmp.pos_aa[:,:,:] += cart_vec_aa[np.newaxis,np.newaxis,iz]
                    new += tmp
#            new.data[:,:,:3] = np.remainder(new.data[:,:,:3],self.abc*self.multiply)
            new._sort() #Maybe rise it to general standard to only use sorted data (makes life much easier). Fo rnow only here

        elif hasattr(data,'eival_cgs'):
            cart_vec_aa = initialise_cart_vec_aa(3,priority)
            new = copy.deepcopy(data)            
            for iz,z in enumerate(multiply):
                tmp = copy.deepcopy(new)            
                for iiz in range(z-1):
                    tmp.pos_au[:,:] += cart_vec_aa[np.newaxis,iz]*constants.l_aa2au
                    new += tmp
#            new.pos_au[:,:] = np.remainder(new.pos_au[:,:],self.abc*self.multiply*constants.l_aa2au)
            new._sort()
        self.abc = self.abc_unitcell*self.multiply
        return new

