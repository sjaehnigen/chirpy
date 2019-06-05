#!/usr/bin/python3
#import unittest
#import logging
#import filecmp
#import types
#import filecmp
#
#from lib import debug
#from collections import OrderedDic

import copy
import numpy as np
from physics import constants
from topology.symmetry import get_cell_vec

class UnitCell(): #maybe all-in-one class
    def __init__(self,cell_aa_deg,**kwargs): #**kwargs for named (dict), *args for unnamed
        '''Angle Convention: alpha=from c to b, beta=from a to c, gamma=from b to a'''
        if cell_aa_deg[ :3 ].sum() == 0.0: raise TypeError( 'ERROR: Zero Cell Size!' )
        self.abc,self.albega = cell_aa_deg[:3],cell_aa_deg[3:]*np.pi/180.
        self.abc_unitcell = copy.deepcopy(self.abc)
    #ABC AND ALBEGA COULD ALSO BE TIME_DEPENDENT!

    def propagate(self, data, multiply, priority=(0, 1, 2)):
        '''data has to be XYZData or VibrationalModes object'''
        self.multiply = np.array(multiply)
#        def initialise_cart_vec_aa(n_fields, priority):
#            # priority defines the alignment of non-rectangular objects in cartesian space, be convention z axis is the
#            # odd one (i.e. not aligned), and cell vectors are calculated accordingly (e.g. in VMD); here, this
#            # corresponds to priority (0,1,2). 
#            # Any deviation (e.g. z-axis IS aligned, alpha > 90°, ...) can be taken account for by adjusting priority,
#            # and the correct cell vectors are calculated. However, VMD or other programmes may not still calculate the
#            # wrong cell vectors. Hence, ...
#            # priority should always be (0,1,2) and symmetry conventions be used (e.g. for monoclinic cells: beta is
#            # the angle > 90°; CPMD wants alpha to be >90° but this is wrong and CELL VECTORS should be used instead)
#            cart_vec_aa = np.zeros((3, n_fields))
#            v0, v1, v2 = priority
#            cart_vec_aa[v0, v0] = self.abc_unitcell[v0]
#            cart_vec_aa[v1, v1] = self.abc_unitcell[v1] * np.sin(self.albega[(3 - v0 - v1)])
#            cart_vec_aa[v1, v0] = self.abc_unitcell[v1] * np.cos(self.albega[(3 - v0 - v1)])
#            cart_vec_aa[v2, v2] = self.abc_unitcell[v2] * np.sin(self.albega[(3 - v0 - v2)]) * np.sin(self.albega[(3 - v1 - v2)])
#            cart_vec_aa[v2, v0] = self.abc_unitcell[v2] * np.cos(self.albega[(3 - v0 - v2)])
#            cart_vec_aa[v2, v1] = self.abc_unitcell[v2] * np.cos(self.albega[(3 - v1 - v2)])
##            cart_vec_aa[v,v      ] -= (cart_vec_aa[v,v]-self.abc[v]*np.abs(np.sin(self.albega[(v-1)  ])))*(np.linalg.norm(cart_vec_aa[(v+1)%3])!=0)                        
##            cart_vec_aa[v,(v+1)%3] += self.abc[v]*np.cos(self.albega[(v-1)  ])*(cart_vec_aa[(v+1)%3]!=0)                        
##            cart_vec_aa[v,v      ] -= (cart_vec_aa[v]-cart_vec_aa[v]*np.abs(np.sin(self.albega[(v+1)%3])))*(np.linalg.norm(cart_vec_aa[(v-1)])!=0)           
##            cart_vec_aa[v,v-1    ] += cart_vec_aa[v,v]*np.cos(self.albega[(v+1)%3])*(cart_vec_aa[(v-1)  ]!=0)
#            return cart_vec_aa

        #Distinguish between XYZData and VibrationalModes
        if hasattr(data,'pos_aa'): 
            cart_vec_aa = get_cell_vec(cell_aa_deg, n_fields=data.n_fields, priority=priority)
            data.axis_pointer=1
            new = copy.deepcopy(data)            
            for iz,z in enumerate(multiply):
                tmp = copy.deepcopy(new)            
                for iiz in range(z-1):
                    tmp.data[:, :, :3] += cart_vec_aa[None, None, iz]
                    #tmp.pos_aa[:,:,:] += cart_vec_aa[None, None, iz]
                    new += tmp
#            new.data[:,:,:3] = np.remainder(new.data[:,:,:3],self.abc*self.multiply)
            new._sort() #Maybe rise it to general standard to only use sorted data (makes life much easier). Fo rnow only here

        elif hasattr(data,'eival_cgs'):
            cart_vec_aa = get_cell_vec(cell_aa_deg, n_fields=3, priority=priority)
            new = copy.deepcopy(data)            
            for iz,z in enumerate(multiply):
                tmp = copy.deepcopy(new)            
                for iiz in range(z-1):
                    tmp.pos_au[:,:] += cart_vec_aa[np.newaxis,iz]*constants.l_aa2au
                    new += tmp
#            new.pos_au[:,:] = np.remainder(new.pos_au[:,:],self.abc*self.multiply*constants.l_aa2au)
            new._sort()
        #new.abc = self.abc_unitcell*self.multiply
        new.abc = self.abc * self.multiply
        return new

