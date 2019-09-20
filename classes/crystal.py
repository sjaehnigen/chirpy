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
#!/usr/bin/env python

import copy
import numpy as np
from ..physics import constants
from ..topology.symmetry import get_cell_vec

class UnitCell(): #maybe all-in-one class
    def __init__(self,cell_aa_deg,**kwargs): #**kwargs for named (dict), *args for unnamed
        '''Angle Convention: alpha=from c to b, beta=from a to c, gamma=from b to a'''
        if cell_aa_deg[ :3 ].sum() == 0.0: raise TypeError( 'ERROR: Zero Cell Size!' )
        self.cell_aa_deg = cell_aa_deg
        self.abc,self.albega = cell_aa_deg[:3],cell_aa_deg[3:]*np.pi/180.
        self.abc_unitcell = copy.deepcopy(self.abc)
    #ABC AND ALBEGA COULD ALSO BE TIME_DEPENDENT!

    def propagate(self, data, multiply, priority=(0, 1, 2)):
        '''data has to be XYZData or VibrationalModes object'''
        self.multiply = np.array(multiply)

        #Distinguish between XYZData and VibrationalModes
        if hasattr(data,'pos_aa'): 
            cart_vec_aa = get_cell_vec(self.cell_aa_deg, n_fields=data.n_fields, priority=priority)
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
            cart_vec_aa = get_cell_vec(self.cell_aa_deg, n_fields=3, priority=priority)
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
        # store cell_vec_aa?
        return new

