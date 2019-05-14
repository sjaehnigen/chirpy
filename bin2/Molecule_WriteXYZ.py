#!/usr/bin/env python3

import argparse
import numpy as np 
from classes import system
from topology import mapping

def main():
    '''Unit Cell parametres are taken from fn1 if needed'''
    parser=argparse.ArgumentParser(description="Convert any supported input into XYZ format", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("fn", help="file (xyz.pdb,xvibs,...)")
    parser.add_argument("-center_coords", action='store_true', help="Center Coordinates in cell centre or at origin (default: false; box_aa parametre overrides default origin).", default=False)
    parser.add_argument("-cell_aa", nargs=6, help="Orthorhombic cell parametres a b c al be ga in angstrom/degree (default: 0 0 0 90 90 90).", default=[0.0,0.0,0.0,90.,90.,90.])
    parser.add_argument("-f", help="Output file name (standard: 'out.xyz')", default='out.xyz')
    args = parser.parse_args()
    args.cell_aa = np.array(args.cell_aa).astype(float)
    system.Molecule(**vars(args)).XYZData.write(args.f,fmt='xyz')


#def map_atoms_to_crystal(pos_aa,sym,box_aa,replica,vec_trans,pos_cryst,sym_cr):
#    def get_assign(pos,pos_ref,box):
#        dist_array = pos[:,None,:]-pos_ref[None,:,:]
#        dist_array-= np.around(dist_array/box)*box
#        return np.argmin(np.linalg.norm(dist_array,axis=-1),axis=1)
#    
#    assign=np.zeros(sym.shape).astype(int)
#    for s in np.unique(sym):
#        ind = sym==s
#        ind_cr = sym_cr==s
#        ass = get_assign(pos_aa[ind],pos_cryst[ind_cr],box_aa)
#        assign[ind]=np.arange(len(sym_cr))[ind_cr][ass]
#    return assign
#    data, symbols, comments = xyz.ReadTrajectory_Smart(fn_xyz)
#    n_atoms  = np.shape(data)[1]
#    data += 0.5*cell_aa[None,None,:]-centre[None,None,:]
##    n_frames = np.shape(data)[0]
##    n_fields = np.shape(data)[2]
##    print(data)
##    print(geo_dat)
#    data    = WrapAtomPBC(data,cell_aa)
##    data    = WrapMoleculePBC(data,cell_aa)
##    data = data[0,:,0:3] #Up to now only the first frame is considered, later extend script for entire trajectory by looping ove frames
#    print('Results in %s.' % fn_out)
#    WriteXYZFile(fn_out, data, symbols, comments, append=False)

if(__name__ == "__main__"):
    main()
