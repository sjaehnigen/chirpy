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


import argparse
from chirpy.classes import system
from chirpy.topology import mapping

def main():
    '''Unit Cell parametres are taken from fn1 if needed'''
    parser=argparse.ArgumentParser(description="Wrap atoms from XYZ file into PBC box", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("fn1", help="file 1 (xy,pdb,xvibs,...)")
    parser.add_argument("fn2", help="file 2 (xy,pdb,xvibs,...)")
    args = parser.parse_args()
    fn1 = args.fn1
    fn2 = args.fn2
    mol1 = system.Supercell(fn1)
    mol2 = system.Supercell(fn2)
    assign = mapping.map_atoms_by_coordinates(mol1,mol2)
    outbuf = ['%35d -------> %3d'%(i+1,j+1) for i,j in enumerate(assign[0])] # this module uses only frames 0
    print(    '%35s          %s'%(fn1,fn2))
    print('\n'.join(outbuf))


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
