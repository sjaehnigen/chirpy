#!/usr/bin/env python

from chirpy.create.supercell import Solution


sys = Solution( 
                solvent = 'ACN-d3.xyz',
                solutes = [ 'S-MPAA.xyz' ],
                c_mol_L = [ 0.3 ],
                rho_g_cm3 = 0.844,
              )

sys._fill_box(sort=True)
#mv topology.pdb S-MPAA_ACN-d3_03M_topology.pdb
