#!/usr/bin/env python
# ------------------------------------------------------
#
#  ChirPy
#
#    A buoyant python package for analysing supramolecular
#    and electronic structure, chirality and dynamics.
#
#
#  Developers:
#    2010-2016  Arne Scherrer
#    since 2014 Sascha Jähnigen
#
#  https://hartree.chimie.ens.fr/sjaehnigen/chirpy.git
#
# ------------------------------------------------------


import argparse
from chirpy.create.gen_modes import *

def main():
    parser = argparse.ArgumentParser( description = "Read a supercell (xyz, pdb, ...) and print box information", formatter_class = argparse.ArgumentDefaultsHelpFormatter )
    parser.add_argument( "fn", help = "supercell (xyz, pdb, xvibs, ...)" )
#    parser.add_argument( "-cell_aa", nargs = 6, help = "Orthorhombic cell parametres a b c al be ga in angstrom/degree (default: atom spread).", default = None, type = float )
#    parser.add_argument( "-fn_topo", help = "Topology pdb file (optional)", default = None )
#    parser.add_argument("-get_mols", action="store_true", help="Find molecules in cell (slow for medium/large systems)")
    args = parser.parse_args()

#    nargs = {}
#    nargs[ 'cell_aa_deg' ] = args.cell_aa
#    nargs[ 'fn_topo' ] = args.fn_topo
#    if args.get_mols:
#        nargs['define_molecules'] =  args.get_mols
#
#    b = Solution.read( args.fn, **nargs )
#    b.print_info()

if __name__ == "__main__":
    main()


# Snippets

#       def load_localised_power_spectrum(self, fn_spec):
#           a = _np.loadtxt(fn_spec)
#           self.nu_cgs = a[:, 0]  # unit already cm-1 if coming from molsim
#           self.pow_loc = a[:, 1:].swapaxes(0, 1)  # unit?

