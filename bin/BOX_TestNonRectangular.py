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
from chirpy.classes.system import Supercell

def main():
    parser = argparse.ArgumentParser( description = "Read a supercell (xyz, pdb, ...) and print box information", formatter_class = argparse.ArgumentDefaultsHelpFormatter )
    parser.add_argument( "fn", help = "supercell (xyz, pdb, xvibs, ...)" )
    parser.add_argument( "-cell_aa", nargs = 6, help = "Orthorhombic cell parametres a b c al be ga in angstrom/degree (default: atom spread).", default = None, type = float )
    parser.add_argument( "-fn_topo", help = "Topology pdb file (optional)", default = None )
    args = parser.parse_args()

    nargs = {}
    #change to cell_aa_deg
    nargs[ 'cell_aa' ] = args.cell_aa
    nargs[ 'fn_topo' ] = args.fn_topo

    b = Supercell( args.fn, **nargs )
    b.install_molecular_origin_gauge()
    #b.XYZData._wrap_molecules(b.mol_map, b.cell_aa_deg)
    b.XYZData._wrap_atoms(b.cell_aa_deg)

    b.write( "test.pdb" )

if __name__ == "__main__":
    main()
