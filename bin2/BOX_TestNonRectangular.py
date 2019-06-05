#!/usr/bin/env python3

import argparse
import numpy as np 
from classes.system import Supercell
from classes.trajectory import XYZFrame
from generators.gen_box import Solution
from topology import symmetry

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
print( b.XYZData.pos_aa[0] )
print( b.cell_aa_deg )
symmetry._change_to_cell_basis( b.XYZData.pos_aa, b.cell_aa_deg )

