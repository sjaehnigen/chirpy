#!/usr/bin/env python3

import argparse
import numpy as np
from generators.gen_box import Solution

parser = argparse.ArgumentParser( description = "Read a supercell (xyz, pdb, ...) and print box information", formatter_class = argparse.ArgumentDefaultsHelpFormatter )
parser.add_argument( "fn", help = "supercell (xyz, pdb, xvibs, ...)" )
parser.add_argument( "-cell_aa", nargs = 6, help = "Orthorhombic cell parametres a b c al be ga in angstrom/degree (default: atom spread).", default = None, type = float )
parser.add_argument( "-fn_topo", help = "Topology pdb file (optional)", default = None )
args = parser.parse_args()

if args.cell_aa is not None and not np.allclose( np.array( args.cell_aa )[ 3: ], np.ones( ( 3 ) ) * 90. ):
    raise NotImplementedError( 'Only orthorhombic cells are supported!\n' )

nargs = {}
nargs[ 'cell_aa_deg' ] = args.cell_aa
nargs[ 'fn_topo' ] = args.fn_topo

b = Solution.read( args.fn, **nargs )
b.print_info()

