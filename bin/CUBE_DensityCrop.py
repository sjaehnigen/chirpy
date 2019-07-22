#!/usr/bin/env python

import argparse
import copy
from chirpy.classes import quantum

def main():
    parser=argparse.ArgumentParser(description="Crop Density after threshold in order to save disk space", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("fn",       help="coord file (xyz.pdb,xvibs,...)")
#    parser.add_argument("-f",       help="Output file name (default with appendix CROPPED)", default='out.xyz')
    args = parser.parse_args()

    #wfn_cube = 'C0-000001-S%02d.cube'
    fn = args.fn
    system = quantum.ElectronDensity(fn=fn)
    system.auto_crop(thresh=system.thresh/2)
    
    out = copy.deepcopy(system)
    out.write(''.join(fn.split('.')[:-1])+'-CROPPED.'+fn.split('.')[-1])
    del out

if(__name__ == "__main__"):
    main()
