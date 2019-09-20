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

import argparse
import copy
from chirpy.classes import quantum
#imp.reload(quantum)

def main():
    parser=argparse.ArgumentParser(description="Crop Wavefunction after threshold to save disk space", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("fn",       help="coord file (xyz.pdb,xvibs,...)")
#    parser.add_argument("-f",       help="Output file name (default with appendix CROPPED)", default='out.xyz')
    args = parser.parse_args()

    #wfn_cube = 'C0-000001-S%02d.cube'
    fn = args.fn
    system = quantum.WannierFunction(fn=fn)
    system.auto_crop(thresh=1.0)
    
    out = copy.deepcopy(system)
    out.write(''.join(fn.split('.')[:-1])+'-CROPPED.'+fn.split('.')[-1])
    del out

if(__name__ == "__main__"):
    main()
