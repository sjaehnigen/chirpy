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
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f' , '--file',                 default=None)
    parser.add_argument('-c0', '--column0',  type=int,   default=None)
    parser.add_argument('-c1', '--column1',  type=int,   default=None)
    parser.add_argument('-w' , '--width',    type=float, default=None) #Halbwertsbreite
    parser.add_argument('-x0', '--x_start',  type=float, default=None)
    parser.add_argument('-x1', '--x_end',    type=float, default=None)
    parser.add_argument('-n',  '--n_samples',type=int,   default=100)
    parser.add_argument('--no_header', action='store_true', default=False)
    parser.add_argument('-o',  '--fn_out',               default='out.dat')
    parser.add_argument('--type',                        default='Lorentzian (Gaussian not implemented')
    args = parser.parse_args()

    for a in sorted(vars(args)):
        if getattr(args,a) is None: setattr(args,a,input(a+'?\n'))

    #only coloumns with header use zip_longest() to get all entries
    data = np.array([s[1-int(args.no_header):] for s in zip(*[l.strip().split() for l in open(args.file,'r').readlines()])])[[args.column0,args.column1]].astype(float)

    L = lambda x: sum([p[1] / ( 1 + ( ( p[0] -x ) / ( args.width / 2 ) )**2 ) for p  in data.swapaxes(0,1)])

    X = np.linspace(args.x_start,args.x_end,args.n_samples)

    with open(args.fn_out,'w') as f: f.write(''.join(["%12.6f %12.6f\n"%p for p in zip(X,map(L,X))]))

if __name__ == "__main__":
    main()
