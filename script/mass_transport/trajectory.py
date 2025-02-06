#!/usr/bin/env python
import os
import sys
import argparse
from colorama import Fore
from arpaca.mass_transport.inout import DataInfo
from arpaca.mass_transport.trajectory import Parameter

BOLD = '\033[1m'
CYAN = '\033[36m'
MAGENTA = '\033[35m'
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

class EffectiveDiffusionParameter:
    def __init__(self, 
                 data, 
                 interval, 
                 poscar_lattice, 
                 symbol,
                 file_out='parameter.txt', 
                 rmax=3.0,
                 tol=1e-3,
                 tolerance=1e-3,
                 verbose=True):
        self.data = data
        self.interval = interval
        self.poscar_lattice = poscar_lattice
        self.symbol = symbol
        self.file_out = file_out
        self.rmax = rmax
        self.tol = tol
        self.tolerance = tolerance
        self.verbose = verbose
        
        with open(self.file_out, 'w', encoding='utf-8') as f:
            original_stdout = sys.stdout
            sys.stdout = f
            try:
                params = Parameter(data=self.data,
                                   interval=self.interval,
                                   poscar_lattice=self.poscar_lattice,
                                   symbol=self.symbol,
                                   rmax=self.rmax,
                                   tol=self.tol,
                                   tolerance=self.tolerance,
                                   verbose=self.verbose)
            finally:
                sys.stdout = original_stdout  
        params.save_figures()
        
        print(f"{self.file_out} is created.")
        print("D_rand.png is created.")
        print("f_cor.png is created.")
        print("tau.png is created.")



parser = argparse.ArgumentParser(
    description='aRPaCa: ab initio RRAM Parameter Calculator',
    formatter_class=argparse.RawTextHelpFormatter
    )

# arguments for DataInfo
parser.add_argument('-p1', '--prefix1',
                    default='traj',
                    help='name of outer directory (default: traj)')
parser.add_argument('-p2', '--prefix2',
                    default='traj',
                    help='prefix of inner directories. e.g.{prefix2}.{temp}K (default: traj)')

# arguments for Parameter
parser.add_argument('symbol',
                    type=str,
                    help='symbol of moving atom')
parser.add_argument('interval',
                    type=float,
                    help='time interval for averaging in ps')
parser.add_argument('-l', '--lattice',
                    type=str,
                    default='POSCAR_LATTICE',
                    help='lattice file in POSCAR format (default: POSCAR_LATICE)')

args = parser.parse_args()

# print arguments
print(f'{CYAN}{BOLD}VacHopPy is in progress{RESET}')
with open('arg.txt', 'w') as f:
    print(f'{GREEN}{BOLD}Arguments and Values :{RESET}')
    f.write('Arguments and Values :\n')
    for arg, value in vars(args).items():
        print(f'    {arg} = {value}')
        f.write(f'    {arg} = {value}\n')
print('')

data = DataInfo(prefix1=args.prefix1,
                prefix2=args.prefix2,
                verbose=True)

params = EffectiveDiffusionParameter(data,
                                     args.interval,
                                     args.lattice,
                                     args.symbol,
                                     rmax=3.0,
                                     tol=1e-3,
                                     tolerance=1e-3,
                                     verbose=True)