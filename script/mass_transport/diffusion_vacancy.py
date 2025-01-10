#!/usr/bin/env python
import os
import sys
import argparse
import platform
from colorama import Fore
from arpaca.mass_transport.inout import DataInfo
from arpaca.mass_transport.einstein import *
from arpaca.mass_transport.trajectory import *

BOLD = '\033[1m'
CYAN = '\033[36m'
MAGENTA = '\033[35m'
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

# adjust output type for window environment
if platform.system()=="Windows":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

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
params = Parameter(data=data,
                   interval=args.interval,
                   poscar_lattice=args.lattice,
                   symbol=args.symbol,
                   rmax=3.0,
                   tol=1e-3,
                   tolerance=1e-3,
                   verbose=True)
params.save_figures()