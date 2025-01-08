#!/usr/bin/env python
import os
import sys
import argparse
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

# arguments for MSD
parser.add_argument('symbol',
                    type=str,
                    help='symbol of moving atom')
parser.add_argument('t_width',
                    type=float,
                    help='x-range in msd plot. unit: ps')
parser.add_argument('--skip',
                    type=float,
                    default=0,
                    help='steps to be skipped. unit: ps (default: 0)')
parser.add_argument('--start',
                    type=float,
                    default=1,
                    help='initial time used for linear fit. unit: ps (default: 1)')

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

msd = MSD(data=data,
          symbol=args.symbol,
          tmax=args.t_width,
          skip=args.skip,
          start=args.start)

