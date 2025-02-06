import os
import sys
import argparse
from arpaca.mass_transport.trajectory import PostProcess

BOLD = '\033[1m'
CYAN = '\033[36m'
MAGENTA = '\033[35m'
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

class Post_EffectiveDiffusionParameter:
    def __init__(self,
                 file_params='parameter.txt',
                 file_neb='neb.csv',
                 file_out='postprocess.txt',
                 verbose=True):
        self.file_params = file_params
        self.file_neb = file_neb
        self.file_out = file_out
        self.verbose = verbose
        with open(self.file_out, 'w', encoding='utf-8') as f:
            original_stdout = sys.stdout
            sys.stdout = f
            try:
                postprocess = PostProcess(file_params=self.file_params,
                                          file_neb=self.file_neb,
                                          verbose=self.verbose)
            finally:
                sys.stdout = original_stdout
        print(f"{self.file_out} is created.")

parser = argparse.ArgumentParser(
    description='aRPaCa: ab initio RRAM Parameter Calculator',
    formatter_class=argparse.RawTextHelpFormatter
    )

# arguments for Parameter
parser.add_argument('-p', '--parameter',
                    type=str,
                    default='parameter.txt',
                    help='parameter.txt file (default: parameter.txt)')
parser.add_argument('-n', '--neb',
                    type=str,
                    default='neb.csv',
                    help='neb.csv file containing hopping barriers for each path (default: neb.csv)')
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

post = Post_EffectiveDiffusionParameter(file_params=args.parameter,
                                        file_neb=args.neb)