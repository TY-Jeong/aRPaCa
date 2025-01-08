import os
import sys
import numpy as np
import argparse
# from arpaca.utils import extract_force

def extract_force(file_in, 
                  file_out='FORCE'):
    """
    extract force profile from vasprun.xml
    """
    # read vasprun.xml
    with open(file_in, 'r') as f:
        lines = [s.strip() for s in f]
    
    # system info
    nsw, num_atoms = None, None
    for line in lines:
        if 'NSW' in line:
            nsw = int(line.split('>')[-2].split('<')[0])
        if '<atoms>' in line:
            num_atoms = int(line.split()[1])
        if nsw and num_atoms:
            break
    
    # save forces
    step = 0
    forces = np.zeros((nsw, num_atoms, 3))
    for i, line in enumerate(lines):
        if 'forces' in line:
            force = [list(map(float, s.split()[1:4])) for s in lines[i+1:i+1+num_atoms]]
            force = np.array(force)
            forces[step] = force
            step += 1

    # write out_file
    with open(file_out, 'w') as f:
        for i, force in enumerate(forces, start=1):
            f.write(f"Iteration {i}\n")
            for fx, fy, fz in force:
                fx = str(fx).rjust(12)
                fy = str(fy).rjust(12)
                fz = str(fz).rjust(12)
                f.write(f"{fx} {fy} {fz}\n")       

parser = argparse.ArgumentParser(
    description='aRPaCa: ab initio RRAM Parameter Calculator',
    formatter_class=argparse.RawTextHelpFormatter
    )

parser.add_argument('-in', '--input',
                    default='vasprun.xml',
                    help='path to vasprun.xml file. (default: vasprun.xml)')
parser.add_argument('-out', '--output',
                    default='FORCE',
                    help='name of output file. (default: FORCE)')

args = parser.parse_args()

extract_force(file_in=args.input, file_out=args.output)
print(f"{args.output} is created.")