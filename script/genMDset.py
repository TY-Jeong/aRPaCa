import sys
import numpy as np
from arpaca.utils import *

path_poscar = input('directory containing ensembles (str) : ')
temp_str = input('temperature range in K (start end step) : ').split()
temp = np.arange(int(temp_str[0]), int(temp_str[1])+1, int(temp_str[2]))
potcar = input('kind of potcar (lda or pbe) : ')

if potcar not in ['lda', 'pbe']:
    print('  Error! potcar shoud be lda or pbe')
    sys.exit(0)

nsw = int(input('iteration number (int): '))
potim = float(input('time step in fs (float): '))
charge = float(input('charge state of the system (float): '))



getMDset(path_poscar=path_poscar,
         temp=temp,
         potcar=potcar,
         nsw=nsw,
         potim=potim,
         charge=charge)
