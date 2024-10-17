import arpaca.elect_conduct.calculator as aec
import numpy as np

poscar = input('path of poscar: ')
trange = input('lists of temperatures: ') ## e.g. 300 400 500
trange = np.array(list(map(int, trange.split())))

kg_set = aec.GetKGSet(path_poscar=poscar, temp=trange)
