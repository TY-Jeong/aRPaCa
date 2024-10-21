## USAGE : python removeAtom.py {POSCAR} {specie of atom} {number of atom}}
## Example : python removeAtom.py POSCAR O 2 : it will eliminate two O atoms randomly.

import sys
import random
from pymatgen.core import Structure

filename, target, num = sys.argv[1], sys.argv[2], sys.argv[3]
struc = Structure.from_file(filename)

label = []
for i, atom in enumerate(struc):
        if str(struc[i].specie) == target:
                label += [i]

label_removed = random.sample(label, int(num))
label_removed.sort(reverse=True)

for i in label_removed:
        del struc[i]

struc.to(filename='POSCAR_vac', fmt='poscar')
print("POSCAR_vac was generated.")
