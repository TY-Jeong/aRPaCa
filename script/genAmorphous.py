## Arguments: [density], [chem_formula]
import sys
from arpaca.utils import genAmorphous

density, chem_formula = sys.argv[1], sys.argv[2]

genAmorphous(density=density, 
             chem_formula=chem_formula, 
             outfile=f'POSCAR_{chem_formula}')
