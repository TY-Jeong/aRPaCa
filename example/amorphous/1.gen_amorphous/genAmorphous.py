import sys
sys.path.append('/home/taeyoung/github/arpaca/src/amorphous')
from amorphous import *

genAmorphous(density=10.00, chem_formula="Hf34O68", outfile='POSCAR_Hf34O68')
genAmorphous(density=4.14, chem_formula="Hf34O68", outfile='POSCAR_Ti34O68')